#!/usr/bin/env python3
"""Launch the multi-label scoring run for the designed x measured matrix.

2,304 judge calls: 96 scenarios x 8 principles x 3 source models, one judge.
No response is generated -- the task uses the strict pregenerated solver over
archived baseline responses -- so every dollar here is judge tokens.

What this does that a bare `inspect eval` loop does not:

1. **Prints a live cost estimate** from prompt sizes measured off the built
   datasets and an output length calibrated on the 287 gpt-5.1 judge calls the
   reported runs already made against these scenarios. Informational only:
   there is deliberately no balance or spend gate in this script. The account
   is capped on the OpenRouter side, and the in-repo guard this replaced
   aborted two launches while preventing zero overspends (removed in c574c2a).
2. **Stops rather than substitutes if gpt-5.1 is not served.** It is the only
   main-run ensemble judge that is not among the three scored models, and using
   it is what makes the diagonal a same-judge replication of the main run. A
   silent swap would quietly downgrade the sanity check from a judge-drift test
   to a judge-comparison, so an unserved judge is an operator decision, not a
   fallback. Nothing is spent before this check passes.
3. **Writes an immutable launch manifest before the first API call**, so the
   analysis reads what was launched instead of inferring it from timestamps.
4. **Gates on completeness afterwards**, counting samples that carry an on-scale
   severity -- the population the analysis actually admits -- not samples that
   merely exist.

Usage (from the MAIN checkout, where logs/baseline lives):
    python scripts/run_discriminant_evals.py                 # preflight, then stops
    python scripts/run_discriminant_evals.py --smoke         # 1 sample/model
    python scripts/run_discriminant_evals.py --yes 2>&1 | tee discriminant-launch.log

From a worktree, point every stage at the main checkout's logs so the new run
lands beside the archived baselines the verifier compares against:
    python scripts/run_discriminant_evals.py --logs-dir <main>/logs --yes
    python scripts/verify_discriminant_provenance.py --logs-dir <main>/logs
    python scripts/compute_discriminant_validity.py --logs-dir <main>/logs
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Python block-buffers stdout when it is not a terminal, so `| tee run.log`
# shows nothing for minutes and a hung run is indistinguishable from a buffered
# one. The decomposition runner learned this the hard way (59cbec8); same fix.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover - very old interpreters
    pass

from humanebench import provenance as prov  # noqa: E402
from humanebench.discriminant import (  # noqa: E402
    DATA_DIR,
    JUDGE_MODEL,
    JUDGE_SCORE_ATTEMPTS,
    JUDGE_TEMPERATURE,
    LOG_CONDITION,
    MANIFEST_PATH,
    SOURCE_MODEL_SLUGS,
    SOURCE_MODELS,
)
# No balance/usage imports: c574c2a removed the in-repo spend guard after it
# aborted two launches and prevented zero overspends. The budget control is the
# OpenRouter-side account cap, which cannot be wrong in our code. Spend is
# reconstructable from the .eval logs' token usage.
from run_decomposition_evals import (  # noqa: E402
    _openrouter_get,
    best_eval,
    check_model_availability,
    score_census,
)

TASK_FILE = "src/discriminant_multilabel_task.py"
OUT_DIR = REPO_ROOT / "provenance" / "discriminant"
LAUNCH_MANIFEST_PATH = OUT_DIR / "LAUNCH_MANIFEST.json"
RUN_STATUS_PATH = OUT_DIR / "run_status.json"


def _git_head() -> str | None:
    """Current commit, for the launch record. Absent outside a checkout."""
    if not prov.git_available():
        return None
    proc = prov._git("rev-parse", "HEAD")
    return proc.stdout.strip() if proc.returncode == 0 else None


# --- pricing -----------------------------------------------------------------
def judge_pricing(slug: str) -> tuple[float, float] | None:
    """Live ``(usd_per_input_token, usd_per_output_token)`` for one slug."""
    payload = _openrouter_get("/models")
    if not payload or not isinstance(payload.get("data"), list):
        return None
    bare = slug.removeprefix("openrouter/")
    for m in payload["data"]:
        if isinstance(m, dict) and m.get("id") == bare:
            pricing = m.get("pricing") or {}
            try:
                return float(pricing["prompt"]), float(pricing["completion"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def estimate_cost(manifest: dict) -> dict:
    """Estimated spend for the full run, at whatever OpenRouter charges today."""
    cal = manifest["pricing_calibration"]
    in_tokens = manifest["total_judge_prompt_chars"] / cal["chars_per_input_token"]
    out_tokens = manifest["n_judge_calls"] * cal["output_tokens_mean"]
    rates = judge_pricing(JUDGE_MODEL)
    est = {
        "n_calls": manifest["n_judge_calls"],
        "est_input_tokens": round(in_tokens),
        "est_output_tokens": round(out_tokens),
        "rates_usd_per_token": rates,
    }
    if rates:
        est["est_usd"] = round(in_tokens * rates[0] + out_tokens * rates[1], 2)
    else:
        # Informational only -- nothing gates on this number. An unknown rate is
        # reported as unknown rather than guessed.
        est["est_usd"] = None
    return est


# --- preflight ---------------------------------------------------------------
def preflight(manifest: dict) -> tuple[bool, dict]:
    """Return ``(ok_to_spend, report)``. Makes no billable call.

    Deliberately contains no balance or credit check. The account is on
    auto top-up, so the balance is not a ceiling -- it refills on demand -- and
    a gate reading it would refuse a ~$12 run over a $10 balance that would
    never have blocked anything. The decomposition runner's in-repo spend guard
    aborted two launches and prevented zero overspends before c574c2a deleted
    it; the budget control is the OpenRouter-side account cap.
    """
    report: dict = {"checked_at": datetime.now(timezone.utc).isoformat()}
    ok = True

    print("== preflight ==")

    # 0. The API key actually authenticates. Every other preflight call hits the
    # public /models endpoint, which succeeds with a missing or invalid key --
    # the deleted credit check was, accidentally, the only authenticated call in
    # preflight, and removing it opened a path where a bad key sails through and
    # the failure surfaces hours later as a run full of NaN judge scores. /key
    # is authenticated and free, and this is an auth check, not a spend guard.
    key_info = _openrouter_get("/key")
    report["api_key_valid"] = key_info is not None
    if key_info is None:
        print("  ! OPENROUTER_API_KEY missing or invalid (authenticated /key "
              "call failed)")
        ok = False
    else:
        print("  api key   authenticates")

    # 1. Datasets exist and match the manifest they were built with.
    datasets_ok = True
    for model in SOURCE_MODELS:
        path = DATA_DIR / f"multilabel_{model}.jsonl"
        recorded = manifest["per_model"].get(model, {})
        actual = prov.file_sha256(path) if path.exists() else None
        good = actual is not None and actual == recorded.get("dataset_file_sha256")
        report.setdefault("datasets", {})[model] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256_matches_manifest": good,
        }
        print(f"  dataset {model:22s} {'OK' if good else 'MISMATCH'}")
        datasets_ok &= good
    if not datasets_ok:
        print("  ! rebuild with scripts/build_discriminant_multilabel_dataset.py")
    ok &= datasets_ok

    # 2. The judge. Not negotiable in software -- see the module docstring.
    availability = check_model_availability([JUDGE_MODEL])
    judge_served = availability.get(JUDGE_MODEL)
    report["judge"] = {"slug": JUDGE_MODEL, "served": judge_served}
    if judge_served is None:
        print(f"  judge {JUDGE_MODEL}: COULD NOT CHECK (OpenRouter /models unreadable)")
        ok = False
    elif not judge_served:
        print(f"  judge {JUDGE_MODEL}: NOT SERVED")
        report["judge"]["alternatives"] = _judge_alternatives()
        ok = False
    else:
        print(f"  judge {JUDGE_MODEL}: served")

    # 3. Source-model slugs. These are never called -- the strict solver would
    #    raise -- so a retired slug is cosmetic, not fatal. Recorded, not gated.
    slug_report = check_model_availability(list(SOURCE_MODEL_SLUGS.values()))
    report["source_model_slugs"] = slug_report
    retired = [s for s, live in slug_report.items() if live is False]
    if retired:
        print(f"  note: {len(retired)} source slug(s) no longer served: {retired}")
        print("        harmless -- they label the log header and are never called")

    # 4. Cost, printed for the operator. Never gated on.
    est = estimate_cost(manifest)
    report["estimate"] = est
    if est["est_usd"] is not None:
        print(f"  estimate  {est['n_calls']:,} calls  ~${est['est_usd']:.2f} "
              "(informational; the OpenRouter account cap is the budget control)")
    else:
        print(f"  estimate  {est['n_calls']:,} calls  ~{est['est_input_tokens']:,} in / "
              f"{est['est_output_tokens']:,} out tokens (live pricing unreadable)")

    report["ok"] = ok
    return ok, report


def _judge_alternatives() -> list[str]:
    """Other judges that could stand in, for the operator to choose from."""
    candidates = [
        "openrouter/openai/gpt-5",
        "openrouter/google/gemini-2.5-flash",
        "openrouter/openai/gpt-4.1",
    ]
    live = check_model_availability(candidates)
    return [s for s, ok in live.items() if ok]


# --- launch ------------------------------------------------------------------
def build_launch_manifest(manifest: dict, preflight_report: dict, models: list[str]) -> dict:
    """Immutable record of what is about to run, written before the first call."""
    return {
        "schema": "humanebench-discriminant-launch/1",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "design": "designed x measured principle matrix, baseline responses only",
        "task_file": TASK_FILE,
        "source_models": models,
        "source_model_slugs": {m: SOURCE_MODEL_SLUGS[m] for m in models},
        "judge": {
            "model": JUDGE_MODEL,
            "temperature": JUDGE_TEMPERATURE,
            "score_attempts": JUDGE_SCORE_ATTEMPTS,
            "n_judges": 1,
        },
        "one_judge_call_per_principle": True,
        "n_scenarios": manifest["n_scenarios"],
        "n_principles": manifest["n_principles"],
        "n_judge_calls": manifest["n_judge_calls"],
        "frame_ids_file": manifest["frame_ids_file"],
        "frame_ids_sha256": manifest["frame_ids_sha256"],
        "frame_subset_prompt_hash": manifest["frame_subset_prompt_hash"],
        "parent_ids_file": manifest["parent_ids_file"],
        "parent_ids_sha256": manifest["parent_ids_sha256"],
        "source_dataset_sha256": manifest["source_dataset_sha256"],
        "dataset_manifest_sha256": prov.file_sha256(MANIFEST_PATH),
        "expected_prompt_hashes_sha256": manifest["expected_prompt_hashes_sha256"],
        "preflight": preflight_report,
    }


def _log_matches_current_config(model: str, path: Path) -> str | None:
    """None if ``path`` was produced by the current config, else the reason.

    ``inspect eval-retry`` replays the task configuration recorded IN THE LOG,
    not the one on disk today, so retrying a stale-config log silently
    perpetuates that config no matter what the task file now says. Same
    invariant, same shape as the decomposition runner's guard. The dataset's
    *content* is not checkable from the header; the verifier's 2,304
    expected-prompt-hash check is the byte-level backstop for that.
    """
    try:
        header = prov.read_eval_header(path)
    except Exception as exc:
        return f"unreadable header ({exc})"
    ev = header.get("eval") or {}
    if ev.get("model") != SOURCE_MODEL_SLUGS[model]:
        return f"logged model {ev.get('model')!r} != {SOURCE_MODEL_SLUGS[model]!r}"
    logged_ds = (ev.get("dataset") or {}).get("location") or ""
    if Path(logged_ds).name != f"multilabel_{model}.jsonl":
        return (f"logged dataset {Path(logged_ds).name!r} != "
                f"'multilabel_{model}.jsonl'")
    task_args = ev.get("task_args") or {}
    if task_args.get("source_model") not in (None, model):
        return f"logged source_model {task_args.get('source_model')!r} != {model!r}"
    return None


def _fmt_dir(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_one(model: str, log_dir: Path, limit: int | None = None,
            retry_path: Path | None = None) -> dict:
    """One eval for one source model: fresh, or `eval-retry` on a partial log.

    The retry path matters because every dollar here is judge tokens: a run
    killed at 700/768 has ~68 unscored samples, and a fresh `inspect eval`
    would re-pay all 768. `eval-retry` re-runs only the incomplete samples.
    Callers must config-match the log first (`_log_matches_current_config`).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    if retry_path is not None:
        cmd = ["inspect", "eval-retry", str(retry_path),
               f"--log-dir={log_dir}", "--no-log-realtime"]
        mode = "eval-retry"
    else:
        cmd = [
            "inspect", "eval", TASK_FILE,
            "-T", f"source_model={model}",
            f"--model={SOURCE_MODEL_SLUGS[model]}",
            f"--log-dir={log_dir}",
        ]
        if limit is not None:
            cmd.append(f"--limit={limit}")
        mode = "eval"

    started = time.time()
    print(f"Starting ({mode}): {model} -> {_fmt_dir(log_dir)}")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(f"[{model}] {line.rstrip()}")
    proc.wait()
    result = {
        "model": model,
        "mode": mode,
        "log_dir": _fmt_dir(log_dir),
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "duration_s": round(time.time() - started, 1),
    }
    print(f"{'OK' if result['success'] else 'FAILED'}: {model} "
          f"({result['duration_s']:.0f}s)")
    return result


def gate(models: list[str], expected_per_model: int, threshold: float,
         log_root: Path) -> dict:
    """Per-model completeness census against the analysis admission rule.

    Two-sided on purpose. A cell that scored MORE samples than the frame is
    exactly what a run against a stale or oversized dataset looks like, and a
    one-sided ``frac >= threshold`` would wave it through into the matrix. Such
    a cell is ``wrong_frame``, and it fails the gate regardless of fraction.
    """
    report: dict = {"threshold": threshold,
                    "expected_samples_per_model": expected_per_model,
                    "models": {}}
    worst = 1.0
    wrong_frame = False
    for model in models:
        model_dir = log_root / model
        path = best_eval(model_dir) if model_dir.is_dir() else None
        if path is None:
            report["models"][model] = {"status": "missing", "fraction_scored": 0.0}
            worst = 0.0
            continue
        census = score_census(path)
        frac = census["n_fully_scored"] / expected_per_model if expected_per_model else 0.0
        if (census["n_samples"] > expected_per_model
                or census["n_fully_scored"] > expected_per_model):
            status = "wrong_frame"
            wrong_frame = True
        elif frac >= threshold:
            status = "ok"
        else:
            status = "degraded"
        report["models"][model] = {
            "status": status,
            "eval_file": _fmt_dir(path),
            "fraction_scored": round(frac, 5),
            **census,
        }
        worst = min(worst, frac)
    report["worst_fraction_scored"] = round(worst, 5)
    report["passed"] = worst >= threshold and not wrong_frame
    return report


def retry_incomplete(models: list[str], log_root: Path) -> None:
    """One non-interactive `inspect eval-retry` pass per model log.

    Config-matched first: eval-retry replays the log's recorded config, so
    retrying a stale log would silently perpetuate it.
    """
    for model in models:
        model_dir = log_root / model
        if not model_dir.is_dir():
            continue
        path = best_eval(model_dir)
        if path is None:
            continue
        reason = _log_matches_current_config(model, path)
        if reason is not None:
            print(f"NOT retrying {model}: {reason}")
            continue
        print(f"retrying {model}")
        subprocess.run(
            ["inspect", "eval-retry", str(path), f"--log-dir={model_dir}", "--no-log-realtime"],
            cwd=REPO_ROOT, check=False,
        )


def archive_superseded(models: list[str], log_root: Path) -> list[str]:
    """Keep one .eval per model dir; retries leave extras that confuse discovery."""
    moved: list[str] = []
    for model in models:
        model_dir = log_root / model
        if not model_dir.is_dir():
            continue
        evals = sorted(p for p in model_dir.glob("*.eval") if p.is_file())
        if len(evals) <= 1:
            continue
        keep = best_eval(model_dir)
        attic = model_dir / "attic"
        attic.mkdir(exist_ok=True)
        for path in evals:
            if path != keep:
                shutil.move(str(path), str(attic / path.name))
                moved.append(_fmt_dir(path))
    return moved


def _attic_stale_log(model_dir: Path, path: Path, reason: str) -> None:
    """Move a config-mismatched log aside so no later discovery picks it up."""
    attic = model_dir / "attic"
    attic.mkdir(exist_ok=True)
    dest = attic / path.name
    shutil.move(str(path), str(dest))
    print(f"  archived stale log ({reason}) -> {_fmt_dir(dest)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(SOURCE_MODELS),
                    choices=list(SOURCE_MODELS))
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs",
                    help="root logs directory. New run logs are written to "
                         "<logs-dir>/discriminant/<model>. When launching from "
                         "a worktree, point this at the main checkout's logs/ "
                         "so the run lands beside the archived baselines the "
                         "verifier needs -- the verifier, builder and analysis "
                         "all take the same flag.")
    ap.add_argument("--gate-threshold", type=float, default=0.98)
    ap.add_argument("--smoke", action="store_true",
                    help="one sample per model, to prove the path end to end")
    ap.add_argument("--yes", action="store_true", help="run without confirmation")
    ap.add_argument("--force", action="store_true",
                    help="re-run models that already have logs")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    log_root = args.logs_dir / LOG_CONDITION

    ok, report = preflight(manifest)
    if not ok:
        judge = report.get("judge", {})
        if judge.get("served") is False:
            print("\n" + "=" * 72)
            print(f"STOP: the judge {JUDGE_MODEL} is no longer served.")
            print("Nothing has been spent.")
            alts = judge.get("alternatives") or []
            print(f"Judges currently served that are not among the scored models: "
                  f"{alts if alts else 'none of the candidates checked'}")
            print("Substituting one keeps the matrix and the headline contrast, but")
            print("downgrades sanity check 4 from a same-judge drift test to a")
            print("judge-comparison. That is a decision for the operator, not a")
            print("fallback for this script. Re-run with the judge you choose set")
            print("in humanebench/discriminant.py.")
            print("=" * 72)
        print("\npreflight FAILED; nothing launched")
        return 1

    if not args.smoke and not args.yes:
        est = report["estimate"]
        cost = (f"approximately ${est['est_usd']:.2f}" if est.get("est_usd") is not None
                else f"~{est['est_input_tokens']:,} input tokens")
        print(f"\nAbout to spend {cost} on {est['n_calls']:,} judge calls. "
              "Re-run with --yes to proceed, or --smoke first.")
        return 0

    smoke_log_dir = args.logs_dir / f"{LOG_CONDITION}_smoke"
    if args.smoke:
        if smoke_log_dir.exists():
            shutil.rmtree(smoke_log_dir)
        results = [run_one(m, smoke_log_dir / m, limit=1) for m in args.models]
        # Returncode alone is not a smoke pass: `inspect eval` exits 0 even when
        # every judge call fails and the scorer records NaN. The smoke exists to
        # prove the judge path end to end, so demand at least one fully scored
        # sample -- the same admission rule the gate and the analysis use.
        failed = [r["model"] for r in results if not r["success"]]
        for model in args.models:
            if model in failed:
                continue
            path = best_eval(smoke_log_dir / model)
            n = score_census(path)["n_fully_scored"] if path else 0
            if n < 1:
                print(f"smoke {model}: eval exited 0 but no sample carries a "
                      "valid judge score -- the judge path is broken")
                failed.append(model)
        print(f"\nsmoke: {len(args.models) - len(failed)}/{len(args.models)} models OK")
        return 1 if failed else 0

    # Resume on COMPLETENESS, not on the existence of a file -- and resume via
    # `eval-retry`, not a fresh run. Every dollar here is judge tokens: a model
    # killed at 700/768 has ~68 unscored samples, and re-running `inspect eval`
    # would re-pay all 768. A partial log is only retryable if it matches the
    # current config (eval-retry replays the config recorded in the log); a
    # stale log is archived and the model re-run fresh.
    expected = manifest["n_scenarios"] * manifest["n_principles"]
    plan: list[tuple[str, Path | None]] = []  # (model, retry_path or None)
    for model in args.models:
        model_dir = log_root / model
        path = best_eval(model_dir) if model_dir.is_dir() else None
        if path is None or args.force:
            plan.append((model, None))
            continue
        scored = score_census(path)["n_fully_scored"]
        if scored >= expected * args.gate_threshold:
            print(f"skip {model}: {scored}/{expected} already scored")
            continue
        reason = _log_matches_current_config(model, path)
        if reason is None:
            print(f"resume {model}: {scored}/{expected} scored; eval-retry on "
                  "the partial log")
            plan.append((model, path))
        else:
            print(f"fresh {model}: partial log unusable -- {reason}")
            _attic_stale_log(model_dir, path, reason)
            plan.append((model, None))
    if not plan:
        print("all models already complete; running the gate to confirm")

    results: list[dict] = []
    moved: list[str] = []
    if plan:
        todo = [m for m, _p in plan]
        launch = build_launch_manifest(manifest, report, todo)
        # The launch manifest is immutable per launch, not per file: a resume
        # must not clobber the record of what the original launch started.
        if LAUNCH_MANIFEST_PATH.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            prior = OUT_DIR / f"LAUNCH_MANIFEST.{stamp}.json"
            shutil.move(str(LAUNCH_MANIFEST_PATH), str(prior))
            print(f"archived prior launch manifest -> {prior.name}")
        LAUNCH_MANIFEST_PATH.write_text(json.dumps(launch, indent=2) + "\n")
        print(f"\nwrote {LAUNCH_MANIFEST_PATH.relative_to(REPO_ROOT)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(run_one, m, log_root / m, None, retry_path): m
                for m, retry_path in plan
            }
            for fut in concurrent.futures.as_completed(futures):
                model = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # never lose the run's status to one model
                    print(f"EXCEPTION in {model}: {exc!r}")
                    results.append({"model": model, "success": False,
                                    "exception": repr(exc)})

        retry_incomplete(todo, log_root)
        moved = archive_superseded(todo, log_root)

    # The gate runs unconditionally, including when nothing needed running. It
    # is the only thing that reports whether the matrix can be built, so an
    # early return past it would make a partial run look like a clean no-op.
    census = gate(args.models, expected, args.gate_threshold, log_root)
    status = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runs": results,
        "archived_superseded": moved,
        "gate": census,
        # Spend is not tracked here: the account is capped on the OpenRouter
        # side, and the .eval logs carry per-call token usage from which cost is
        # reconstructable exactly.
    }
    RUN_STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")

    print("\n== completeness ==")
    for model, m in census["models"].items():
        print(f"  {model:22s} {m.get('n_fully_scored', 0):4d}/{expected} "
              f"({m['fraction_scored']:.1%})  {m['status']}"
              + (f"  judge_failures={m['n_judge_failures']}" if m.get("n_judge_failures") else ""))
    print(f"wrote {RUN_STATUS_PATH.relative_to(REPO_ROOT)}")

    if not census["passed"]:
        print("\nGATE FAILED: at least one model is short of the threshold. Report "
              "the run as incomplete rather than analysing a partial matrix.")
        return 1
    print("\nGate passed. Next: python scripts/verify_discriminant_provenance.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
