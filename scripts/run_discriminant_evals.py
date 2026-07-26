#!/usr/bin/env python3
"""Launch the multi-label scoring run for the designed x measured matrix.

2,304 judge calls: 96 scenarios x 8 principles x 3 source models, one judge.
No response is generated -- the task uses the strict pregenerated solver over
archived baseline responses -- so every dollar here is judge tokens.

What this does that a bare `inspect eval` loop does not:

1. **Prices the run against live OpenRouter rates** rather than a remembered
   figure, using prompt sizes measured from the built datasets and an output
   length calibrated on the 287 gpt-5.1 judge calls the reported runs already
   made against these same scenarios.
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

Usage:
    python scripts/run_discriminant_evals.py                 # preflight, then stops
    python scripts/run_discriminant_evals.py --smoke         # 1 sample/model
    python scripts/run_discriminant_evals.py --yes 2>&1 | tee discriminant-launch.log
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
from run_decomposition_evals import (  # noqa: E402
    _openrouter_get,
    best_eval,
    check_model_availability,
    get_openrouter_credits,
    get_openrouter_usage,
    score_census,
)

TASK_FILE = "src/discriminant_multilabel_task.py"
LOG_ROOT = REPO_ROOT / "logs" / LOG_CONDITION
SMOKE_LOG_DIR = REPO_ROOT / "logs" / f"{LOG_CONDITION}_smoke"
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
        # Never guess a price into a spend guard. An unknown rate means the gate
        # cannot be evaluated, and the operator is told so rather than shown a
        # number that looks measured.
        est["est_usd"] = None
    return est


# --- preflight ---------------------------------------------------------------
def preflight(manifest: dict, min_credit_factor: float, require_credit: bool) -> tuple[bool, dict]:
    """Return ``(ok_to_spend, report)``. Makes no billable call."""
    report: dict = {"checked_at": datetime.now(timezone.utc).isoformat()}
    ok = True

    print("== preflight ==")

    # 1. Datasets exist and match the manifest they were built with.
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
        ok &= good
    if not ok:
        print("  ! rebuild with scripts/build_discriminant_multilabel_dataset.py")

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

    # 4. Money.
    est = estimate_cost(manifest)
    report["estimate"] = est
    credits = get_openrouter_credits()
    report["credits_usd"] = credits
    if est["est_usd"] is None:
        print("  ! could not read live pricing; the spend gate cannot be evaluated")
        ok = False
    else:
        needed = est["est_usd"] * min_credit_factor
        report["credit_required_usd"] = round(needed, 2)
        print(f"  estimate  {est['n_calls']:,} calls  ~${est['est_usd']:.2f}  "
              f"(gate needs ${needed:.2f} at {min_credit_factor}x)")
        if credits is None:
            print("  ! OpenRouter balance unreadable")
            if require_credit:
                ok = False
        else:
            print(f"  credit    ${credits:.2f}")
            if credits < needed:
                print("  ! insufficient credit")
                ok = False

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


def run_one(model: str, log_dir: Path, limit: int | None) -> dict:
    """One `inspect eval` for one source model's 768-row dataset."""
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "inspect", "eval", TASK_FILE,
        f"-T", f"source_model={model}",
        f"--model={SOURCE_MODEL_SLUGS[model]}",
        f"--log-dir={log_dir}",
    ]
    if limit is not None:
        cmd.append(f"--limit={limit}")

    started = time.time()
    print(f"Starting: {model} -> {log_dir.relative_to(REPO_ROOT)}")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(f"[{model}] {line.rstrip()}")
    proc.wait()
    result = {
        "model": model,
        "log_dir": str(log_dir.relative_to(REPO_ROOT)),
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "duration_s": round(time.time() - started, 1),
    }
    print(f"{'OK' if result['success'] else 'FAILED'}: {model} "
          f"({result['duration_s']:.0f}s)")
    return result


def gate(models: list[str], expected_per_model: int, threshold: float) -> dict:
    """Per-model completeness census against the analysis admission rule."""
    report: dict = {"threshold": threshold,
                    "expected_samples_per_model": expected_per_model,
                    "models": {}}
    worst = 1.0
    for model in models:
        model_dir = LOG_ROOT / model
        path = best_eval(model_dir) if model_dir.is_dir() else None
        if path is None:
            report["models"][model] = {"status": "missing", "fraction_scored": 0.0}
            worst = 0.0
            continue
        census = score_census(path)
        frac = census["n_fully_scored"] / expected_per_model if expected_per_model else 0.0
        report["models"][model] = {
            "status": "ok" if frac >= threshold else "degraded",
            "eval_file": str(path.relative_to(REPO_ROOT)),
            "fraction_scored": round(frac, 5),
            **census,
        }
        worst = min(worst, frac)
    report["worst_fraction_scored"] = round(worst, 5)
    report["passed"] = worst >= threshold
    return report


def retry_incomplete(models: list[str]) -> None:
    """One non-interactive `inspect eval-retry` pass per model log."""
    for model in models:
        model_dir = LOG_ROOT / model
        if not model_dir.is_dir():
            continue
        path = best_eval(model_dir)
        if path is None:
            continue
        print(f"retrying {model}")
        subprocess.run(
            ["inspect", "eval-retry", str(path), f"--log-dir={model_dir}", "--no-log-realtime"],
            cwd=REPO_ROOT, check=False,
        )


def archive_superseded(models: list[str]) -> list[str]:
    """Keep one .eval per model dir; retries leave extras that confuse discovery."""
    moved: list[str] = []
    for model in models:
        model_dir = LOG_ROOT / model
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
                moved.append(str(path.relative_to(REPO_ROOT)))
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(SOURCE_MODELS),
                    choices=list(SOURCE_MODELS))
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--min-credit-factor", type=float, default=1.2)
    ap.add_argument("--gate-threshold", type=float, default=0.98)
    ap.add_argument("--smoke", action="store_true",
                    help="one sample per model, to prove the path end to end")
    ap.add_argument("--yes", action="store_true", help="run without confirmation")
    ap.add_argument("--no-require-credit-check", action="store_true",
                    help="proceed when the balance cannot be read (not when it is "
                         "readable and too low)")
    ap.add_argument("--force", action="store_true",
                    help="re-run models that already have logs")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())

    ok, report = preflight(manifest, args.min_credit_factor,
                           require_credit=not args.no_require_credit_check)
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
        print(f"\nAbout to spend approximately ${est['est_usd']:.2f} on "
              f"{est['n_calls']:,} judge calls. Re-run with --yes to proceed, or "
              "--smoke first.")
        return 0

    if args.smoke:
        if SMOKE_LOG_DIR.exists():
            shutil.rmtree(SMOKE_LOG_DIR)
        results = [run_one(m, SMOKE_LOG_DIR / m, limit=1) for m in args.models]
        failed = [r["model"] for r in results if not r["success"]]
        print(f"\nsmoke: {len(results) - len(failed)}/{len(results)} models OK")
        return 1 if failed else 0

    todo = []
    for model in args.models:
        model_dir = LOG_ROOT / model
        if model_dir.is_dir() and any(model_dir.glob("*.eval")) and not args.force:
            print(f"skip {model}: logs already present (use --force to re-run)")
            continue
        todo.append(model)
    if not todo:
        print("nothing to run")
        return 0

    usage_before = get_openrouter_usage()
    launch = build_launch_manifest(manifest, report, todo)
    launch["usage_before_usd"] = usage_before
    LAUNCH_MANIFEST_PATH.write_text(json.dumps(launch, indent=2) + "\n")
    print(f"\nwrote {LAUNCH_MANIFEST_PATH.relative_to(REPO_ROOT)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, m, LOG_ROOT / m, None): m for m in todo}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    retry_incomplete(todo)
    moved = archive_superseded(todo)

    expected = manifest["n_scenarios"] * manifest["n_principles"]
    census = gate(args.models, expected, args.gate_threshold)
    usage_after = get_openrouter_usage()
    status = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runs": results,
        "archived_superseded": moved,
        "gate": census,
        "usage_before_usd": usage_before,
        "usage_after_usd": usage_after,
        "spend_usd": (round(usage_after - usage_before, 4)
                      if usage_before is not None and usage_after is not None else None),
    }
    RUN_STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")

    print("\n== completeness ==")
    for model, m in census["models"].items():
        print(f"  {model:22s} {m.get('n_fully_scored', 0):4d}/{expected} "
              f"({m['fraction_scored']:.1%})  {m['status']}"
              + (f"  judge_failures={m['n_judge_failures']}" if m.get("n_judge_failures") else ""))
    if status["spend_usd"] is not None:
        print(f"\nspend: ${status['spend_usd']:.2f}")
    print(f"wrote {RUN_STATUS_PATH.relative_to(REPO_ROOT)}")

    if not census["passed"]:
        print("\nGATE FAILED: at least one model is short of the threshold. Report "
              "the run as incomplete rather than analysing a partial matrix.")
        return 1
    print("\nGate passed. Next: python scripts/verify_discriminant_provenance.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
