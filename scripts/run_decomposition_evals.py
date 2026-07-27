#!/usr/bin/env python3
"""Launch the goal-vs-tactics decomposition evaluation runs.

Runs conditions B -> C -> D -> E strictly in that order, parallel across models
within a condition. The ordering means an interrupted night leaves the earlier
conditions complete and self-contained rather than four half-finished arms.

Budget enforcement is deliberately NOT done here. An in-repo spend guard aborted
two launches on readings of an auto-top-up balance that refills on demand and
means nothing as a ceiling; the cap now lives on the OpenRouter account, where
it is enforced server-side and cannot be wrong in this code.

Three things this does that a bare `inspect eval` loop does not:

1. Writes an immutable launch manifest BEFORE the first API call, recording the
   verbatim prompts and their hashes, the models, the scale, the subsample, and
   the seeds. The analysis session reads that file instead of reconstructing
   what ran from timestamps.
2. Resumes per MODEL: a cell with a complete log is skipped, a partial log is
   continued with `inspect eval-retry` (only if it matches the current config),
   so re-running after any interruption never re-pays for finished work.
3. Gates each condition on completeness *after* it finishes -- counting samples
   that carry a full 3-judge on-scale score, not just samples that exist -- and
   stops the pipeline if a condition came out degraded.

Usage:
    python scripts/run_decomposition_evals.py --smoke      # 1 sample/condition
    python scripts/run_decomposition_evals.py --yes 2>&1 | tee decomp-launch.log
    python scripts/run_decomposition_evals.py --conditions decomp_c_prose --yes
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import concurrent.futures
from collections import Counter
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Python block-buffers stdout when it is not a terminal, so `> run.log` hides all
# progress until several KB accumulate -- which on a slow condition can be many
# minutes of apparent silence. Reconfigure to line buffering so the log is
# readable live without the caller having to remember `python -u`.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover - very old interpreters
    pass

from humanebench import decomposition as dc
from humanebench import provenance as prov
from humanebench.excluded import load_excluded_ids
from run_parallel_evals import run_evaluation

OUT_DIR = REPO_ROOT / "provenance" / "decomposition"
LAUNCH_MANIFEST_PATH = OUT_DIR / "LAUNCH_MANIFEST.json"
RUN_STATUS_PATH = OUT_DIR / "run_status.json"
SMOKE_LOG_DIR = REPO_ROOT / "logs" / "decomp_smoke"

OPENROUTER_API = "https://openrouter.ai/api/v1"
VALID_SEVERITIES = {-1.0, -0.5, 0.5, 1.0}


# --- OpenRouter pre-flight ---------------------------------------------------
def _openrouter_get(path: str, timeout: float = 20.0) -> dict | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    req = urllib.request.Request(
        f"{OPENROUTER_API}{path}", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  ! OpenRouter {path} failed: {exc}")
        return None


def check_model_availability(slugs: list[str]) -> dict[str, bool]:
    """Membership-test each slug against OpenRouter's live model list."""
    payload = _openrouter_get("/models")
    if not payload or not isinstance(payload.get("data"), list):
        return {}
    live = {m.get("id") for m in payload["data"] if isinstance(m, dict)}
    return {s: s.removeprefix("openrouter/") in live for s in slugs}


# --- eval log inspection -----------------------------------------------------
def latest_eval(model_dir: Path) -> Path | None:
    """Newest .eval by filename (Inspect prefixes an ISO timestamp)."""
    evals = sorted(p for p in model_dir.glob("*.eval") if p.is_file())
    return evals[-1] if evals else None


def best_eval(model_dir: Path, exclude: set[str] | None = None,
              cond: "dc.Condition | None" = None) -> Path | None:
    """The .eval holding the most usable data, tie-broken by newest.

    "Newest" is the wrong criterion on its own: a re-run that dies early leaves
    a fresh but truncated file next to a complete one from a previous night.
    Selecting on scored-sample count keeps the run that actually has the data.

    When ``cond`` is given, logs from a stale config are excluded outright --
    otherwise a big log written under an old prompt or dataset outranks a
    smaller correct one, and the gate would attest the wrong experiment.

    A truncated or corrupt archive (what a SIGKILLed launch can leave behind)
    scores as zero rather than crashing discovery: one bad file must not make
    an entire cell unreadable.
    """
    evals = sorted(p for p in model_dir.glob("*.eval") if p.is_file())
    if cond is not None:
        model_full = next((m for m in dc.MODELS if m.split("/")[-1] == model_dir.name), None)
        if model_full is not None:
            kept = []
            for p in evals:
                reason = _log_matches_current_config(cond, model_full, p)
                if reason is None:
                    kept.append(p)
                else:
                    print(f"  [{model_dir.name}] ignoring {p.name}: {reason}")
            evals = kept
    if not evals:
        return None
    if len(evals) == 1:
        return evals[0]

    def usable(p: Path) -> int:
        try:
            return score_census(p, exclude)["n_fully_scored"]
        except Exception as exc:
            print(f"  [{model_dir.name}] unreadable {p.name}: {exc!r}; treating as empty")
            return -1

    return max(evals, key=lambda p: (usable(p), p.name))


def score_census(eval_path: Path, exclude: set[str] | None = None) -> dict:
    """Count how many samples will survive into the analysis.

    ``n_fully_scored`` deliberately mirrors the admission rule in
    ``compute_inter_judge_agreement.collect_long_table``: a sample counts if it
    has an ``overseer`` score whose ``individual_scores`` matches
    ``ensemble_models`` in length and whose severities are all on the canonical
    4-point scale. That is the population every downstream table is built from,
    so gating on anything else would gate on a number the analysis never uses.

    Two diagnostics are tracked separately because they mean different things:

    ``n_judge_failures``  a judge never returned after ``score_attempts`` tries,
        so the scorer early-returned with no ``individual_scores`` at all.
    ``n_nan_with_scores``  all judges answered, but one flagged the *evaluated
        model's* response invalid (in the reported runs these are empty model
        responses). The ensemble value is NaN while the three judge severities
        survive, so the analysis pipeline still admits these samples.

    Neither is recoverable by ``inspect eval-retry``: both leave a sample that
    is complete from Inspect's point of view.
    """
    exclude = exclude or set()
    n_samples = n_full = n_judge_fail = n_offscale = n_nan_with_scores = n_excluded = 0
    for sample in prov.iter_eval_samples(eval_path):
        n_samples += 1
        if sample.get("id") in exclude:
            n_excluded += 1
            continue
        overseer = (sample.get("scores") or {}).get("overseer")
        if not overseer:
            n_judge_fail += 1
            continue
        meta = overseer.get("metadata") or {}
        individual = meta.get("individual_scores")
        judges = meta.get("ensemble_models")
        if not individual or not judges or len(individual) != len(judges):
            n_judge_fail += 1
            continue
        if any(s not in VALID_SEVERITIES for s in individual):
            n_offscale += 1
            continue
        n_full += 1
        value = overseer.get("value")
        if isinstance(value, float) and math.isnan(value):
            n_nan_with_scores += 1
    return {
        "n_samples": n_samples,
        "n_excluded_by_flag": n_excluded,
        "n_fully_scored": n_full,
        "n_judge_failures": n_judge_fail,
        "n_off_scale": n_offscale,
        "n_nan_with_scores": n_nan_with_scores,
    }


def gate_condition(cond: dc.Condition, models: list[str], threshold: float) -> dict:
    """Per-model completeness census for a finished condition."""
    exclude = load_excluded_ids(cond.dataset_path)
    denom = cond.expected_analysis_samples()
    report: dict = {
        "task_type": cond.task_type,
        "threshold": threshold,
        "expected_analysis_samples": denom,
        "models": {},
    }
    worst = 1.0
    wrong_frame: list[str] = []
    for model in models:
        model_dir = cond.log_dir / model.split("/")[-1]
        path = best_eval(model_dir, exclude, cond=cond) if model_dir.is_dir() else None
        if path is None:
            report["models"][model] = {"status": "missing", "fraction_scored": 0.0}
            worst = 0.0
            continue
        try:
            census = score_census(path, exclude=exclude)
        except Exception as exc:
            report["models"][model] = {"status": "unreadable", "error": repr(exc),
                                       "fraction_scored": 0.0,
                                       "eval_file": str(path.relative_to(REPO_ROOT))}
            worst = 0.0
            continue
        frac = census["n_fully_scored"] / denom if denom else 0.0
        # Two-sided. A one-sided `frac >= threshold` attests a run that scored
        # MORE than the expected frame as complete -- which is exactly what a
        # wrong-dataset run looks like, and the wrong frame is silent in every
        # downstream table.
        over = census["n_samples"] > cond.expected_samples
        status = ("ok" if (frac >= threshold and not over)
                  else "wrong_frame" if over else "degraded")
        report["models"][model] = {
            "status": status,
            "eval_file": str(path.relative_to(REPO_ROOT)),
            "fraction_scored": round(frac, 5),
            **census,
        }
        worst = min(worst, frac)
        if over:
            wrong_frame.append(model)
    report["worst_fraction_scored"] = round(worst, 5)
    report["wrong_frame_models"] = wrong_frame
    report["passed"] = worst >= threshold and not wrong_frame
    report["serving_providers"] = provider_census(cond, models)
    return report


def provider_census(cond: dc.Condition, models: list[str]) -> dict:
    """Which upstream stack actually answered, per model.

    The reported runs recorded only the slug, so their provider mixture had to be
    recovered afterwards from raw responses. Capturing it at run time means the
    decomposition does not inherit that gap: a cell served by several stacks is
    visible in the status file rather than needing a later forensic pass.
    """
    out: dict = {}
    for model in models:
        model_dir = cond.log_dir / model.split("/")[-1]
        path = best_eval(model_dir, cond=cond) if model_dir.is_dir() else None
        if path is None:
            continue
        counts: Counter = Counter()
        try:
            for sample in prov.iter_eval_samples(path):
                for ev in sample.get("events") or []:
                    if ev.get("event") != "model" or ev.get("model") != model:
                        continue
                    resp = (ev.get("call") or {}).get("response")
                    if isinstance(resp, dict):
                        counts[resp.get("provider")] += 1
                    break
        except Exception as exc:  # a census must never fail a completed run
            out[model] = {"error": repr(exc)}
            continue
        total = sum(counts.values()) or 1
        out[model] = {
            "n_distinct_providers": len(counts),
            "shares": {str(k): round(v / total, 4) for k, v in counts.most_common()},
        }
    return out


def archive_superseded(cond: dc.Condition, models: list[str],
                       exclude: set[str] | None = None) -> list[str]:
    """Keep exactly one .eval per model dir; retries leave extras behind.

    Every discovery path downstream (provenance, the analysis scripts) assumes
    one file per model directory, so extras are moved aside rather than deleted.
    """
    moved: list[str] = []
    for model in models:
        model_dir = cond.log_dir / model.split("/")[-1]
        if not model_dir.is_dir():
            continue
        evals = sorted(p for p in model_dir.glob("*.eval") if p.is_file())
        if len(evals) <= 1:
            continue
        keep = best_eval(model_dir, exclude, cond=cond)
        attic = model_dir / "attic"
        attic.mkdir(exist_ok=True)
        for path in evals:
            if path == keep:
                continue
            shutil.move(str(path), str(attic / path.name))
            moved.append(str(path.relative_to(REPO_ROOT)))
    return moved


def retry_incomplete(cond: dc.Condition, models: list[str], max_workers: int) -> None:
    """One non-interactive `inspect eval-retry` pass over each model's latest log."""
    targets: list[Path] = []
    for model in models:
        model_dir = cond.log_dir / model.split("/")[-1]
        if not model_dir.is_dir():
            continue
        path = best_eval(model_dir, cond=cond)
        if path is None:
            continue
        census = score_census(path)
        if census["n_samples"] < cond.expected_samples:
            targets.append(path)
    if not targets:
        return
    print(f"  retrying {len(targets)} incomplete run(s) for {cond.task_type}")

    def _retry(path: Path) -> None:
        # --log-dir is mandatory here: the CLI defaults it to ./logs, so without
        # it a recovered run lands in the logs/ root instead of the model's
        # directory. The retry would spend money, succeed, and then be invisible
        # to the gate, the archiver, and every analysis path.
        subprocess.run(
            [
                "inspect", "eval-retry", str(path),
                f"--log-dir={path.parent}",
                "--max-connections=10",
            ],
            cwd=REPO_ROOT,
            check=False,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_retry, targets))


# --- manifest / status -------------------------------------------------------
def build_launch_manifest(conditions: list[dc.Condition], models: list[str]) -> dict:
    summary = dc.load_subset_summary()
    subset_ids = dc.load_subset_ids()
    return {
        "schema": "humanebench-decomposition-launch/1",
        "description": (
            "Goal-vs-tactics decomposition of the HumaneBench adversarial condition. "
            "Reported as a robustness analysis of the adversarial condition, not as a "
            "fourth headline condition."
        ),
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or None,
        "anchor_condition": {
            "persona": dc.ANCHOR_PERSONA,
            "note": "Existing reported run (condition A); not re-run.",
        },
        "conditions": [
            {
                "condition_label": c.label,
                "task_type": c.task_type,
                "task_file": c.task_file_rel,
                "rationale": c.rationale,
                "system_prompt": dc.prompt_for(c),
                "system_prompt_sha256": dc.sha256_text(dc.prompt_for(c)),
                "prompt_source": "adversarial-conditions.md RQ5",
                "deviations_from_source": list(c.deviations),
                "dataset": c.dataset_rel,
                "scale": c.scale,
                "expected_samples_per_model": c.expected_samples,
                "expected_prompt_hash": dc.expected_prompt_hash(c),
                "est_cost_usd": round(c.est_cost_usd(len(models)), 2),
                "log_dir": f"logs/{c.task_type}",
            }
            for c in conditions
        ],
        "models": list(models),
        "n_models": len(models),
        "judge_ensemble": dc.JUDGE_ENSEMBLE,
        "judge_ensemble_as_published": prov.JUDGE_ENSEMBLE,
        "cohort": {
            "models_published": dc.MODELS_PUBLISHED,
            "models_retired_since": dc.RETIRED_MODELS,
            "note": (
                "4 of the 15 reported models are no longer served and have no "
                "same-model substitute; all 4 are flipping models. The surviving "
                "cohort holds 6 of the 10 flippers and all 4 robust models. Every "
                "contrast against the reported conditions must restrict them to "
                "these same models."
            ),
        },
        "subsample": {
            "ids_file": dc.SUBSET_IDS_REL,
            "ids_file_sha256": summary["ids_file_sha256"],
            "dataset_file": dc.SUBSET_DATASET_REL,
            "dataset_file_sha256": summary["jsonl_file_sha256"],
            "n_scenarios": len(subset_ids),
            "seed": summary["seed"],
            "stratification": summary["stratification"],
            "subset_prompt_hash": summary["subset_prompt_hash"],
            "note": (
                "Drawn once and frozen. Conditions C/D/E score exactly these "
                "scenarios; every contrast against baseline/bad_persona is "
                "restricted to them so the comparison stays paired."
            ),
        },
        "frozen_prompt_hash": prov.FROZEN_PROMPT_HASH,
        "bootstrap_seed": 20260407,
        "cost_anchor": (
            f"${dc.COST_PER_800_SAMPLES_USD} per 800-sample evaluation "
            "(scripts/run_parallel_retries.py); engagement-framed prompts may "
            "exceed it because responses run longer."
        ),
        "provider_routing": {
            "policy": "unpinned",
            "rationale": (
                "OpenRouter fulfils one slug from several upstream serving stacks. "
                "The reported runs were routed unpinned -- 7 of 15 models drew from "
                "more than one provider (llama-4-maverick from 8, "
                "deepseek-v3.1-terminus from 5). Pinning providers here would make "
                "this condition more controlled than the conditions it is contrasted "
                "against, adding a second axis of difference alongside the prompt "
                "change. A confound shared by both arms is preferable to an "
                "asymmetry between them."
            ),
            "reported_run_mixture": "tables/serving_provenance.md",
            "per_response_provider_recorded": True,
            "unversioned_slugs": [
                "openrouter/google/gemini-2.5-pro",
                "openrouter/google/gemini-2.5-flash",
            ],
            "unversioned_note": (
                "These two carry no version marker in their OpenRouter "
                "canonical_slug, so whether they are the weights tested in Nov 2025 "
                "is not determinable from any exposed field. The other nine resolve "
                "to a dated or named snapshot predating those runs."
            ),
        },
        "temporal_caveat": dc.TEMPORAL_CAVEAT,
        "analysis_commitments": [
            "Report difference-in-differences (delta_bad - delta_condition), never a ratio: "
            "delta_bad is -0.03 for GPT-5, so ratios explode or flip sign for the most "
            "robust models. Ratio framing is restricted to the 10 flipping models.",
            "Krippendorff's alpha, design effects, item counts, and the judge "
            "self-preference DiD are NOT re-pooled across these conditions; the published "
            "values are conditioned on the original 3 personas and new-condition agreement "
            "is computed separately.",
            "A null or attenuated result is pre-committed as publishable "
            "(specification-dependence) and will not be softened.",
        ],
    }


def load_status() -> dict:
    if RUN_STATUS_PATH.exists():
        return json.loads(RUN_STATUS_PATH.read_text())
    return {"schema": "humanebench-decomposition-status/1", "conditions": {}, "events": []}


def save_status(status: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    RUN_STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")


def record_event(status: dict, kind: str, **fields) -> None:
    status["events"].append(
        {"at": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    )


# --- run ---------------------------------------------------------------------
def _log_matches_current_config(cond: dc.Condition, model: str, path: Path) -> str | None:
    """Return None if ``path`` was produced by the current config, else a reason.

    This gates resumption, and it has to be strict. ``inspect eval-retry``
    replays the task configuration recorded *in the log*, not the one on disk
    today -- so retrying a log written under a stale config silently perpetuates
    that config no matter what the task file now says. A log is only resumable
    if the things that define the experiment still match.
    """
    try:
        header = prov.read_eval_header(path)
    except Exception as exc:
        return f"unreadable header ({exc})"

    ev = header.get("eval") or {}
    if ev.get("model") != model:
        return f"logged model {ev.get('model')!r} != {model!r}"

    logged_ds = (ev.get("dataset") or {}).get("location") or ""
    if Path(logged_ds).name != cond.dataset_path.name:
        return (f"logged dataset {Path(logged_ds).name!r} != "
                f"{cond.dataset_path.name!r}")

    expected_prompt = dc.prompt_for(cond)
    for sample in prov.iter_eval_samples(path):
        sysmsg = next((m.get("content") for m in (sample.get("messages") or [])
                       if m.get("role") == "system"), None)
        if sysmsg != expected_prompt:
            return "logged system message differs from the current prompt"
        break
    else:
        return "log holds no samples"
    return None


def _resumable(cond: dc.Condition, model: str) -> tuple[Path | None, int]:
    """Find a resumable log for this cell and how many samples it already has."""
    model_dir = cond.log_dir / model.split("/")[-1]
    if not model_dir.is_dir():
        return None, 0
    best, best_n = None, -1
    for path in sorted(model_dir.glob("*.eval")):
        reason = _log_matches_current_config(cond, model, path)
        if reason is not None:
            print(f"  [{model.split('/')[-1]}] not resumable: {reason}")
            continue
        n = sum(1 for _ in prov.iter_eval_samples(path))
        if n > best_n:
            best, best_n = path, n
    return best, max(best_n, 0)


def _run_one(cond: dc.Condition, model: str) -> dict:
    """Run one model, resuming an existing partial log where possible.

    Without this, a run killed partway is unrecoverable in practice: the
    orchestrator resumes per *condition*, so re-running relaunches every model
    and re-pays for the ones that already finished. Resuming per *model* makes
    a kill cost only the samples that had not been generated yet.

    The subprocess cwd is pinned to the repo root explicitly; os.chdir is
    process-global and would race across the worker threads.
    """
    short = model.split("/")[-1]
    log_dir = cond.log_dir / short
    path, have = _resumable(cond, model)

    if path is not None and have >= cond.expected_samples:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] [{short}] already "
          f"complete ({have} samples); skipping", flush=True)
        return {"task_type": cond.task_type, "model": model, "success": True,
            "resumed": False, "skipped_complete": True}

    if path is not None and have > 0:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] [{short}] resuming "
          f"from {have}/{cond.expected_samples} samples", flush=True)
        cmd = ["inspect", "eval-retry", str(path),
           f"--log-dir={path.parent}", "--max-connections=10"]
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        return {"task_type": cond.task_type, "model": model,
            "success": rc == 0, "resumed": True, "resumed_from": have}

    # cwd is passed through to the subprocess; os.chdir was a race -- it is
    # process-global and this function runs on six threads at once.
    return run_evaluation(cond.task_type, model, log_dir, cwd=REPO_ROOT)


def run_condition(
    cond: dc.Condition, models: list[str], max_workers: int, status: dict
) -> dict:
    print(f"\n{'=' * 72}\n[{datetime.now().strftime('%H:%M:%S')}] "
          f"Condition {cond.label}  ({cond.task_type})")
    print(f"  {cond.expected_samples} samples x {len(models)} models   "
          f"est ${cond.est_cost_usd(len(models)):.2f}\n{'=' * 72}")

    started = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    # Submit in waves and top the pool up as completions land; the executor
    # never sees more than one wave of queued work at a time.
    wave = max(max_workers, 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one, cond, m): m for m in models[:wave]
        }
        queued = list(models[wave:])
        while futures:
            fut = next(concurrent.futures.as_completed(list(futures)))
            model_name = futures.pop(fut)
            try:
                results.append(fut.result())
            except Exception as exc:  # never lose the run's status to one model
                print(f"  ! {model_name} raised: {exc}")
                results.append({"model": model_name, "success": False,
                                "error": repr(exc)})
                record_event(status, "model_exception", condition=cond.task_type,
                             model=model_name, error=repr(exc))

            if queued:
                nxt = queued.pop(0)
                futures[ex.submit(_run_one, cond, nxt)] = nxt

    n_failed = sum(1 for r in results if not r.get("success"))
    print(f"  subprocesses: {len(results) - n_failed} ok, {n_failed} failed")

    exclude = load_excluded_ids(cond.dataset_path)
    retry_incomplete(cond, models, max_workers)
    moved = archive_superseded(cond, models, exclude)
    if moved:
        print(f"  archived {len(moved)} superseded .eval file(s) to attic/")

    return {
        "status": "ran",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_subprocess_failures": n_failed,
        "archived": moved,
    }


def run_smoke(conditions: list[dc.Condition], model: str) -> int:
    print(f"Smoke test: 1 sample per condition on {model}\n")
    failures = 0
    for cond in conditions:
        log_dir = SMOKE_LOG_DIR / cond.task_type
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["inspect", "eval", cond.task_file_rel, f"--model={model}",
               f"--log-dir={log_dir}", "--limit=1"]
        print(f"[{cond.label}] {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        path = latest_eval(log_dir)
        census = score_census(path) if path else {"n_samples": 0, "n_fully_scored": 0}
        ok = rc == 0 and census["n_fully_scored"] >= 1
        print(f"[{cond.label}] rc={rc} {census}  ->  {'PASS' if ok else 'FAIL'}\n")
        failures += 0 if ok else 1
    print(f"smoke: {len(conditions) - failures}/{len(conditions)} conditions passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=dc.TASK_TYPES, choices=dc.TASK_TYPES,
                    help="subset to run; execution order is always B -> C -> D -> E")
    ap.add_argument("--models", nargs="+", default=dc.MODELS)
    ap.add_argument("--max-workers", type=int, default=6,
                    help="parallel models within a condition (keep low: judge 429s "
                         "consume score_attempts and land as NaN)")
    ap.add_argument("--gate-threshold", type=float, default=0.98)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-model", default="openrouter/google/gemini-2.5-flash")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run every pre-launch check and exit WITHOUT spending. "
                         "Preflight and launch were previously the same command "
                         "distinguished only by the y/N answer, which made an "
                         "accidental launch a single keystroke away.")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="run unattended: suppress the interactive confirmation only")
    ap.add_argument("--allow-missing-slugs", action="store_true",
                    help="proceed even if an evaluated model's slug is not live on "
                         "OpenRouter. A missing *judge* slug always aborts.")
    ap.add_argument("--force", action="store_true",
                    help="re-run conditions already recorded complete in run_status.json")
    args = ap.parse_args()

    selected = [c for c in dc.CONDITIONS if c.task_type in set(args.conditions)]
    models = list(args.models)

    if args.smoke:
        return run_smoke(selected, args.smoke_model)

    missing = [c.task_file_rel for c in selected if not c.task_file.exists()]
    if missing:
        print(f"ERROR: missing task file(s): {missing}")
        return 2
    if any(c.scale == "subset" for c in selected) and not dc.SUBSET_DATASET_PATH.exists():
        print("ERROR: frozen subsample missing. Run scripts/build_decomposition_subsample.py")
        return 2

    # A condition declares its dataset twice -- here and inside its task file --
    # and nothing forces those to agree. A mismatch is silent: this runner would
    # report the size it expects and gate on that number while the eval read a
    # different file, or none. Checked before anything is spent.
    problems = dc.check_dataset_consistency(tuple(selected))
    if problems:
        print("ERROR: task file / condition dataset mismatch:")
        for pr in problems:
            print(f"  - {pr}")
        return 2

    total_cost = sum(c.est_cost_usd(len(models)) for c in selected)
    print("Goal-vs-tactics decomposition launch")
    print(f"  conditions : {', '.join(f'{c.label}={c.task_type}' for c in selected)}")
    print(f"  models     : {len(models)}")
    print(f"  est. total : ${total_cost:.2f}")

    if not args.skip_preflight:
        print("\nPre-flight")
        # No balance check, deliberately: on an auto-top-up account the balance
        # refills on demand and says nothing about whether a run can complete.
        # The budget cap is set on the OpenRouter account, server-side.
        judge_slugs = list(dc.JUDGE_MODELS)
        avail = check_model_availability(sorted(set(models) | set(judge_slugs)))
        if avail:
            dead = sorted(s for s, ok in avail.items() if not ok)
            print(f"  slugs: {len(avail) - len(dead)}/{len(avail)} live on OpenRouter")
            if dead:
                print("  ! NOT FOUND: " + ", ".join(dead))
            # A dead judge slug is never survivable: every sample it scores
            # fails, the ensemble returns NaN, and the condition burns its full
            # cost producing nothing usable. This aborts regardless of --yes.
            dead_judges = [s for s in dead if s in judge_slugs]
            if dead_judges:
                print(f"  ABORT: judge slug(s) unavailable: {', '.join(dead_judges)}. "
                      "Every sample would fail to score.")
                return 3
            if dead and not args.allow_missing_slugs:
                print("  ABORT: evaluated-model slug(s) unavailable. Re-run with "
                      "--allow-missing-slugs to proceed without them, or drop them "
                      "from --models.")
                return 3
        else:
            msg = ("OpenRouter /models returned nothing, so judge-slug "
                   "availability could not be checked")
            if not args.allow_missing_slugs:
                print(f"  ABORT: {msg}. A retired judge slug fails every sample. "
                      "Re-run with --allow-missing-slugs to proceed unchecked.")
                return 3
            print(f"  ! {msg}; proceeding (--allow-missing-slugs)")

    if args.preflight_only:
        print("\n--preflight-only: all checks passed, exiting without spending.")
        return 0

    if not args.yes:
        try:
            if input("\nLAUNCH and spend? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("aborted")
                return 1
        except EOFError:
            print("\nnon-interactive stdin and --yes not given; aborting")
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LAUNCH_MANIFEST_PATH.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prior = OUT_DIR / f"LAUNCH_MANIFEST.{stamp}.json"
        shutil.move(str(LAUNCH_MANIFEST_PATH), str(prior))
        print(f"\nprior launch manifest preserved -> {prior.relative_to(REPO_ROOT)}")
    manifest = build_launch_manifest(selected, models)
    LAUNCH_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"launch manifest -> {LAUNCH_MANIFEST_PATH.relative_to(REPO_ROOT)}  "
          "(written before the first API call)")

    status = load_status()
    # Keys written by the removed spend guard would otherwise be carried
    # forward verbatim and read as if they described this run.
    for stale in ("usage_baseline_usd", "max_spend_usd"):
        status.pop(stale, None)
    record_event(status, "launch", conditions=[c.task_type for c in selected],
                 n_models=len(models), est_cost_usd=round(total_cost, 2))
    save_status(status)

    t0 = time.time()
    for cond in selected:  # dc.CONDITIONS order == priority order
        # Resume: a condition already recorded complete is never re-run, so
        # restarting after a partial night costs nothing and cannot overwrite
        # good data with a fresh truncated run.
        prior = status["conditions"].get(cond.task_type) or {}
        if prior.get("completeness") == "complete" and not args.force:
            print(f"\nCondition {cond.label} ({cond.task_type}) already complete "
                  f"(gate {prior.get('gate', {}).get('worst_fraction_scored')}); "
                  "skipping. Use --force to re-run.")
            continue

        outcome = run_condition(cond, models, args.max_workers, status)
        if outcome["status"] != "ran":
            status["conditions"][cond.task_type] = outcome
            save_status(status)
            print(f"\nStopping: condition {cond.label} did not run "
                  f"({outcome['status']}). Earlier conditions stand.")
            break

        gate = gate_condition(cond, models, args.gate_threshold)
        outcome["gate"] = gate
        outcome["completeness"] = "complete" if gate["passed"] else "incomplete"
        status["conditions"][cond.task_type] = outcome
        save_status(status)

        print(f"  gate: worst model fully-scored fraction = "
              f"{gate['worst_fraction_scored']:.3f}  ->  "
              f"{'PASS' if gate['passed'] else 'FAIL'}")
        if not gate["passed"]:
            degraded = [m for m, r in gate["models"].items() if r["status"] != "ok"]
            print(f"  degraded models: {', '.join(degraded)}")
            record_event(status, "gate_fail", condition=cond.task_type, models=degraded)
            save_status(status)
            print(f"\nStopping after condition {cond.label}: below the "
                  f"{args.gate_threshold:.0%} completeness gate. It is recorded as "
                  "incomplete and must be excluded from, or explicitly caveated in, "
                  "the analysis.")
            break

    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min")
    print(f"status -> {RUN_STATUS_PATH.relative_to(REPO_ROOT)}")
    done = [t for t, o in status["conditions"].items() if o.get("completeness") == "complete"]
    print(f"complete conditions: {', '.join(done) if done else 'none'}")
    # Exit status must reflect what happened: a night that stopped on a failed
    # gate is not a success, and anything chained on $? (build_provenance, a
    # cron alert, the operator's own check) would otherwise read it as one.
    not_done = [c.task_type for c in selected
                if (status["conditions"].get(c.task_type) or {}).get("completeness")
                != "complete"]
    if not_done:
        print(f"INCOMPLETE conditions: {', '.join(not_done)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
