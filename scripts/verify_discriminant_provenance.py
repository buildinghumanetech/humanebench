#!/usr/bin/env python3
"""Verify the multi-label discriminant run against its frozen inputs.

The designed x measured matrix makes a claim the main benchmark does not: that
its diagonal *is* the main-run procedure, re-executed. That claim is only worth
anything if the judge received the same bytes it received in November, so this
script checks it rather than asserting it.

Checks, in the order they run:

 1. FRAME.        The 96 ids are 12 per principle and still a subset of the
                  frozen 200. If the parent was redrawn after the draw, every
                  nesting and pairing claim downstream is void.
 2. SCAFFOLDS.    The judge-prompt scaffold -- everything outside the scored
                  conversation -- hashes to the same eight values the reported
                  runs produced (tables/judge_blinding_check.csv). This is what
                  "rubric scaffolds reused verbatim" means operationally, and it
                  is exhaustive over the eight principles rather than a sample.
 3. PROMPTS.      Every judge prompt actually sent hashes to the value predicted
                  before launch in data/discriminant/expected_prompt_hashes.csv.
                  All 2,304, not a diagonal sample.
 4. DIAGONAL.     The 288 diagonal prompts are byte-identical to the archived
                  November prompts for the same (scenario, model, principle).
                  This is the one that licenses reading the diagonal as a
                  replication rather than as a similar procedure.
 5. ANALYTIC.     Exactly one judge call per sample, exactly eight samples per
                  response covering all eight principles, and no prompt naming
                  more than one principle. A single call rating eight would
                  manufacture the correlation structure being measured.
 6. JUDGE.        One judge, the expected slug, temperature 0.
 7. SERVING.      Which upstream stack answered. A slug is not a system; the
                  reported runs had to recover this forensically afterwards.
 8. CENSUS.       Samples carrying an on-scale severity, and residual failures.

Exit code is the number of failed checks, capped at 1, so it works as a gate.

Run from repo root, after the eval run:
    python scripts/verify_discriminant_provenance.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench import provenance as prov  # noqa: E402
from humanebench.bootstrap import PRINCIPLES  # noqa: E402
from humanebench.discriminant import (  # noqa: E402
    FRAME_JSONL,
    GLOBAL_RULES,
    HASHES_PATH,
    IDS_PATH,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    LOG_CONDITION,
    MANIFEST_PATH,
    PARENT_IDS_PATH,
    PER_PRINCIPLE,
    SOURCE_MODELS,
    scaffold_text,
)

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

CONV_START = "=== CONVERSATION START ==="
CONV_END = "=== CONVERSATION END ==="
VALID_SEVERITIES = {-1.0, -0.5, 0.5, 1.0}


class Report:
    def __init__(self) -> None:
        self.failed = 0
        self.skipped = 0

    def ok(self, msg: str) -> None:
        print(f"  {GREEN}PASS{RESET}  {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        print(f"  {RED}FAIL{RESET}  {msg}")

    def skip(self, msg: str) -> None:
        self.skipped += 1
        print(f"  {YELLOW}SKIP{RESET}  {msg}")

    def check(self, cond: bool, msg: str) -> None:
        self.ok(msg) if cond else self.fail(msg)


def _resolve(content, attachments: dict) -> str:
    if not isinstance(content, str):
        return ""
    if content.startswith("attachment://"):
        return attachments.get(content.split("://", 1)[1], "")
    return content


def judge_events(sample: dict) -> list[tuple[str, str, dict]]:
    """``(judge_model, judge_prompt, event)`` for each scoring call in a sample."""
    att = sample.get("attachments") or {}
    out = []
    for ev in sample.get("events") or []:
        if ev.get("event") != "model":
            continue
        for msg in ev.get("input") or []:
            if msg.get("role") != "user":
                continue
            text = _resolve(msg.get("content"), att)
            if GLOBAL_RULES in text:
                out.append((str(ev.get("model") or ""), text, ev))
                break
    return out


def scaffold_of(judge_prompt: str) -> str:
    """Strip the scored conversation, leaving only harness-controlled text."""
    if CONV_START not in judge_prompt or CONV_END not in judge_prompt:
        return judge_prompt
    return judge_prompt.split(CONV_START)[0] + judge_prompt.split(CONV_END)[1]


def archived_diagonal_prompts(logs_dir: Path, ids: set[str]) -> dict[tuple[str, str], str]:
    """``{(scenario_id, model): judge_prompt}`` from the reported baseline runs."""
    out: dict[tuple[str, str], str] = {}
    for model in SOURCE_MODELS:
        model_dir = logs_dir / "baseline" / model
        paths = sorted(model_dir.glob("*.eval"))
        if not paths:
            continue
        with zipfile.ZipFile(paths[0]) as z:
            for name in z.namelist():
                if not name.startswith("samples/"):
                    continue
                sample = json.loads(z.read(name))
                sid = sample.get("id")
                if sid not in ids:
                    continue
                found = judge_events(sample)
                if found:
                    out[(sid, model)] = found[0][1]
    return out


def load_run_samples(logs_dir: Path, models: list[str]) -> tuple[dict, list[str]]:
    """``({model: [sample, ...]}, missing_models)`` for the discriminant run."""
    by_model: dict[str, list[dict]] = {}
    missing: list[str] = []
    for model in models:
        model_dir = logs_dir / LOG_CONDITION / model
        paths = sorted(p for p in model_dir.glob("*.eval")) if model_dir.is_dir() else []
        if not paths:
            missing.append(model)
            continue
        # `attic/` holds superseded retries; glob above is non-recursive so they
        # are already excluded. If more than one remains, prefer the fuller file.
        best = max(paths, key=lambda p: (len(list(prov.iter_eval_samples(p))), p.name))
        by_model[model] = list(prov.iter_eval_samples(best))
    return by_model, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--models", nargs="+", default=list(SOURCE_MODELS))
    ap.add_argument("--blinding-csv", type=Path,
                    default=REPO_ROOT / "tables" / "judge_blinding_check.csv")
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found; run "
              "scripts/build_discriminant_multilabel_dataset.py first", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text())
    r = Report()

    # ---- 1. Frame -------------------------------------------------------
    print("Frame:")
    ids = [ln.strip() for ln in IDS_PATH.read_text().splitlines() if ln.strip()]
    id_set = set(ids)
    # Read the designed principle from the frame's `target`, not from the id
    # prefix. The prefix convention holds today; the target is the definition.
    designed = {}
    with FRAME_JSONL.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                designed[row["id"]] = row["target"]
    per_principle = Counter(designed.get(i) for i in ids)
    r.check(len(ids) == PER_PRINCIPLE * len(PRINCIPLES) and len(id_set) == len(ids),
            f"{len(ids)} unique scenario ids")
    r.check(set(per_principle.values()) == {PER_PRINCIPLE},
            f"{PER_PRINCIPLE} scenarios per principle across all {len(PRINCIPLES)}")
    parent = {ln.strip() for ln in PARENT_IDS_PATH.read_text().splitlines() if ln.strip()}
    r.check(id_set <= parent,
            f"nested inside the frozen {len(parent)} ({PARENT_IDS_PATH.name})")
    r.check(manifest["parent_ids_sha256"] == prov.file_sha256(PARENT_IDS_PATH),
            "parent frame unchanged since the draw")
    r.check(manifest["frame_ids_sha256"] == prov.file_sha256(IDS_PATH),
            "frame ids file unchanged since the manifest was written")
    r.check(manifest["expected_prompt_hashes_sha256"] == prov.file_sha256(HASHES_PATH),
            "expected-prompt-hash table unchanged since the manifest was written")

    expected: dict[tuple[str, str, str], str] = {}
    diagonal_keys: set[tuple[str, str, str]] = set()
    for row in csv.DictReader(HASHES_PATH.open()):
        key = (row["scenario_id"], row["source_model"], row["scored_principle"])
        expected[key] = row["expected_judge_prompt_sha256"]
        if row["is_diagonal"] == "True":
            diagonal_keys.add(key)
    r.check(len(expected) == manifest["n_judge_calls"],
            f"{len(expected):,} judge calls predicted before launch")

    # ---- 2. Scaffolds ---------------------------------------------------
    print("\nRubric scaffolds (reused verbatim?):")
    if args.blinding_csv.exists():
        archived_scaffolds: dict[str, set[str]] = defaultdict(set)
        for row in csv.DictReader(args.blinding_csv.open()):
            archived_scaffolds[row["principle"]].add(row["scaffold_sha256"])
        mismatched = []
        for p in PRINCIPLES:
            h = hashlib.sha256(scaffold_text(p).encode()).hexdigest()
            if archived_scaffolds.get(p) != {h}:
                mismatched.append(p)
        r.check(not mismatched,
                f"all {len(PRINCIPLES)} scaffold hashes identical to the reported runs"
                + (f" (differs: {mismatched})" if mismatched else ""))
    else:
        r.skip(f"{args.blinding_csv.name} absent; run "
               "scripts/compute_judge_artifact_controls.py to enable this check")

    # ---- run logs -------------------------------------------------------
    by_model, missing = load_run_samples(args.logs_dir, args.models)
    if missing:
        print(f"\n{YELLOW}The discriminant run has not produced logs for: "
              f"{missing}{RESET}")
    if not by_model:
        print("\nNo discriminant run on disk. Checks 3-8 cannot run.")
        print(f"\n{r.failed} failed, {r.skipped} skipped")
        return min(r.failed, 1) if r.failed else 0

    # ---- 3-6. Prompts, diagonal, analytic scoring, judge ----------------
    print("\nJudge prompts actually sent:")
    seen: set[tuple[str, str, str]] = set()
    prompt_mismatch: list[tuple[str, str, str]] = []
    multi_call: list[str] = []
    wrong_judge: Counter = Counter()
    scaffold_hashes_seen: set[str] = set()
    principles_per_response: dict[tuple[str, str], set[str]] = defaultdict(set)
    sent_prompts: dict[tuple[str, str, str], str] = {}
    n_calls = 0

    for model, samples in by_model.items():
        for sample in samples:
            meta = (sample.get("metadata") or {}).get("metadata") or {}
            scenario = meta.get("scenario_id")
            scored = meta.get("scored_principle") or sample.get("target")
            if scenario is None:
                # Fall back to the composite id if metadata is unexpectedly bare.
                parts = str(sample.get("id") or "").split("|")
                scenario = parts[0] if parts else None
            key = (scenario, model, scored)
            calls = judge_events(sample)
            n_calls += len(calls)
            if len(calls) != 1:
                multi_call.append(f"{sample.get('id')} ({len(calls)} calls)")
            for judge_model, prompt, _ev in calls:
                wrong_judge[judge_model] += 1
                scaffold_hashes_seen.add(
                    hashlib.sha256(scaffold_of(prompt).encode()).hexdigest())
                got = hashlib.sha256(prompt.encode()).hexdigest()
                if expected.get(key) != got:
                    prompt_mismatch.append(key)
                sent_prompts[key] = prompt
            seen.add(key)
            principles_per_response[(scenario, model)].add(scored)

    r.check(not prompt_mismatch,
            f"{len(seen):,} judge prompts hash to their pre-launch predictions"
            + (f" ({len(prompt_mismatch)} mismatched)" if prompt_mismatch else ""))
    unseen = set(expected) - seen
    r.check(not unseen, f"every predicted call is present in the logs"
            + (f" ({len(unseen)} missing)" if unseen else ""))

    print("\nDiagonal vs the reported November runs:")
    archived = archived_diagonal_prompts(args.logs_dir, id_set)
    if not archived:
        r.skip("baseline logs unavailable; diagonal byte-equality not tested")
    else:
        diff = [k for k in sorted(diagonal_keys)
                if k in sent_prompts
                and archived.get((k[0], k[1])) is not None
                and archived[(k[0], k[1])] != sent_prompts[k]]
        covered = [k for k in diagonal_keys
                   if k in sent_prompts and (k[0], k[1]) in archived]
        r.check(not diff,
                f"{len(covered)} diagonal prompts byte-identical to November"
                + (f" ({len(diff)} differ)" if diff else ""))
        uncovered = len(diagonal_keys) - len(covered)
        if uncovered:
            print(f"        note: {uncovered} diagonal cell(s) have no archived "
                  "counterpart (a judge failed before this judge was called in "
                  "the reported run); they are scored here but have no main-run "
                  "comparator")

    print("\nAnalytic scoring (one call per principle):")
    r.check(not multi_call,
            f"exactly one judge call per sample across {n_calls:,} calls"
            + (f" (offenders: {multi_call[:3]})" if multi_call else ""))
    bad_coverage = {k: sorted(v) for k, v in principles_per_response.items()
                    if len(v) != len(PRINCIPLES)}
    r.check(not bad_coverage,
            f"all {len(principles_per_response)} responses scored against all "
            f"{len(PRINCIPLES)} principles"
            + (f" ({len(bad_coverage)} incomplete)" if bad_coverage else ""))
    r.check(scaffold_hashes_seen == {
        hashlib.sha256(scaffold_text(p).encode()).hexdigest() for p in PRINCIPLES},
        f"{len(scaffold_hashes_seen)} distinct scaffolds in the run -- one per "
        "principle, no prompt naming two")

    print("\nJudge:")
    r.check(set(wrong_judge) == {JUDGE_MODEL},
            f"every call went to {JUDGE_MODEL} (saw: {dict(wrong_judge)})")

    # ---- 7. Serving provenance ------------------------------------------
    print("\nServing provenance (a slug is not a system):")
    providers: Counter = Counter()
    for samples in by_model.values():
        for sample in samples:
            for _m, _p, ev in judge_events(sample):
                resp = (ev.get("call") or {}).get("response")
                if isinstance(resp, dict):
                    providers[str(resp.get("provider"))] += 1
    if providers:
        total = sum(providers.values())
        for name, count in providers.most_common():
            print(f"        {name}: {count / total:.1%}")
        r.check(len(providers) >= 1,
                f"{len(providers)} upstream provider(s) answered the judge")
    else:
        r.skip("no provider recorded in the raw responses")

    # ---- 8. Census -------------------------------------------------------
    print("\nCompleteness:")
    expected_per_model = manifest["n_scenarios"] * manifest["n_principles"]
    for model, samples in sorted(by_model.items()):
        n_full = n_fail = n_offscale = n_nan = 0
        for sample in samples:
            overseer = (sample.get("scores") or {}).get("overseer")
            if not overseer:
                n_fail += 1
                continue
            meta = overseer.get("metadata") or {}
            individual = meta.get("individual_scores")
            judges = meta.get("ensemble_models")
            if not individual or not judges or len(individual) != len(judges):
                n_fail += 1
                continue
            if any(s not in VALID_SEVERITIES for s in individual):
                n_offscale += 1
                continue
            n_full += 1
            value = overseer.get("value")
            if isinstance(value, float) and value != value:  # NaN
                n_nan += 1
        detail = f"{n_full}/{expected_per_model} on-scale"
        if n_fail or n_offscale or n_nan:
            detail += f"  (judge failures {n_fail}, off-scale {n_offscale}, " \
                      f"invalid-flagged {n_nan})"
        r.check(n_full == expected_per_model, f"{model:22s} {detail}")

    print(f"\n{r.failed} failed, {r.skipped} skipped")
    if r.failed:
        print(f"{RED}Do not analyse this run until these are resolved.{RESET}")
    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
