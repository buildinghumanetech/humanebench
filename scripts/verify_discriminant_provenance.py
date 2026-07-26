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

Exit code: 0 only if every check ran and passed. 1 on any failure, including
an absent run -- "nothing was checked" is not a pass. 2 if checks were skipped
for missing inputs, unless --allow-skips says that is acceptable.

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

    def exit_code(self, allow_skips: bool) -> int:
        """Non-zero unless every check actually ran and passed.

        A skip is not a pass. The two checks that establish the paper's
        byte-equality claim both skip when their inputs are absent -- a missing
        `judge_blinding_check.csv`, an archived-off `logs/baseline/` -- and if
        skips did not affect the exit code this script would exit 0 having
        verified nothing, while the generated report asserts those exact checks
        were performed. `--allow-skips` exists for the case where an operator
        has decided a skip is acceptable and is saying so out loud.
        """
        if self.failed:
            return 1
        return 0 if (self.skipped == 0 or allow_skips) else 2


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


def archived_diagonal_prompts(
    logs_dir: Path, ids: set[str], models: list[str]
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """``({(scenario_id, model): judge_prompt}, problems)`` from the baseline runs.

    Enforces the same two guards ``build_discriminant_multilabel_dataset`` applies
    to this archive, because the datasets were derived under them and an
    attestation made against a different file is worth nothing:

    * exactly one ``.eval`` per model directory -- a stray retry or restored
      archive means we might read a file the datasets did not come from;
    * all of a sample's archived judge prompts identical -- the ensemble was sent
      one prompt, so taking the first without checking hides a divergence.

    Problems are returned rather than raised so the caller can FAIL the specific
    check instead of aborting the whole verification.
    """
    out: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    for model in models:
        model_dir = logs_dir / "baseline" / model
        paths = sorted(model_dir.glob("*.eval"))
        if not paths:
            problems.append(f"{model}: no archived baseline .eval")
            continue
        if len(paths) != 1:
            problems.append(
                f"{model}: {len(paths)} baseline .eval files; the builder requires "
                "exactly one, so which archive the datasets came from is ambiguous"
            )
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
                if not found:
                    continue
                prompts = {p for _m, p, _e in found}
                if len(prompts) != 1:
                    problems.append(
                        f"{model}/{sid}: {len(prompts)} distinct archived judge "
                        "prompts; the ensemble must have been sent one"
                    )
                    continue
                out[(sid, model)] = found[0][1]
    return out, problems


def _n_scored_samples(path: Path) -> int:
    """Samples with a full, on-scale severity -- the analysis's admission rule."""
    n = 0
    for sample in prov.iter_eval_samples(path):
        overseer = (sample.get("scores") or {}).get("overseer")
        if not overseer:
            continue
        meta = overseer.get("metadata") or {}
        individual, judges = meta.get("individual_scores"), meta.get("ensemble_models")
        if not individual or not judges or len(individual) != len(judges):
            continue
        if all(s in VALID_SEVERITIES for s in individual):
            n += 1
    return n


def _cells_without_judge_row(
    raw_csv: Path, ids: set[str], models: list[str]
) -> set[tuple[str, str]]:
    """(scenario, model) pairs the reported run never scored with this judge.

    The scorer early-returns on the first judge failure, so if an earlier
    ensemble member failed, this judge was never called and no severity row
    exists. Those cells legitimately have no November comparator. Any *other*
    missing archive is a broken archive, and the two must not be conflated.
    """
    if not raw_csv.exists():
        return set()
    judge = JUDGE_MODEL.rsplit("/", 1)[-1]
    present: set[tuple[str, str]] = set()
    with raw_csv.open() as fh:
        for row in csv.DictReader(fh):
            if (row.get("persona") == "baseline"
                    and row.get("judge_name") == judge
                    and row.get("model") in models
                    and row.get("sample_id") in ids):
                present.add((row["sample_id"], row["model"]))
    return {(s, m) for s in ids for m in models} - present


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
        # are already excluded. If more than one remains, pick by SCORED sample
        # count -- the same rule the runner's gate and the analysis use, so all
        # three read the same file. Ranking on raw sample count instead lets a
        # fuller-but-less-scored retry win here and lose there.
        best = max(paths, key=lambda p: (_n_scored_samples(p), p.name))
        by_model[model] = list(prov.iter_eval_samples(best))
    return by_model, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--models", nargs="+", default=list(SOURCE_MODELS))
    ap.add_argument("--allow-skips", action="store_true",
                    help="exit 0 even when checks were skipped for missing "
                         "inputs. Only pass this if you have decided the skip "
                         "is acceptable -- a skipped check verified nothing.")
    ap.add_argument("--raw-csv", type=Path,
                    default=REPO_ROOT / "tables" / "inter_judge_raw_regenerated.csv",
                    help="reported-run per-judge severities, used to tell a real "
                         "judge failure from a missing archive")
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
        # Absent-run is a FAILED gate, not a passed one. Returning 0 here made
        # `verify && compute` treat "nothing was checked" as "provenance
        # verified": zero judge prompts hashed, the diagonal never compared to
        # November, the judge slug never confirmed.
        print("\nNo discriminant run on disk. Checks 3-8 cannot run.")
        print(f"\n{r.failed} failed, {r.skipped} skipped")
        print(f"{RED}Nothing was verified. This is not a pass.{RESET}")
        return 1

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

    # Count prompts actually HASHED, not samples seen. A sample with no recorded
    # judge event contributes to `seen` but to nothing that was checked, so
    # ranking on `seen` would pass this check on a log where zero prompts were
    # ever compared -- the vacuous pass this whole section exists to prevent.
    r.check(not prompt_mismatch and n_calls == len(expected),
            f"{n_calls:,} of {len(expected):,} judge prompts hash to their "
            "pre-launch predictions"
            + (f" ({len(prompt_mismatch)} mismatched)" if prompt_mismatch else "")
            + (f" -- {len(expected) - n_calls} prompt(s) never recorded, so they "
               "were not checked at all" if n_calls != len(expected) else ""))
    unseen = set(expected) - seen
    r.check(not unseen, f"every predicted call is present in the logs"
            + (f" ({len(unseen)} missing)" if unseen else ""))

    print("\nDiagonal vs the reported November runs:")
    # Cells the reported run genuinely never sent to this judge: an earlier
    # ensemble judge failed, so the scorer early-returned before reaching it.
    # These are the only legitimate gaps, and they are checkable rather than
    # assumed -- read from the reported run's own per-judge severity table.
    known_missing_comparators = _cells_without_judge_row(
        args.raw_csv, id_set, args.models)
    archived, archive_problems = archived_diagonal_prompts(
        args.logs_dir, id_set, args.models)
    for problem in archive_problems:
        r.fail(f"archived baseline unusable -- {problem}")
    if not archived and not archive_problems:
        r.skip("baseline logs unavailable; diagonal byte-equality not tested")
    elif archived:
        diff = [k for k in sorted(diagonal_keys)
                if k in sent_prompts
                and archived.get((k[0], k[1])) is not None
                and archived[(k[0], k[1])] != sent_prompts[k]]
        covered = [k for k in diagonal_keys
                   if k in sent_prompts and (k[0], k[1]) in archived]
        r.check(not diff and len(covered) > 0,
                f"{len(covered)} diagonal prompts byte-identical to November"
                + (f" ({len(diff)} differ)" if diff else "")
                + (" -- ZERO compared, which verifies nothing" if not covered else ""))

        # Coverage is its own check. Previously an uncovered cell printed a note
        # blaming a judge failure -- but an absent or renamed archive produces
        # exactly the same gap, so the note asserted a cause it had not
        # established, and the check passed on whatever subset was on disk.
        # A judge failure is verifiable: the reported run's own score table has
        # no gpt-5.1 row for that cell.
        uncovered = sorted(k for k in diagonal_keys if (k[0], k[1]) not in archived)
        expected_gaps = {k for k in uncovered
                         if (k[0], k[1]) in known_missing_comparators}
        unexplained = [k for k in uncovered if k not in expected_gaps]
        r.check(
            not unexplained,
            f"every diagonal cell has an archived counterpart, except "
            f"{len(expected_gaps)} where the reported run recorded no "
            f"{JUDGE_MODEL.rsplit('/', 1)[-1]} call"
            + (f" ({len(unexplained)} unexplained -- archive incomplete, NOT a "
               "judge failure)" if unexplained else ""),
        )
        for k in unexplained[:3]:
            print(f"        unexplained gap: {k[0]} / {k[1]}")

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
    if r.skipped and not args.allow_skips:
        print(f"{YELLOW}Skipped checks verified nothing; exiting non-zero. "
              f"Pass --allow-skips to accept them explicitly.{RESET}")
    if r.failed:
        print(f"{RED}Do not analyse this run until these are resolved.{RESET}")
    return r.exit_code(args.allow_skips)


if __name__ == "__main__":
    raise SystemExit(main())
