#!/usr/bin/env python3
"""Independently verify HumaneBench provenance.

Recomputes every claim in provenance/MANIFEST.json from the logs + repo and the
constants in humanebench/provenance.py, and exits non-zero on any mismatch. This
turns the provenance argument from "trust us" into "run this script". It needs
no network and no third-party services.

Usage:
    python scripts/verify_provenance.py
    python scripts/verify_provenance.py --logs-dir /path/to/extracted/zenodo/logs

Checks (all must pass):
    1. every reported run's embedded prompts hash to the frozen prompt hash
    2. every run's file sha256 matches the manifest (no silent tampering)
    3. every run's eval.created postdates the freeze commit
    4. the current dataset's prompts hash to the frozen prompt hash
    5. each post-freeze dataset commit preserves the frozen prompt hash (git)
    6. the freeze commit's tree hashes to the frozen prompt hash (git)
    7. the on-disk run set matches the manifest's run set

Decomposition runs (the goal-vs-tactics robustness analysis) are verified in a
separate block so no reported-run count or published aggregate widens:
    8. the frozen 200-scenario subsample hashes to its pinned subset hash, and
       every subsample id is in the frozen dataset and not flagged out of analysis
    9. each decomposition run hashes to the frozen set (full-scale) or the frozen
       subset (subset-scale) -- a subset cannot reproduce the whole-set hash
   10. every (id, input, target) triple a decomposition run scored is
       byte-identical to the frozen dataset's triple for that id
   11. each decomposition run's file sha256 matches the manifest and its
       creation timestamp postdates the freeze
   12. the on-disk decomposition run set matches the manifest's

git-dependent checks (5, 6) are reported as SKIP when run outside the repo, e.g.
against a bare Zenodo download; the content-binding checks (1-4, 7) still hold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanebench import provenance as prov  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", type=Path, default=prov.LOGS_DIR)
    ap.add_argument("--manifest", type=Path,
                    default=prov.REPO_ROOT / "provenance" / "MANIFEST.json")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest}; run "
              f"scripts/build_provenance.py first.", file=sys.stderr)
        return 2
    manifest = json.loads(args.manifest.read_text())
    by_path = {r["path"]: r for r in manifest["reported_runs"]}

    r = Report()
    print(f"Frozen prompt hash: {prov.FROZEN_PROMPT_HASH}")
    print(f"Freeze commit:      {prov.FREEZE_COMMIT} ({prov.FREEZE_COMMIT_DATE})\n")

    # ---- Checks 1-3, 7: reported runs ------------------------------------
    print("Reported runs (content-binding):")
    disk_runs = prov.reported_runs(args.logs_dir)
    disk_paths = set()
    if not disk_runs:
        r.skip(f"no .eval runs found under {args.logs_dir} (pass --logs-dir to "
               f"point at the published logs)")
    for persona, model, path in disk_runs:
        rel = manifest_rel(path)
        disk_paths.add(rel)
        entry = by_path.get(rel)
        prompt_hash, n = prov.eval_prompt_hash(path)
        r.check(prompt_hash == prov.FROZEN_PROMPT_HASH,
                f"{persona}/{model}: {n} prompts hash to frozen set")
        if entry is None:
            r.fail(f"{persona}/{model}: present on disk but missing from manifest")
            continue
        r.check(prov.file_sha256(path) == entry["file_sha256"],
                f"{persona}/{model}: file sha256 matches manifest")
        created = prov.read_eval_header(path)["eval"].get("created")
        r.check(bool(created) and prov.created_after_freeze(created),
                f"{persona}/{model}: created {created} postdates freeze")

    # check 7: manifest runs all present on disk (only when we have a log dir)
    if disk_runs:
        missing = sorted(set(by_path) - disk_paths)
        r.check(not missing,
                f"all {len(by_path)} manifest runs present on disk"
                + (f" (missing: {missing})" if missing else ""))

    # ---- Check 4: current dataset ----------------------------------------
    print("\nFinalized dataset:")
    cur_hash, cur_n = prov.prompt_hash_from_jsonl(prov.DATASET_PATH.read_text())
    r.check(cur_hash == prov.FROZEN_PROMPT_HASH,
            f"current data/humane_bench.jsonl ({cur_n} prompts) hashes to frozen set")

    # ---- Checks 5-6: git history -----------------------------------------
    print("\nDataset git history:")
    if not prov.git_available():
        r.skip("git history unavailable (running outside the repo) -- skipping "
               "freeze-commit and post-freeze-commit checks")
    else:
        if prov.git_commit_present(prov.FREEZE_COMMIT):
            text = prov.git_file_at_commit(prov.FREEZE_COMMIT)
            ph, _ = prov.prompt_hash_from_jsonl(text)
            r.check(ph == prov.FROZEN_PROMPT_HASH,
                    f"freeze commit {prov.FREEZE_COMMIT[:12]} tree hashes to frozen set")
        else:
            r.skip(f"freeze commit {prov.FREEZE_COMMIT[:12]} not in this clone")
        for sha, label in prov.POST_FREEZE_DATASET_COMMITS:
            if not prov.git_commit_present(sha):
                r.skip(f"post-freeze commit {sha[:12]} not in this clone ({label})")
                continue
            ph, _ = prov.prompt_hash_from_jsonl(prov.git_file_at_commit(sha))
            r.check(ph == prov.FROZEN_PROMPT_HASH,
                    f"post-freeze commit {sha[:12]} preserves prompts ({label})")

    # ---- Checks 8-12: decomposition runs ----------------------------------
    # These are a robustness analysis of the adversarial condition, verified in
    # their own block so that no reported-run count or aggregate widens.
    verify_decomposition(r, manifest, args.logs_dir)

    # ---- Summary ----------------------------------------------------------
    print()
    if r.failed:
        print(f"{RED}PROVENANCE FAILED{RESET}: {r.failed} check(s) failed, "
              f"{r.skipped} skipped.")
        return 1
    print(f"{GREEN}PROVENANCE VERIFIED{RESET}: all checks passed"
          + (f" ({r.skipped} skipped)." if r.skipped else "."))
    return 0


def verify_decomposition(r: Report, manifest: dict, logs_dir: Path) -> None:
    """Checks 8-12 for the goal-vs-tactics decomposition runs.

    A subset run cannot reproduce the frozen prompt hash by construction, so the
    content-binding guarantee is carried by the frozen *subset* hash plus a
    per-triple identity test against the frozen dataset -- jointly stronger than
    the aggregate digest alone, since they also prove no prompt text was edited.
    """
    from humanebench import decomposition as dc

    print("\nDecomposition runs (goal-vs-tactics robustness analysis):")

    entries = manifest.get("decomposition_runs")
    if entries is None:
        r.skip("manifest predates the decomposition schema; nothing to verify")
        return

    try:
        disk_runs = prov.decomposition_runs(logs_dir)
    except RuntimeError as exc:  # >1 .eval in a model dir
        r.fail(str(exc))
        return

    # The frozen subsample is verified whenever it exists, independently of
    # whether any run has been executed against it yet: it is the artifact the
    # C/D/E contrasts are pinned to, so it should be attested before launch.
    anchors = (manifest.get("anchors") or {}).get("decomposition") or {}
    if dc.SUBSET_SUMMARY_PATH.exists():
        summary = dc.load_subset_summary()
        r.check(summary.get("subset_prompt_hash") == dc.SUBSET_PROMPT_HASH,
                "frozen subsample hash matches the pinned constant")
        subset_ids = set(dc.load_subset_ids())
        r.check(len(subset_ids) == summary.get("n_scenarios"),
                f"subsample ids file holds {len(subset_ids)} unique ids")
        recomputed, n = prov.prompt_hash_from_jsonl(dc.SUBSET_DATASET_PATH.read_text())
        r.check(recomputed == dc.SUBSET_PROMPT_HASH,
                f"subsample dataset ({n} prompts) hashes to the frozen subset hash")
        if anchors.get("subset_dataset_sha256"):
            r.check(prov.file_sha256(dc.SUBSET_DATASET_PATH) == anchors["subset_dataset_sha256"],
                    "subsample dataset sha256 matches manifest")
        frozen_ids = set(prov.frozen_triples())
        r.check(subset_ids <= frozen_ids, "every subsample id exists in the frozen dataset")
        try:
            from humanebench.excluded import load_excluded_ids
            r.check(not (subset_ids & load_excluded_ids()),
                    "no subsample id is flagged out of analysis")
        except Exception as exc:  # pragma: no cover - dataset always present in repo
            r.skip(f"excluded-id check unavailable: {exc}")
    else:
        r.skip("frozen subsample not present; skipping subsample checks")

    if not disk_runs and not entries:
        r.skip("no decomposition runs recorded or on disk yet")
        return

    by_path = {e["path"]: e for e in entries}
    disk_paths: set[str] = set()
    triples = prov.frozen_triples()

    for condition, model, path in disk_runs:
        rel = manifest_rel(path)
        disk_paths.add(rel)
        entry = by_path.get(rel)
        if entry is None:
            # A run that is still generating is absent from the manifest by
            # construction -- the manifest is rebuilt after runs finish. Calling
            # that a provenance failure would leave the verifier red for the
            # whole of every run, which is the fastest way to teach people to
            # ignore it. Only a COMPLETE run missing from the manifest is a real
            # failure: that one means the manifest needs rebuilding.
            spec = dc.CONDITIONS_BY_TASK_TYPE.get(condition)
            n_have = sum(1 for _ in prov.iter_eval_samples(path))
            if spec is not None and n_have < spec.expected_samples:
                r.skip(f"{condition}/{model}: in progress "
                       f"({n_have}/{spec.expected_samples} samples), not yet in manifest")
            else:
                r.fail(f"{condition}/{model}: complete but missing from manifest "
                       f"-- rerun scripts/build_provenance.py")
            continue

        spec = dc.CONDITIONS_BY_TASK_TYPE[condition]
        expected = dc.expected_prompt_hash(spec)
        prompt_hash, n = prov.eval_prompt_hash(path)
        label = "frozen set" if spec.scale == "full" else "frozen subset"
        r.check(prompt_hash == expected,
                f"{condition}/{model}: {n} prompts hash to the {label}")

        mismatched = unknown = 0
        for sample in prov.iter_eval_samples(path):
            sid = sample.get("id")
            if sid not in triples:
                unknown += 1
            elif (sample.get("input"), sample.get("target")) != triples[sid]:
                mismatched += 1
        r.check(not mismatched and not unknown,
                f"{condition}/{model}: all {n} (id, input, target) triples are "
                f"byte-identical to the frozen dataset"
                + (f" ({mismatched} altered, {unknown} unknown)" if (mismatched or unknown) else ""))

        r.check(prov.file_sha256(path) == entry["file_sha256"],
                f"{condition}/{model}: file sha256 matches manifest")
        created = prov.read_eval_header(path)["eval"].get("created")
        r.check(bool(created) and prov.created_after_freeze(created),
                f"{condition}/{model}: created {created} postdates freeze")

    missing = sorted(set(by_path) - disk_paths)
    r.check(not missing,
            f"all {len(by_path)} manifest decomposition runs present on disk"
            + (f" (missing: {missing})" if missing else ""))


def manifest_rel(path: Path) -> str:
    try:
        return str(path.relative_to(prov.REPO_ROOT))
    except ValueError:
        # When verifying an extracted Zenodo dir, match on the tail the manifest
        # stored (logs/<condition>/<model>/<file>).
        from humanebench.decomposition import TASK_TYPES as _DECOMP

        known = set(prov.PERSONAS) | set(_DECOMP)
        parts = path.parts
        for i, p in enumerate(parts):
            if p in known:
                return str(Path("logs", *parts[i:]))
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
