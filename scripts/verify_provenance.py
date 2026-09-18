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

    # ---- Summary ----------------------------------------------------------
    print()
    if r.failed:
        print(f"{RED}PROVENANCE FAILED{RESET}: {r.failed} check(s) failed, "
              f"{r.skipped} skipped.")
        return 1
    print(f"{GREEN}PROVENANCE VERIFIED{RESET}: all checks passed"
          + (f" ({r.skipped} skipped)." if r.skipped else "."))
    return 0


def manifest_rel(path: Path) -> str:
    try:
        return str(path.relative_to(prov.REPO_ROOT))
    except ValueError:
        # When verifying an extracted Zenodo dir, match on the tail the manifest
        # stored (logs/<persona>/<model>/<file>).
        parts = path.parts
        for i, p in enumerate(parts):
            if p in prov.PERSONAS:
                return str(Path("logs", *parts[i:]))
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
