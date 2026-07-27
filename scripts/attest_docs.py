#!/usr/bin/env python3
"""Attest the content of a document without publishing it.

Some working documents establish something about *when* a decision was made --
`results/decomposition_precommitment.md` fixes the interpretation of a run's
two possible outcomes before the run existed -- but are not themselves meant
for publication, and `results/` is gitignored, so they carry no commit of their
own and their mtimes prove nothing.

This records the sha256 of such a document in a tracked manifest. The document
stays private; the hash goes into git. Anyone can later be shown the document
and check it against the hash committed here.

    Evidentiary weight comes from the COMMIT that carries this manifest, not
    from the `attested_at` field. `attested_at` is self-asserted and a liar
    could set it to anything; the commit date and the tree it points at are
    what make backdating hard. Quote the commit, not the timestamp.

Usage:
    python scripts/attest_docs.py results/decomposition_precommitment.md
    python scripts/attest_docs.py /abs/path/to/doc.md --as results/doc.md
    python scripts/attest_docs.py --verify

Re-attesting a document whose content has not changed is a no-op: the original
`attested_at` survives, because the earliest attestation is the evidence. If
the content HAS changed, the previous record is pushed onto that entry's
`superseded` list -- records are added and amended, never dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "provenance" / "doc_attestations.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str | None:
    """Run a git command in the repo; None if git is unavailable or fails."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def main_worktree_root() -> Path | None:
    """Path of the repository's primary worktree.

    Attestations are run from whichever worktree is convenient, but the stored
    path must be the same string either way, so it is always resolved against
    the primary worktree.
    """
    listing = git("worktree", "list", "--porcelain")
    if not listing:
        return None
    for line in listing.splitlines():
        if line.startswith("worktree "):
            return Path(line[len("worktree "):])
    return None


def logical_path(target: Path, override: str | None) -> str:
    """Repo-relative POSIX path to store for `target`.

    Absolute paths are never stored: they leak a username, and this manifest
    ships inside an anonymized supplementary package.
    """
    if override:
        if Path(override).is_absolute():
            raise SystemExit(f"--as must be repo-relative, got: {override}")
        return Path(override).as_posix()

    resolved = target.resolve()
    for root in (main_worktree_root(), REPO_ROOT):
        if root is None:
            continue
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    raise SystemExit(
        f"{target} is outside the repository; pass --as <repo-relative-path> "
        "to say where it logically lives. Absolute paths are not stored."
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"description": DESCRIPTION, "attestations": []}
    with path.open() as fh:
        data = json.load(fh)
    data.setdefault("attestations", [])
    return data


def write_manifest(path: Path, data: dict) -> None:
    data["description"] = DESCRIPTION
    data["attestations"].sort(key=lambda e: e["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


DESCRIPTION = (
    "sha256 attestations for working documents that are not published. "
    "The document content is not in this repository; only its hash is. "
    "Priority is established by the commit carrying a given entry, not by "
    "the self-asserted attested_at field. Written by scripts/attest_docs.py."
)


def attest(manifest: dict, target: Path, stored: str) -> str:
    """Add or amend one entry. Returns a one-word outcome for the caller."""
    digest = sha256_of(target)
    size = target.stat().st_size
    entries = manifest["attestations"]
    existing = next((e for e in entries if e["path"] == stored), None)

    if existing is not None and existing["sha256"] == digest:
        return "unchanged"

    record = {
        "path": stored,
        "sha256": digest,
        "size_bytes": size,
        "attested_at": utc_now(),
        "git_head": git("rev-parse", "HEAD"),
    }

    if existing is None:
        entries.append(record)
        return "added"

    # Content changed: keep the old record so the earlier claim is still
    # readable, and so a reader can see that it was revised rather than
    # silently swapped.
    superseded = existing.pop("superseded", [])
    superseded.append({k: v for k, v in existing.items()})
    record["superseded"] = superseded
    entries[entries.index(existing)] = record
    return "amended"


def verify(manifest: dict) -> int:
    """Recheck every attested document that is present on this machine."""
    root = main_worktree_root() or REPO_ROOT
    failures = 0
    if not manifest["attestations"]:
        print("no attestations recorded")
        return 0
    for entry in manifest["attestations"]:
        target = root / entry["path"]
        if not target.exists():
            print(f"  SKIP  {entry['path']} (not present on this machine)")
            continue
        actual = sha256_of(target)
        if actual == entry["sha256"]:
            print(f"  PASS  {entry['path']}")
        else:
            failures += 1
            print(f"  FAIL  {entry['path']}")
            print(f"        attested {entry['sha256']}")
            print(f"        actual   {actual}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("docs", nargs="*", type=Path, help="documents to attest")
    ap.add_argument(
        "--as",
        dest="stored_as",
        help="repo-relative path to record (required for docs outside the repo)",
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="recheck attested documents against the manifest and exit",
    )
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)

    if args.verify:
        if args.docs:
            ap.error("--verify takes no document arguments")
        return 1 if verify(manifest) else 0

    if not args.docs:
        ap.error("give at least one document, or --verify")
    if args.stored_as and len(args.docs) > 1:
        ap.error("--as applies to a single document")

    for doc in args.docs:
        if not doc.is_file():
            raise SystemExit(f"not a file: {doc}")
        stored = logical_path(doc, args.stored_as)
        outcome = attest(manifest, doc, stored)
        print(f"  {outcome:<9} {stored}")

    write_manifest(args.manifest, manifest)
    try:
        shown = args.manifest.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = args.manifest  # a manifest outside the repo, e.g. under test
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
