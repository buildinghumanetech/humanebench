#!/usr/bin/env python3
"""Generate the HumaneBench provenance manifest.

Binds every reported eval run to the finalized dataset by content hash, proving
each run scored prompts byte-identical to the frozen prompt set. Writes:

    provenance/MANIFEST.json   machine-readable, committed
    provenance/MANIFEST.md     human-readable summary

Usage:
    python scripts/build_provenance.py
    python scripts/build_provenance.py --logs-dir /path/to/extracted/zenodo/logs

See PROVENANCE.md for the narrative and how to verify independently.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanebench import provenance as prov  # noqa: E402

OUT_DIR = prov.REPO_ROOT / "provenance"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(prov.REPO_ROOT))
    except ValueError:
        return str(path)


def build_anchors() -> dict:
    """Anchor block: the freeze, the current dataset, and post-freeze commits.

    Every hash here is recomputed from git / disk and checked against the
    constants in humanebench/provenance.py, so the manifest also validates the
    module's own claims rather than just echoing them.
    """
    git_ok = prov.git_available()

    # Self-check: the freeze commit's tree really hashes to FROZEN_PROMPT_HASH.
    freeze_recomputed = None
    if git_ok and prov.git_commit_present(prov.FREEZE_COMMIT):
        text = prov.git_file_at_commit(prov.FREEZE_COMMIT)
        if text is not None:
            freeze_recomputed, _ = prov.prompt_hash_from_jsonl(text)

    # Current on-disk dataset.
    dataset_text = prov.DATASET_PATH.read_text()
    cur_prompt_hash, cur_n = prov.prompt_hash_from_jsonl(dataset_text)
    cur_file_sha = prov.file_sha256(prov.DATASET_PATH)

    # Post-freeze dataset commits must each preserve the prompt hash.
    post_freeze = []
    for sha, label in prov.POST_FREEZE_DATASET_COMMITS:
        entry = {"commit": sha, "label": label, "prompt_hash": None,
                 "preserves_frozen_prompt_hash": None}
        if git_ok and prov.git_commit_present(sha):
            text = prov.git_file_at_commit(sha)
            if text is not None:
                ph, _ = prov.prompt_hash_from_jsonl(text)
                entry["prompt_hash"] = ph
                entry["preserves_frozen_prompt_hash"] = (ph == prov.FROZEN_PROMPT_HASH)
        post_freeze.append(entry)

    return {
        "freeze_commit": prov.FREEZE_COMMIT,
        "freeze_commit_date": prov.FREEZE_COMMIT_DATE,
        "frozen_prompt_hash": prov.FROZEN_PROMPT_HASH,
        "frozen_prompt_hash_recomputed_at_freeze_commit": freeze_recomputed,
        "frozen_prompt_hash_selfcheck_ok": (freeze_recomputed == prov.FROZEN_PROMPT_HASH)
        if freeze_recomputed is not None else None,
        "prompt_hash_recipe": "sha256 over sorted (id, input, target) triples, "
                              "joined id \\x1f input \\x1f target \\x1e, metadata excluded",
        "dataset_current": {
            "path": prov.DATASET_REPO_REL,
            "n_prompts": cur_n,
            "prompt_hash": cur_prompt_hash,
            "prompt_hash_matches_frozen": cur_prompt_hash == prov.FROZEN_PROMPT_HASH,
            "file_sha256": cur_file_sha,
        },
        "post_freeze_dataset_commits": post_freeze,
        "judge_ensemble": prov.JUDGE_ENSEMBLE,
        "git_history_available": git_ok,
    }


def build_run_entry(persona: str, model: str, path: Path) -> dict:
    header = prov.read_eval_header(path)
    ev = header.get("eval", {})
    stats = header.get("stats", {})
    rev = ev.get("revision") or {}
    rev_commit = rev.get("commit")

    prompt_hash, n_samples = prov.eval_prompt_hash(path)
    created = ev.get("created")

    return {
        "persona": persona,
        "model_dir": model,
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "file_sha256": prov.file_sha256(path),
        "eval_model": ev.get("model"),
        "eval_created": created,
        "started_at": stats.get("started_at"),
        "completed_at": stats.get("completed_at"),
        "inspect_ai_version": (ev.get("packages") or {}).get("inspect_ai"),
        "revision_commit": rev_commit,
        # The run-time commit is often a local/unpushed working commit; we record
        # whether it resolves in the canonical repo and never depend on it.
        "revision_resolved_in_repo": prov.git_commit_present(rev_commit)
        if (rev_commit and prov.git_available()) else False,
        "dataset_location": (ev.get("dataset") or {}).get("location"),
        "dataset_samples": (ev.get("dataset") or {}).get("samples"),
        "n_samples_hashed": n_samples,
        "prompt_hash": prompt_hash,
        "prompt_hash_matches_frozen": prompt_hash == prov.FROZEN_PROMPT_HASH,
        "created_after_freeze": prov.created_after_freeze(created) if created else None,
    }


def build_manifest(logs_dir: Path) -> dict:
    runs = []
    for persona, model, path in prov.reported_runs(logs_dir):
        print(f"  hashing {persona}/{model} ...", flush=True)
        runs.append(build_run_entry(persona, model, path))

    anchors = build_anchors()

    matches = sum(1 for r in runs if r["prompt_hash_matches_frozen"])
    after = sum(1 for r in runs if r["created_after_freeze"])
    post_ok = all(c["preserves_frozen_prompt_hash"] in (True, None)
                  for c in anchors["post_freeze_dataset_commits"])

    all_pass = (
        len(runs) > 0
        and matches == len(runs)
        and after == len(runs)
        and anchors["dataset_current"]["prompt_hash_matches_frozen"]
        and post_ok
        and anchors.get("frozen_prompt_hash_selfcheck_ok") in (True, None)
    )

    return {
        "schema": "humanebench-provenance/1",
        "description": "Content-binding provenance: each reported eval run scored "
                       "prompts byte-identical to the frozen HumaneBench dataset.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zenodo_doi": None,  # filled in after the Zenodo deposit
        "anchors": anchors,
        "reported_runs": runs,
        "summary": {
            "n_reported_runs": len(runs),
            "runs_prompt_hash_matches_frozen": matches,
            "runs_created_after_freeze": after,
            "post_freeze_commits_preserve_prompts": post_ok,
            "all_pass": all_pass,
        },
    }


def write_markdown(manifest: dict, path: Path) -> None:
    a = manifest["anchors"]
    s = manifest["summary"]
    lines = [
        "# HumaneBench provenance manifest",
        "",
        "Auto-generated by `scripts/build_provenance.py`. Do not edit by hand; "
        "verify with `python scripts/verify_provenance.py`.",
        "",
        "## Anchors",
        "",
        f"- **Prompt-content freeze commit:** `{a['freeze_commit']}` "
        f"({a['freeze_commit_date']})",
        f"- **Frozen prompt hash:** `{a['frozen_prompt_hash']}`",
        f"  (recipe: {a['prompt_hash_recipe']})",
        f"- **Current dataset** (`{a['dataset_current']['path']}`): "
        f"{a['dataset_current']['n_prompts']} prompts, "
        f"prompt-hash matches frozen = **{a['dataset_current']['prompt_hash_matches_frozen']}**, "
        f"file sha256 `{a['dataset_current']['file_sha256']}`",
        f"- **Zenodo DOI:** {manifest['zenodo_doi'] or '_(pending deposit)_'}",
        "",
        "### Post-freeze dataset commits (must preserve the prompt hash)",
        "",
        "| commit | preserves prompts | label |",
        "| --- | --- | --- |",
    ]
    for c in a["post_freeze_dataset_commits"]:
        lines.append(
            f"| `{c['commit'][:12]}` | {c['preserves_frozen_prompt_hash']} | {c['label']} |"
        )
    lines += [
        "",
        "## Reported runs",
        "",
        f"**{s['n_reported_runs']}** runs · "
        f"prompt-hash match: **{s['runs_prompt_hash_matches_frozen']}/{s['n_reported_runs']}** · "
        f"created-after-freeze: **{s['runs_created_after_freeze']}/{s['n_reported_runs']}** · "
        f"all pass: **{s['all_pass']}**",
        "",
        "| persona | model | created | prompts | match | after freeze | file sha256 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in manifest["reported_runs"]:
        lines.append(
            f"| {r['persona']} | {r['model_dir']} | {r['eval_created']} | "
            f"{r['n_samples_hashed']} | {r['prompt_hash_matches_frozen']} | "
            f"{r['created_after_freeze']} | `{r['file_sha256'][:16]}…` |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", type=Path, default=prov.LOGS_DIR,
                    help="Directory holding {baseline,good_persona,bad_persona}/ "
                         "(default: repo logs/). Point at an extracted Zenodo tarball "
                         "to build from the published artifact.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    runs = prov.reported_runs(args.logs_dir)
    if not runs:
        print(f"ERROR: no reported .eval runs found under {args.logs_dir}", file=sys.stderr)
        return 2
    print(f"Building manifest from {len(runs)} runs under {args.logs_dir} ...")

    manifest = build_manifest(args.logs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_markdown(manifest, args.out_dir / "MANIFEST.md")

    s = manifest["summary"]
    print(f"\nWrote {args.out_dir / 'MANIFEST.json'} and MANIFEST.md")
    print(f"  reported runs                : {s['n_reported_runs']}")
    print(f"  prompt-hash matches frozen   : {s['runs_prompt_hash_matches_frozen']}/{s['n_reported_runs']}")
    print(f"  created after freeze         : {s['runs_created_after_freeze']}/{s['n_reported_runs']}")
    print(f"  post-freeze commits preserve : {s['post_freeze_commits_preserve_prompts']}")
    print(f"  ALL PASS                     : {s['all_pass']}")
    return 0 if s["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
