#!/usr/bin/env python3
"""Draw 24 new scenarios per principle for the discriminant expansion.

The existing 96 (12/principle) were drawn from the frozen 200. The expansion
draws from the full 800, excluding the original 96 and pds-084 (0-char archived
gemini response — strict solver raises on falsy output). The pooled 36/principle
target is apportioned by ``largest_remainder`` over the full-800 per-principle
domain distribution, and the 24 new slots are the difference between that target
and the existing 12.

This script does NOT reuse ``stratified_subsample`` from
``build_decomposition_subsample.py`` because that function's draws do not nest
at different ``per_principle`` sizes — verified: 24/principle from the 200
contains only 95 of the existing 96. Instead it imports ``largest_remainder``
and applies the apportionment itself.

Outputs (to data/decomposition/, refusing overwrite without --force):
  - discriminant_expansion_192_ids.txt
  - humane_bench_discriminant_expansion_192.jsonl
  - discriminant_expansion_192_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import BOOTSTRAP_SEED, PRINCIPLES  # noqa: E402
from humanebench.excluded import load_excluded_ids  # noqa: E402
from humanebench.provenance import DATASET_PATH, canonical_prompt_hash, file_sha256  # noqa: E402
from scripts.build_decomposition_subsample import largest_remainder, load_rows  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "decomposition"
ORIGINAL_IDS_PATH = OUT_DIR / "discriminant_96_ids.txt"
ORIGINAL_SUMMARY_PATH = OUT_DIR / "discriminant_96_summary.json"

EMPTY_RESPONSE_IDS = {"protect-dignity-and-safety-084"}

POOLED_TARGET = 36
NEW_PER_PRINCIPLE = 24


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing expansion frame")
    args = ap.parse_args()

    ids_path = OUT_DIR / "discriminant_expansion_192_ids.txt"
    jsonl_path = OUT_DIR / "humane_bench_discriminant_expansion_192.jsonl"
    summary_path = OUT_DIR / "discriminant_expansion_192_summary.json"

    existing = [p for p in (ids_path, jsonl_path, summary_path) if p.exists()]
    if existing and not args.force:
        print("Expansion frame already exists; refusing to redraw:")
        for p in existing:
            print(f"  {p.relative_to(REPO_ROOT)}")
        print("Pass --force only if you intend to invalidate all expansion results.")
        return 1

    original_ids = {ln.strip()
                    for ln in ORIGINAL_IDS_PATH.read_text().splitlines() if ln.strip()}
    if len(original_ids) != 96:
        raise SystemExit(f"expected 96 original ids, got {len(original_ids)}")

    original_summary = json.loads(ORIGINAL_SUMMARY_PATH.read_text())
    original_composition = original_summary["composition"]

    rows = load_rows(DATASET_PATH)
    excluded = load_excluded_ids(DATASET_PATH)
    all_excluded = excluded | original_ids | EMPTY_RESPONSE_IDS

    by_principle: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    full_domain_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _raw, row in rows:
        principle = row["target"]
        domain = (row.get("metadata") or {}).get("domain") or "__untagged__"
        full_domain_counts[principle][domain] += 1
        if row["id"] not in all_excluded:
            by_principle[principle][domain].append(row["id"])

    rng = np.random.default_rng(args.seed)
    picked: list[str] = []
    composition: dict[str, dict[str, int]] = {}
    pooled_composition: dict[str, dict[str, int]] = {}

    for principle in PRINCIPLES:
        target36 = largest_remainder(dict(full_domain_counts[principle]), POOLED_TARGET)

        existing_counts: dict[str, int] = {}
        for d, n in original_composition[principle].items():
            existing_counts[d] = n

        alloc: dict[str, int] = {}
        for d in target36:
            alloc[d] = max(0, target36[d] - existing_counts.get(d, 0))

        total_alloc = sum(alloc.values())
        while total_alloc > NEW_PER_PRINCIPLE:
            over_rep = {}
            for d in alloc:
                if alloc[d] == 0:
                    continue
                actual = alloc[d] + existing_counts.get(d, 0)
                ideal = POOLED_TARGET * full_domain_counts[principle].get(d, 0) / 100
                over_rep[d] = actual - ideal
            worst = max((d for d in over_rep if alloc[d] > 0),
                        key=lambda d: (over_rep[d], d))
            alloc[worst] -= 1
            total_alloc -= 1

        while total_alloc < NEW_PER_PRINCIPLE:
            available = by_principle[principle]
            under_rep = {}
            for d in available:
                actual = alloc.get(d, 0) + existing_counts.get(d, 0)
                ideal = POOLED_TARGET * full_domain_counts[principle].get(d, 0) / 100
                headroom = len(available.get(d, [])) - alloc.get(d, 0)
                if headroom > 0:
                    under_rep[d] = ideal - actual
            best = max(under_rep, key=lambda d: (under_rep[d], -alloc.get(d, 0), d))
            alloc[best] = alloc.get(best, 0) + 1
            total_alloc += 1

        for d in sorted(alloc):
            avail = len(by_principle[principle].get(d, []))
            if alloc[d] > avail:
                overflow = alloc[d] - avail
                alloc[d] = avail
                headroom_domains = sorted(
                    (dd for dd in by_principle[principle]
                     if len(by_principle[principle][dd]) > alloc.get(dd, 0)),
                    key=lambda dd: (-(len(by_principle[principle][dd]) - alloc.get(dd, 0)), dd),
                )
                for dd in headroom_domains:
                    if overflow <= 0:
                        break
                    space = len(by_principle[principle][dd]) - alloc.get(dd, 0)
                    add = min(overflow, space)
                    alloc[dd] = alloc.get(dd, 0) + add
                    overflow -= add

        per_domain: dict[str, int] = {}
        for domain in sorted(by_principle[principle]):
            k = alloc.get(domain, 0)
            if k == 0:
                continue
            candidates = sorted(by_principle[principle][domain])
            if k > len(candidates):
                raise SystemExit(
                    f"{principle}/{domain}: need {k} but only {len(candidates)} eligible"
                )
            chosen_idx = rng.choice(len(candidates), size=k, replace=False)
            picked.extend(candidates[i] for i in sorted(chosen_idx))
            per_domain[domain] = k

        composition[principle] = per_domain
        pooled = {}
        for d in set(list(per_domain.keys()) + list(existing_counts.keys())):
            pooled[d] = per_domain.get(d, 0) + existing_counts.get(d, 0)
        pooled_composition[principle] = pooled

    if len(picked) != NEW_PER_PRINCIPLE * len(PRINCIPLES):
        raise SystemExit(f"expected {NEW_PER_PRINCIPLE * len(PRINCIPLES)}, drew {len(picked)}")
    if len(set(picked)) != len(picked):
        raise SystemExit("duplicate ids in draw")
    if set(picked) & original_ids:
        raise SystemExit("expansion overlaps with original 96")
    if set(picked) & EMPTY_RESPONSE_IDS:
        raise SystemExit("expansion contains empty-response ids")

    picked = sorted(picked)
    id_set = set(picked)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(picked) + "\n")

    parsed_by_id = {row["id"]: (raw, row) for raw, row in rows}
    jsonl_path.write_text(
        "".join(f"{parsed_by_id[sid][0]}\n" for sid in picked
                if sid in parsed_by_id)
    )

    subset_hash, n_hashed = canonical_prompt_hash(
        (sid, parsed_by_id[sid][1]["input"], parsed_by_id[sid][1].get("target"))
        for sid in picked
    )

    summary = {
        "schema": "humanebench-decomposition-subsample/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "per_principle": NEW_PER_PRINCIPLE,
        "n_scenarios": len(picked),
        "stratification": (
            f"{NEW_PER_PRINCIPLE} per principle; domain apportioned to hit "
            f"pooled-{POOLED_TARGET} target via largest-remainder over full-800 "
            f"domain counts, no VP strata"
        ),
        "vp_strata": False,
        "source_dataset": "data/humane_bench.jsonl",
        "source_dataset_sha256": file_sha256(DATASET_PATH),
        "n_excluded_in_source": len(excluded),
        "extra_excluded_ids": sorted(EMPTY_RESPONSE_IDS),
        "subset_prompt_hash": subset_hash,
        "n_prompts_hashed": n_hashed,
        "ids_file": str(ids_path.relative_to(REPO_ROOT)),
        "parent_ids_file": None,
        "parent_ids_sha256": None,
        "sibling_ids_file": str(ORIGINAL_IDS_PATH.relative_to(REPO_ROOT)),
        "sibling_ids_sha256": file_sha256(ORIGINAL_IDS_PATH),
        "nested_in_parent": False,
        "ids_file_sha256": file_sha256(ids_path),
        "jsonl_file": str(jsonl_path.relative_to(REPO_ROOT)),
        "jsonl_file_sha256": file_sha256(jsonl_path),
        "pooled_target_per_principle": POOLED_TARGET,
        "composition": composition,
        "pooled_composition": pooled_composition,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {len(picked)} ids -> {ids_path.relative_to(REPO_ROOT)}")
    print(f"wrote dataset    -> {jsonl_path.relative_to(REPO_ROOT)}")
    print(f"wrote summary    -> {summary_path.relative_to(REPO_ROOT)}")
    print(f"\ncomposition (principle -> domain -> n):")
    for principle in PRINCIPLES:
        parts = ", ".join(f"{d}={n}" for d, n in sorted(composition[principle].items()))
        print(f"  {principle:35s} {parts}")
    print(f"\npooled composition (existing 12 + new 24 = 36):")
    for principle in PRINCIPLES:
        parts = ", ".join(f"{d}={n}" for d, n in sorted(pooled_composition[principle].items()))
        print(f"  {principle:35s} {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
