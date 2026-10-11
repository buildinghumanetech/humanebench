#!/usr/bin/env python3
"""Draw and freeze the 200-scenario subsample used by decomposition conditions C/D/E.

The goal-vs-tactics decomposition runs condition B at full scale but conditions
C/D/E on a subsample, to conserve budget. Every C/D/E-vs-A or C/D/E-vs-baseline
contrast is therefore restricted to *these* scenarios, which only works if all
three conditions score the identical set. So the draw happens exactly once, is
seeded, and is frozen to disk; the analysis reads the frozen ids rather than
re-drawing.

Stratification is 25 per principle (8 x 25 = 200), then by vulnerable-population
bucket, then across that principle's domains in proportion to their size via
largest-remainder rounding. The VP level is on by default because the frozen
subsample used it: without it the children count lands anywhere in 5-13
depending on the seed, and with it the headline VP counts are constant across
seeds. A default of off would mean the documented command no longer reproduces
the frozen draw.
Principle balance is the load-bearing property -- HumaneScore is a mean of the 8
principle means, so an unbalanced draw would reweight the metric itself. Domain
proportionality is secondary but cheap, and it keeps the domain mix of each
principle roughly intact given the known principle/domain confound (Cramer's
V = 0.432).

The scenario rows are copied as *raw lines* from the source dataset rather than
re-serialized, so the (id, input, target) triples are byte-identical to the
frozen set and the provenance check reduces to a subset test.

Usage:
    python scripts/build_decomposition_subsample.py [--force]
"""
# Paper: produces data/decomposition/subsample_200_ids.txt and humane_bench_subsample_200.jsonl -
#        the frozen, seeded 200-scenario draw (25 per principle) that decomposition conditions
#        C, D and E are scored on (main paper, "Engagement Pressure Alone Drives Degradation" /
#        supplement, "Dose-Response Across Adversarial Wordings", whose contrasts are computed
#        on this subsample).
# Paper: implements that draw's stratification - principle, then vulnerable-population bucket,
#        then domain by largest-remainder rounding - seeded so it is drawn once and reproducible.
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanebench.bootstrap import BOOTSTRAP_SEED, PRINCIPLES
from humanebench.excluded import load_excluded_ids
from humanebench.provenance import (
    DATASET_PATH,
    canonical_prompt_hash,
    file_sha256,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "decomposition"
IDS_PATH = OUT_DIR / "subsample_200_ids.txt"
JSONL_PATH = OUT_DIR / "humane_bench_subsample_200.jsonl"
SUMMARY_PATH = OUT_DIR / "subsample_200_summary.json"

PER_PRINCIPLE = 25


def load_rows(dataset_path: Path) -> list[tuple[str, dict]]:
    """Return ``(raw_line, parsed_row)`` for each non-empty dataset line."""
    rows: list[tuple[str, dict]] = []
    with dataset_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append((line.rstrip("\n"), json.loads(line)))
    return rows


def largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    """Apportion ``total`` slots across keys proportionally to ``weights``.

    Deterministic: ties in the remainder are broken by (-remainder, key), so the
    result depends only on the weights, never on dict insertion order.
    """
    pool = sum(weights.values())
    if pool == 0:
        return {k: 0 for k in weights}
    exact = {k: total * w / pool for k, w in weights.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    remaining = total - sum(alloc.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in order[:remaining]:
        alloc[k] += 1
    return alloc


# Vulnerable-population buckets used as a middle stratum. The three named groups
# are the ones the paper reports; everything else tagged is pooled, and untagged
# rows form their own bucket.
#
# Without this level, VP composition is left to luck: a draw stratified only on
# principle and domain reproduced the population's 34% VP *share* but returned
# 5 children-tagged scenarios against 10.2 expected -- the worst of 40 seeds.
# Stratifying on the bucket makes the counts a property of the design rather
# than of the seed, which is the difference between a defensible draw and one
# that has to be explained.
VP_HEADLINE_GROUPS = ("children", "teenagers", "elderly")


def vp_bucket(row: dict) -> str:
    tag = (row.get("metadata") or {}).get("vulnerable-population") or ""
    if tag in VP_HEADLINE_GROUPS:
        return tag
    return "other-vp" if tag else "none"


def stratified_subsample(
    rows: list[tuple[str, dict]],
    excluded: set[str],
    *,
    per_principle: int = PER_PRINCIPLE,
    seed: int = BOOTSTRAP_SEED,
    vp_strata: bool = True,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Return ``(sorted_ids, {principle: {stratum: n}})`` for the frozen draw.

    Principle is always the top stratum -- HumaneScore is the mean of 8
    principle means, so imbalance there reweights the metric itself. With
    ``vp_strata`` the second level is the VP bucket and the third is domain;
    without it, domain is the second level and VP composition is incidental.
    """
    rng = np.random.default_rng(seed)

    def key(row: dict) -> str:
        domain = (row.get("metadata") or {}).get("domain") or "__untagged__"
        return f"{vp_bucket(row)}|{domain}" if vp_strata else domain

    by_principle: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for _raw, row in rows:
        if row["id"] in excluded:
            continue
        principle = row["target"]
        by_principle[principle][key(row)].append(row["id"])

    missing = [p for p in PRINCIPLES if p not in by_principle]
    if missing:
        raise SystemExit(f"dataset is missing principles: {missing}")

    picked: list[str] = []
    composition: dict[str, dict[str, int]] = {}
    for principle in PRINCIPLES:  # canonical order -> deterministic rng consumption
        domains = by_principle[principle]
        available = {d: len(ids) for d, ids in domains.items()}
        n_available = sum(available.values())
        if n_available < per_principle:
            raise SystemExit(
                f"principle {principle} has only {n_available} eligible scenarios, "
                f"need {per_principle}"
            )
        alloc = largest_remainder(available, per_principle)

        # A domain can be allocated more slots than it has rows only if rounding
        # overshoots a tiny domain; spill the excess to the largest domains.
        overflow = 0
        for d in sorted(alloc):
            if alloc[d] > available[d]:
                overflow += alloc[d] - available[d]
                alloc[d] = available[d]
        while overflow > 0:
            headroom = sorted(
                (d for d in alloc if available[d] > alloc[d]),
                key=lambda d: (-(available[d] - alloc[d]), d),
            )
            if not headroom:
                raise SystemExit(f"cannot allocate {per_principle} within {principle}")
            alloc[headroom[0]] += 1
            overflow -= 1

        per_domain: dict[str, int] = {}
        for domain in sorted(domains):  # sorted -> rng draw sequence is stable
            k = alloc[domain]
            if k == 0:
                continue
            candidates = sorted(domains[domain])
            chosen_idx = rng.choice(len(candidates), size=k, replace=False)
            picked.extend(candidates[i] for i in sorted(chosen_idx))
            per_domain[domain] = k
        composition[principle] = per_domain

    if len(picked) != per_principle * len(PRINCIPLES):
        raise SystemExit(f"expected {per_principle * len(PRINCIPLES)} ids, drew {len(picked)}")
    if len(set(picked)) != len(picked):
        raise SystemExit("duplicate ids in draw")
    return sorted(picked), composition


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--per-principle", type=int, default=PER_PRINCIPLE)
    ap.add_argument(
        "--from-ids",
        type=Path,
        default=None,
        help="restrict the draw to the ids in this file, producing a set nested "
             "inside it. Independent draws at different sizes do NOT nest -- "
             "np.random.choice returns a different selection per size, not a "
             "prefix -- so a smaller arm must be drawn FROM the larger one for "
             "any comparison between them to stay paired.",
    )
    ap.add_argument("--no-vp-strata", dest="vp_strata", action="store_false",
                    help="stratify on principle and domain only, leaving VP "
                         "composition to the draw. NOT what the frozen subsample "
                         "used -- it is stratified principle -> VP bucket -> "
                         "domain, and the default must stay on so the documented "
                         "invocation reproduces the frozen draw.")
    ap.set_defaults(vp_strata=True)
    ap.add_argument("--name", default=None,
                    help="output basename stem (default: subsample_<n>)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing frozen subsample (it is meant to be drawn once)",
    )
    args = ap.parse_args()

    n_total = args.per_principle * 8
    stem = args.name or f"subsample_{n_total}"
    ids_path = OUT_DIR / f"{stem}_ids.txt"
    jsonl_path = OUT_DIR / f"humane_bench_{stem}.jsonl"
    summary_path = OUT_DIR / f"{stem}_summary.json"

    existing = [p for p in (ids_path, jsonl_path, summary_path) if p.exists()]
    if existing and not args.force:
        print("Frozen subsample already exists; refusing to redraw:")
        for p in existing:
            print(f"  {p.relative_to(REPO_ROOT)}")
        print("Pass --force only if you intend to invalidate every C/D/E contrast.")
        return 1

    rows = load_rows(DATASET_PATH)
    excluded = load_excluded_ids(DATASET_PATH)

    parent_ids: set[str] | None = None
    if args.from_ids:
        args.from_ids = Path(args.from_ids).resolve()
        parent_ids = {ln.strip() for ln in args.from_ids.read_text().splitlines() if ln.strip()}
        rows = [(raw, row) for raw, row in rows if row["id"] in parent_ids]
        print(f"restricted to parent set {args.from_ids.name}: {len(rows)} rows")
    n_elig=len([r for _x, r in rows if r["id"] not in excluded])
    print(f"pool rows: {len(rows)}   eligible after exclusions: {n_elig}")

    ids, composition = stratified_subsample(
        rows, excluded, per_principle=args.per_principle, seed=args.seed,
        vp_strata=args.vp_strata
    )
    id_set = set(ids)

    raw_by_id = {row["id"]: raw for raw, row in rows}
    parsed_by_id = {row["id"]: row for _raw, row in rows}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(ids) + "\n")
    # Raw lines, in dataset order, byte-identical to the source.
    jsonl_path.write_text(
        "".join(f"{raw}\n" for raw, row in rows if row["id"] in id_set)
    )

    subset_hash, n_hashed = canonical_prompt_hash(
        (i, parsed_by_id[i]["input"], parsed_by_id[i].get("target")) for i in ids
    )

    summary = {
        "schema": "humanebench-decomposition-subsample/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "per_principle": args.per_principle,
        "n_scenarios": len(ids),
        "stratification": (
            f"{args.per_principle} per principle; "
            + ("VP bucket then domain" if args.vp_strata else "domain")
            + " apportioned within each principle by largest-remainder rounding"
        ),
        "vp_strata": bool(args.vp_strata),
        "source_dataset": "data/humane_bench.jsonl",
        "source_dataset_sha256": file_sha256(DATASET_PATH),
        "n_excluded_in_source": len(excluded),
        "subset_prompt_hash": subset_hash,
        "n_prompts_hashed": n_hashed,
        "ids_file": str(ids_path.relative_to(REPO_ROOT)),
        "parent_ids_file": (str(args.from_ids.relative_to(REPO_ROOT))
                            if args.from_ids else None),
        "parent_ids_sha256": (file_sha256(args.from_ids) if args.from_ids else None),
        "nested_in_parent": bool(parent_ids),
        "ids_file_sha256": file_sha256(ids_path),
        "jsonl_file": str(jsonl_path.relative_to(REPO_ROOT)),
        "jsonl_file_sha256": file_sha256(jsonl_path),
        "composition": composition,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nwrote {len(ids)} ids -> {ids_path.relative_to(REPO_ROOT)}")
    print(f"wrote dataset    -> {jsonl_path.relative_to(REPO_ROOT)}")
    print(f"wrote summary    -> {summary_path.relative_to(REPO_ROOT)}")
    print(f"\nDECOMP_SUBSET_PROMPT_HASH = {subset_hash!r}")
    print("\ncomposition (principle -> domain -> n):")
    for principle in PRINCIPLES:
        parts = ", ".join(f"{d}={n}" for d, n in sorted(composition[principle].items()))
        print(f"  {principle:35s} {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
