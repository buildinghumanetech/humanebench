"""Select a stratified subset of curated partner turns for a judge-comparison run.

Strata (first match wins, in priority order; tagged as curation.subset_stratum):
  worst      — any principle judged -1.0 by the original judge (ALL included)
  sentinel   — synthetic/QA-test rows (ALL included; known-mundane consistency probes)
  repeated   — assistant_response occurring >= --repeat-response-min times
               (ALL occurrences included; direct same-input consistency test)
  negative   — some principle <= -0.5 but none -1.0 (random sample)
  trivial    — trivial-tagged rows not otherwise selected (random sample)
  positive   — no negative principle judgments (random sample)

A repeat slice re-judges selected turns a second time (sample_id suffixed
"__rep2") to measure panel self-consistency. Selection is deterministic for a
given --seed.

Usage:
    python scripts/select_comparison_subset.py \
        --input <curated.jsonl> --output <subset.jsonl> [--seed 42] \
        [--negative-sample 60] [--positive-sample 40] [--trivial-sample 30] \
        [--repeat-slice 50] [--repeat-response-min 10]
"""
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


def worst_severity(row: dict) -> float:
    return min(p["severity"] for p in row["principles"].values())


def select(rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], Counter]:
    rng = random.Random(args.seed)
    selected: dict[str, dict] = {}
    stats: Counter = Counter()

    response_counts = Counter(r["assistant_response"] for r in rows)

    def take(row: dict, stratum: str) -> None:
        if row["sample_id"] not in selected:
            row = copy.deepcopy(row)
            row.setdefault("curation", {})["subset_stratum"] = stratum
            selected[row["sample_id"]] = row
            stats[stratum] += 1

    for row in rows:
        if worst_severity(row) <= -1.0:
            take(row, "worst")
    for row in rows:
        if (row.get("curation") or {}).get("synthetic_test"):
            take(row, "sentinel")
    for row in rows:
        if response_counts[row["assistant_response"]] >= args.repeat_response_min:
            take(row, "repeated")

    def sample_from(pool: list[dict], n: int, stratum: str) -> None:
        pool = [r for r in pool if r["sample_id"] not in selected]
        for row in rng.sample(pool, min(n, len(pool))):
            take(row, stratum)

    sample_from(
        [r for r in rows if -1.0 < worst_severity(r) <= -0.5],
        args.negative_sample,
        "negative",
    )
    sample_from(
        [r for r in rows if (r.get("curation") or {}).get("trivial")],
        args.trivial_sample,
        "trivial",
    )
    sample_from(
        [r for r in rows if worst_severity(r) > -0.5],
        args.positive_sample,
        "positive",
    )

    out = list(selected.values())

    repeats = rng.sample(out, min(args.repeat_slice, len(out)))
    for row in repeats:
        rep = copy.deepcopy(row)
        rep["curation"]["repeat_of"] = rep["sample_id"]
        rep["curation"]["subset_stratum"] = "repeat"
        rep["sample_id"] = rep["sample_id"] + "__rep2"
        out.append(rep)
        stats["repeat"] += 1

    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-sample", type=int, default=60)
    parser.add_argument("--positive-sample", type=int, default=40)
    parser.add_argument("--trivial-sample", type=int, default=30)
    parser.add_argument("--repeat-slice", type=int, default=50)
    parser.add_argument("--repeat-response-min", type=int, default=10)
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f]

    out, stats = select(rows, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_principles = 8
    n_judges = 3
    print(f"Selected {len(out)} rows (incl. repeats) from {len(rows)}:")
    for stratum, count in sorted(stats.items()):
        print(f"  {stratum:10s} {count}")
    print(
        f"Fan-out estimate: {len(out)} turns x {n_principles} principles = "
        f"{len(out) * n_principles} samples; x {n_judges} judges = "
        f"{len(out) * n_principles * n_judges} judge calls"
    )


if __name__ == "__main__":
    main()
