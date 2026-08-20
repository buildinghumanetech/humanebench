"""Select a stratified subset of curated partner turns for a judge-comparison run.

Strata (first match wins, in priority order; tagged as curation.subset_stratum):
  worst            — any principle judged -1.0 by the original judge (ALL included)
  qa_test          — synthetic/QA-test rows (ALL included). Their replies VARY,
                     so they probe the panel's false-alarm floor on known-mundane
                     traffic — they are NOT consistency probes.
  repeated         — assistant_response occurring >= --repeat-response-min times
                     (ALL occurrences included). Keyed on the response text
                     alone — the audit's identical-reply inconsistency exhibit —
                     so the *prompts* may differ across occurrences.
  positive_extreme — any principle judged +1.0, none -1.0 (random sample).
                     Regression-to-the-mean control: an equally noisy second
                     judge regresses BOTH extreme tails toward the middle;
                     a genuinely better judge corrects asymmetrically
                     (concentrated on the over-escalated negative tail).
  negative         — some principle <= -0.5 but none -1.0 (random sample)
  trivial          — trivial-tagged rows not otherwise selected (random sample)
  positive         — no negative principle judgments (random sample)

Because first-match-wins priority labels give wrong denominators for analysis
(e.g. most QA rows also sit in the worst stratum), every selected row also
carries boolean membership flags (curation.flags: is_worst, is_qa_test,
is_repeated_reply, is_positive_extreme, is_negative, is_trivial), and a
manifest JSON records per-stratum pool sizes and sampling fractions — required
for any reweighted/pooled estimate. Report per-stratum by default.

A repeat slice re-judges selected turns a second time (sample_id suffixed
"__rep2") to measure panel self-consistency on identical inputs WITHIN one
eval run. Because within-run repeats share the provider load regime, they give
a lower bound on nondeterminism; a companion file (<output>_between_run.jsonl,
"__rep3" ids) holds the same turns for a separate-day invocation to measure
between-run reliability. Selection is deterministic for a given --seed.

The input must be the output of curate_production_pairs.py (curation tags are
required). Rows with no per-principle judgments are excluded (reported).

Usage:
    python scripts/select_comparison_subset.py \
        --input <curated.jsonl> --output <subset.jsonl> [--seed 42] \
        [--negative-sample 60] [--positive-sample 40] [--trivial-sample 30] \
        [--positive-extreme-sample 60] [--repeat-slice 50] \
        [--repeat-response-min 10]
"""
import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


def severities(row: dict) -> list[float]:
    return [p["severity"] for p in row["principles"].values()]


def worst_severity(row: dict) -> float:
    return min(severities(row))


def membership_flags(row: dict, repeated_reply: bool) -> dict:
    sev = severities(row)
    cur = row.get("curation") or {}
    return {
        "is_worst": min(sev) <= -1.0,
        "is_negative": min(sev) <= -0.5,
        "is_positive_extreme": max(sev) >= 1.0,
        "is_qa_test": bool(cur.get("synthetic_test")),
        "is_trivial": bool(cur.get("trivial")),
        "is_repeated_reply": repeated_reply,
    }


def select(
    rows: list[dict], args: argparse.Namespace
) -> tuple[list[dict], Counter, dict]:
    if not any("curation" in r for r in rows):
        raise SystemExit(
            "input has no curation tags — run curate_production_pairs.py "
            "first (the qa_test and trivial strata depend on its tags)"
        )

    stats: Counter = Counter()
    n_before = len(rows)
    rows = [r for r in rows if r.get("principles")]
    stats["excluded_no_judgments"] = n_before - len(rows)

    rng = random.Random(args.seed)
    selected: dict[str, dict] = {}
    manifest: dict = {"seed": args.seed, "population": len(rows), "strata": {}}

    response_counts = Counter(r["assistant_response"] for r in rows)

    def is_repeated(row: dict) -> bool:
        return response_counts[row["assistant_response"]] >= args.repeat_response_min

    def take(row: dict, stratum: str) -> None:
        if row["sample_id"] not in selected:
            row = copy.deepcopy(row)
            cur = row.setdefault("curation", {})
            cur["subset_stratum"] = stratum
            cur["flags"] = membership_flags(row, is_repeated(row))
            selected[row["sample_id"]] = row
            stats[stratum] += 1

    def take_all(predicate, stratum: str) -> None:
        pool = [r for r in rows if predicate(r)]
        for row in pool:
            take(row, stratum)
        # All pool members are in the subset; some may carry a higher-priority
        # stratum label — flags, not labels, give the true denominators.
        manifest["strata"][stratum] = {
            "pool": len(pool),
            "selected": len(pool),
            "labeled": stats[stratum],
            "sampling_fraction": 1.0,
        }

    def sample_from(predicate, n: int, stratum: str) -> None:
        pool = [r for r in rows if predicate(r) and r["sample_id"] not in selected]
        for row in rng.sample(pool, min(n, len(pool))):
            take(row, stratum)
        manifest["strata"][stratum] = {
            "pool": len(pool),
            "selected": stats[stratum],
            "sampling_fraction": (stats[stratum] / len(pool)) if pool else 0.0,
        }

    take_all(lambda r: worst_severity(r) <= -1.0, "worst")
    take_all(lambda r: (r.get("curation") or {}).get("synthetic_test"), "qa_test")
    take_all(is_repeated, "repeated")
    sample_from(
        lambda r: max(severities(r)) >= 1.0 and worst_severity(r) > -1.0,
        args.positive_extreme_sample,
        "positive_extreme",
    )
    sample_from(
        lambda r: -1.0 < worst_severity(r) <= -0.5, args.negative_sample, "negative"
    )
    sample_from(
        lambda r: (r.get("curation") or {}).get("trivial"),
        args.trivial_sample,
        "trivial",
    )
    sample_from(lambda r: worst_severity(r) > -0.5, args.positive_sample, "positive")

    out = list(selected.values())

    # Population-level flag totals (denominators for reweighted estimates).
    manifest["population_flags"] = dict(
        sum(
            (Counter({k: int(v) for k, v in membership_flags(r, is_repeated(r)).items()})
             for r in rows),
            Counter(),
        )
    )
    manifest["selected_flags"] = dict(
        sum(
            (Counter({k: int(v) for k, v in r["curation"]["flags"].items()})
             for r in out),
            Counter(),
        )
    )

    repeats = rng.sample(out, min(args.repeat_slice, len(out)))
    between_run: list[dict] = []
    for row in repeats:
        rep = copy.deepcopy(row)
        rep["curation"]["repeat_of"] = rep["sample_id"]
        rep["curation"]["subset_stratum"] = "repeat"
        rep["sample_id"] = rep["sample_id"] + "__rep2"
        out.append(rep)
        stats["repeat"] += 1

        between = copy.deepcopy(row)
        between["curation"]["repeat_of"] = between["sample_id"]
        between["curation"]["subset_stratum"] = "between_run_repeat"
        between["sample_id"] = between["sample_id"] + "__rep3"
        between_run.append(between)

    return out, stats, {"manifest": manifest, "between_run": between_run}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-sample", type=int, default=60)
    parser.add_argument("--positive-sample", type=int, default=40)
    parser.add_argument("--trivial-sample", type=int, default=30)
    parser.add_argument("--positive-extreme-sample", type=int, default=60)
    parser.add_argument("--repeat-slice", type=int, default=50)
    parser.add_argument("--repeat-response-min", type=int, default=10)
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out, stats, extras = select(rows, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(extras["manifest"], indent=2) + "\n")

    between_path = args.output.with_name(args.output.stem + "_between_run.jsonl")
    with open(between_path, "w") as f:
        for row in extras["between_run"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_principles = 8
    n_judges = 3
    print(f"Selected {len(out)} rows (incl. repeats) from {len(rows)}:")
    for stratum, count in sorted(stats.items()):
        print(f"  {stratum:24s} {count}")
    print(f"Manifest (pools/fractions/flags): {manifest_path}")
    print(f"Between-run repeat file (judge on a DIFFERENT day): {between_path}")
    print(
        f"Fan-out estimate: {len(out)} turns x {n_principles} principles = "
        f"{len(out) * n_principles} samples; x {n_judges} judges = "
        f"{len(out) * n_principles * n_judges} judge calls"
    )


if __name__ == "__main__":
    main()
