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
  positive_extreme — any principle judged +1.0 AND no negative cells (random
                     sample). Regression-to-the-mean control: an equally noisy
                     second judge regresses BOTH extreme tails toward the
                     middle; a genuinely better judge corrects asymmetrically
                     (concentrated on the over-escalated negative tail).
  negative         — worst cell in (-1.0, -0.5] (random sample)
  trivial          — trivial-tagged rows not otherwise selected (random sample)
  positive         — no negative cells and no +1.0 cell (random sample)

The four severity classes (worst / negative / positive_extreme / positive)
partition the severity space, so a turn belongs to exactly one — mixed turns
(a +1.0 alongside a negative cell) are negative-class: their negative tail is
what the comparison targets. qa_test / repeated / trivial cross-cut the
severity classes.

Every selected row carries boolean membership flags (curation.flags) that use
EXACTLY the stratum predicates above — flags, not first-match priority labels,
give correct analysis denominators. The manifest JSON records, per stratum:
  eligible          — rows matching the predicate in the whole population
  residual_pool     — eligible rows still available after higher-priority strata
  labeled           — rows carrying this stratum's priority label
  sampling_fraction — labeled/residual_pool (1.0 for all-included strata)
Per-row inclusion probability = 1.0 if the row matches any all-included
stratum, else its class's sampling_fraction. Report per-stratum by default;
pooled estimates must reweight by these fractions.

Each sampled stratum draws from its own seeded RNG stream (seed + stratum
name), so adding or resizing one stratum never changes another's draws.

A repeat slice re-judges selected turns a second time (sample_id suffixed
"__rep2") to measure panel self-consistency on identical inputs WITHIN one
eval run. Because within-run repeats share the provider load regime, they give
a lower bound on nondeterminism; a companion file (<output>_between_run.jsonl,
"__rep3" ids, written only when non-empty) holds the same turns for a
separate-day invocation to measure between-run reliability — convert it with
convert_partner_results.py and judge it exactly like the main subset.

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

FLAG_NAMES = [
    "is_worst",
    "is_negative",
    "is_positive_extreme",
    "is_positive",
    "is_qa_test",
    "is_trivial",
    "is_repeated_reply",
]


def severities(row: dict) -> list[float]:
    return [p["severity"] for p in row["principles"].values()]


def membership_flags(row: dict, repeated_reply: bool) -> dict:
    """Stratum-eligibility flags. The four severity flags partition the
    severity space (exactly one is true per row); the rest cross-cut."""
    sev = severities(row)
    mn, mx = min(sev), max(sev)
    cur = row.get("curation") or {}
    return {
        "is_worst": mn <= -1.0,
        "is_negative": -1.0 < mn <= -0.5,
        "is_positive_extreme": mn > -0.5 and mx >= 1.0,
        "is_positive": mn > -0.5 and mx < 1.0,
        "is_qa_test": bool(cur.get("synthetic_test")),
        "is_trivial": bool(cur.get("trivial")),
        "is_repeated_reply": repeated_reply,
    }


def flag_totals(flag_dicts: list[dict]) -> dict[str, int]:
    """Zero-safe totals: every flag name appears, even when its count is 0."""
    return {
        name: sum(1 for f in flag_dicts if f.get(name))
        for name in FLAG_NAMES
    }


# (stratum name, eligibility flag, all-included?) in priority order.
STRATA = [
    ("worst", "is_worst", True),
    ("qa_test", "is_qa_test", True),
    ("repeated", "is_repeated_reply", True),
    ("positive_extreme", "is_positive_extreme", False),
    ("negative", "is_negative", False),
    ("trivial", "is_trivial", False),
    ("positive", "is_positive", False),
]

SAMPLE_SIZE_ARG = {
    "positive_extreme": "positive_extreme_sample",
    "negative": "negative_sample",
    "trivial": "trivial_sample",
    "positive": "positive_sample",
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

    response_counts = Counter(r["assistant_response"] for r in rows)
    flags_by_id = {
        r["sample_id"]: membership_flags(
            r, response_counts[r["assistant_response"]] >= args.repeat_response_min
        )
        for r in rows
    }

    selected: dict[str, dict] = {}
    manifest: dict = {"seed": args.seed, "population": len(rows), "strata": {}}

    def take(row: dict, stratum: str) -> None:
        if row["sample_id"] not in selected:
            row = copy.deepcopy(row)
            cur = row.setdefault("curation", {})
            cur["subset_stratum"] = stratum
            cur["flags"] = flags_by_id[row["sample_id"]]
            selected[row["sample_id"]] = row
            stats[stratum] += 1

    for stratum, flag, take_all in STRATA:
        eligible = [r for r in rows if flags_by_id[r["sample_id"]][flag]]
        residual = [r for r in eligible if r["sample_id"] not in selected]
        if take_all:
            for row in residual:
                take(row, stratum)
        else:
            # Independent stream per stratum: adding or resizing one stratum
            # never perturbs another's draws for the same base seed.
            rng = random.Random(f"{args.seed}:{stratum}")
            n = getattr(args, SAMPLE_SIZE_ARG[stratum])
            for row in rng.sample(residual, min(n, len(residual))):
                take(row, stratum)
        manifest["strata"][stratum] = {
            "eligible": len(eligible),
            "residual_pool": len(residual),
            "labeled": stats[stratum],
            "sampling_fraction": (
                1.0 if take_all
                else (stats[stratum] / len(residual)) if residual else 0.0
            ),
        }

    out = list(selected.values())

    manifest["population_flags"] = flag_totals(list(flags_by_id.values()))
    manifest["selected_flags"] = flag_totals([r["curation"]["flags"] for r in out])

    rng_repeat = random.Random(f"{args.seed}:repeat")
    repeats = rng_repeat.sample(out, min(args.repeat_slice, len(out)))
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

    n_principles = 8
    n_judges = 3
    print(f"Selected {len(out)} rows (incl. repeats) from {len(rows)}:")
    for stratum, count in sorted(stats.items()):
        print(f"  {stratum:24s} {count}")
    print(f"Manifest (eligible/residual/fractions/flags): {manifest_path}")

    if extras["between_run"]:
        between_path = args.output.with_name(args.output.stem + "_between_run.jsonl")
        with open(between_path, "w") as f:
            for row in extras["between_run"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"Between-run repeat file (convert + judge in a separate invocation "
            f"on a DIFFERENT day): {between_path}"
        )
    else:
        print("No repeat slice requested — no between-run file written.")

    print(
        f"Fan-out estimate: {len(out)} turns x {n_principles} principles = "
        f"{len(out) * n_principles} samples; x {n_judges} judges = "
        f"{len(out) * n_principles * n_judges} judge calls"
    )


if __name__ == "__main__":
    main()
