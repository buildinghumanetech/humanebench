#!/usr/bin/env python3
"""Age gradient analysis: does children < teenagers < elderly hold per principle?

For each principle covered by at least two of the three age-group VPs,
show baseline mean and steerability delta side by side. Flag which
principles follow the gradient pattern and which break it.
"""

# Paper: utility -- no paper-facing output; prints a per-principle check of whether the
#   children / teenagers / elderly ordering holds in both baseline level and steerability
#   delta, over tables/vp_sample_scores.csv. Supports the exploratory supplement section,
#   "Vulnerable-Population Analysis"; the figures/vp_age_gradient.* chart of that name is
#   produced by scripts/vp_tables_and_figure.py, not here.

import pandas as pd
from pathlib import Path

TABLES = Path(__file__).resolve().parent.parent / "tables"

df = pd.read_csv(TABLES / "vp_sample_scores.csv")

AGE_GROUPS = ["children", "teenagers", "elderly"]
AGE_ORDER = {name: i for i, name in enumerate(AGE_GROUPS)}

age = df[df["vulnerable_population"].isin(AGE_GROUPS)].copy()

if __name__ == "__main__":
    # Per age-group x principle: baseline mean, bad_persona mean, delta, good_persona mean
    rows = []
    for (vp_name, principle), group in age.groupby(["vulnerable_population", "principle"]):
        bl = group[group["persona"] == "baseline"]["score"]
        bp = group[group["persona"] == "bad_persona"]["score"]
        gp = group[group["persona"] == "good_persona"]["score"]
        rows.append({
            "age_group": vp_name,
            "principle": principle,
            "baseline": bl.mean(),
            "bad_persona": bp.mean(),
            "delta": bp.mean() - bl.mean(),
            "good_persona": gp.mean(),
            "n_scenarios": group["sample_id"].nunique(),
        })

    detail = pd.DataFrame(rows)

    # Keep principles with at least 2 age groups
    principle_coverage = detail.groupby("principle")["age_group"].nunique()
    shared_principles = principle_coverage[principle_coverage >= 2].index.tolist()
    detail = detail[detail["principle"].isin(shared_principles)].copy()

    # Sort by age order within each principle
    detail["age_order"] = detail["age_group"].map(AGE_ORDER)
    detail = detail.sort_values(["principle", "age_order"])

    print("=" * 110)
    print("Age gradient by principle: children → teenagers → elderly")
    print("Gradient holds if baseline increases AND delta becomes more negative with age")
    print("=" * 110)

    for principle in sorted(shared_principles):
        subset = detail[detail["principle"] == principle].copy()
        groups_present = subset["age_group"].tolist()

        print(f"\n--- {principle} (coverage: {', '.join(groups_present)}) ---")
        print(subset[["age_group", "baseline", "bad_persona", "delta", "good_persona", "n_scenarios"]].to_string(
            index=False, float_format="%.3f"
        ))

        if len(groups_present) >= 2:
            baselines = subset["baseline"].tolist()
            deltas = subset["delta"].tolist()

            baseline_increasing = all(baselines[i] <= baselines[i + 1] for i in range(len(baselines) - 1))
            delta_decreasing = all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))

            signals = []
            if baseline_increasing:
                signals.append("baseline increases with age")
            else:
                signals.append("baseline does NOT increase with age")
            if delta_decreasing:
                signals.append("steerability increases with age")
            else:
                signals.append("steerability does NOT increase with age")

            holds = "FULL" if baseline_increasing and delta_decreasing else "PARTIAL" if baseline_increasing or delta_decreasing else "NONE"
            print(f"  → Gradient: {holds} ({'; '.join(signals)})")

    # Summary
    print()
    print()
    print("=" * 110)
    print("Summary: gradient status per principle")
    print("=" * 110)
    print()

    for principle in sorted(shared_principles):
        subset = detail[detail["principle"] == principle]
        groups_present = subset["age_group"].tolist()
        baselines = subset["baseline"].tolist()
        deltas = subset["delta"].tolist()

        baseline_increasing = all(baselines[i] <= baselines[i + 1] for i in range(len(baselines) - 1))
        delta_decreasing = all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))
        holds = "FULL" if baseline_increasing and delta_decreasing else "PARTIAL" if baseline_increasing or delta_decreasing else "NONE"

        coverage = f"({len(groups_present)}/3)"
        print(f"  {principle:40s} {coverage}  {holds:8s}  baselines: {[f'{b:.3f}' for b in baselines]}  deltas: {[f'{d:.3f}' for d in deltas]}")
