#!/usr/bin/env python3
"""For each VP, identify the most uniquely problematic principle.

"Uniquely problematic" = the principle where a VP's bad_persona score
is furthest below the all-VP average for that same principle. This
surfaces VP-specific weaknesses rather than principles that are
universally bad.

Reports multi-principle VPs (>1 principle) and single-principle VPs
(mostly Design for Equity & Inclusion) separately.
"""

# Paper: utility -- no paper-facing output; prints, per vulnerable population, the
#   principle whose bad-persona mean falls furthest below the all-VP mean for that same
#   principle, over tables/vp_sample_scores.csv. Supports the exploratory supplement
#   section, "Vulnerable-Population Analysis".

import pandas as pd
from pathlib import Path

TABLES = Path(__file__).resolve().parent.parent / "tables"

df = pd.read_csv(TABLES / "vp_sample_scores.csv")
vp = df[df["vulnerable_population"].notna() & (df["vulnerable_population"] != "")].copy()

if __name__ == "__main__":
    bad = vp[vp["persona"] == "bad_persona"].copy()

    # All-VP average bad_persona score per principle (the reference point)
    all_vp_avg = bad.groupby("principle")["score"].mean()

    # Per VP x principle bad_persona mean
    vp_principle = bad.groupby(["vulnerable_population", "principle"])["score"].agg(
        ["mean", "count"]
    ).reset_index()
    vp_principle.columns = ["vulnerable_population", "principle", "bad_mean", "n_scores"]
    n_scenarios = (
        bad.groupby(["vulnerable_population", "principle"])["sample_id"]
        .nunique()
        .reset_index(name="n_scenarios")
    )
    vp_principle = vp_principle.merge(n_scenarios, on=["vulnerable_population", "principle"])

    # Add the all-VP reference and compute gap
    vp_principle["all_vp_avg"] = vp_principle["principle"].map(all_vp_avg)
    vp_principle["gap"] = vp_principle["bad_mean"] - vp_principle["all_vp_avg"]

    # Count principles per VP
    principle_counts = vp_principle.groupby("vulnerable_population")["principle"].nunique()
    vp_principle["n_principles"] = vp_principle["vulnerable_population"].map(principle_counts)

    # Split multi vs single principle VPs
    multi = vp_principle[vp_principle["n_principles"] > 1].copy()
    single = vp_principle[vp_principle["n_principles"] == 1].copy()

    # --- Multi-principle VPs: full breakdown ---
    print("=" * 110)
    print("All VP x principle cells for multi-principle VPs (>1 principle)")
    print("gap = VP's bad_persona mean minus all-VP average for that principle")
    print("=" * 110)
    print()

    for vp_name in multi.sort_values("n_principles", ascending=False)["vulnerable_population"].unique():
        subset = multi[multi["vulnerable_population"] == vp_name].sort_values("gap")
        print(f"--- {vp_name} ({subset['n_principles'].iloc[0]} principles) ---")
        print(subset[["principle", "bad_mean", "all_vp_avg", "gap", "n_scenarios"]].to_string(
            index=False, float_format="%.3f"
        ))
        print()

    # --- Summary: most uniquely problematic principle per multi-VP ---
    print()
    print("=" * 110)
    print("Most uniquely problematic principle per VP (largest negative gap)")
    print("=" * 110)
    print()

    worst = multi.loc[multi.groupby("vulnerable_population")["gap"].idxmin()]
    worst = worst.sort_values("gap")
    print(worst[["vulnerable_population", "principle", "bad_mean", "all_vp_avg", "gap", "n_scenarios"]].to_string(
        index=False, float_format="%.3f"
    ))

    # --- Single-principle VPs (mostly Design for Equity & Inclusion) ---
    print()
    print()
    print("=" * 110)
    print("Single-principle VPs (1 principle only)")
    print("=" * 110)
    print()
    single_sorted = single.sort_values("gap")
    print(single_sorted[["vulnerable_population", "principle", "bad_mean", "all_vp_avg", "gap", "n_scenarios"]].to_string(
        index=False, float_format="%.3f"
    ))
