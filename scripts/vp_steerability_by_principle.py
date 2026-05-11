#!/usr/bin/env python3
"""Steerability by principle for VP vs non-VP scenarios.

For each principle, compute the mean score under bad_persona vs baseline
(pooled across all models), then report the delta. A large negative delta
means the bad persona successfully degrades that principle.

Reports two tables side by side: one for VP-tagged scenarios only,
one for non-VP (general population) scenarios, so readers can see
whether VP steerability patterns differ from the general case.
"""

import pandas as pd
from pathlib import Path

TABLES = Path(__file__).resolve().parent.parent / "tables"

df = pd.read_csv(TABLES / "vp_sample_scores.csv")

vp = df[df["vulnerable_population"].notna() & (df["vulnerable_population"] != "")].copy()
non_vp = df[df["vulnerable_population"].isna() | (df["vulnerable_population"] == "")].copy()


def steerability_table(subset, label):
    baseline = subset[subset["persona"] == "baseline"].groupby("principle")["score"].mean()
    bad = subset[subset["persona"] == "bad_persona"].groupby("principle")["score"].mean()
    good = subset[subset["persona"] == "good_persona"].groupby("principle")["score"].mean()

    summary = pd.DataFrame({
        "baseline_mean": baseline,
        "bad_persona_mean": bad,
        "good_persona_mean": good,
    })
    summary["delta_bad_minus_baseline"] = summary["bad_persona_mean"] - summary["baseline_mean"]
    summary["n_baseline"] = subset[subset["persona"] == "baseline"].groupby("principle")["score"].count()
    summary["n_bad"] = subset[subset["persona"] == "bad_persona"].groupby("principle")["score"].count()
    summary = summary.sort_values("delta_bad_minus_baseline")

    print(f"Total rows: {len(subset)}")
    print(f"Unique models: {subset['model'].nunique()}")
    print()
    print("=" * 90)
    print(f"{label}: steerability delta by principle (bad_persona - baseline)")
    print("=" * 90)
    print()
    print(summary.to_string(float_format="%.3f"))
    print()
    return summary


if __name__ == "__main__":
    print(">>> VP SCENARIOS <<<")
    print()
    vp_summary = steerability_table(vp, "VP scenarios")

    print()
    print(">>> NON-VP SCENARIOS <<<")
    print()
    non_vp_summary = steerability_table(non_vp, "Non-VP scenarios")

    # Side-by-side comparison of deltas
    print()
    print("=" * 90)
    print("Side-by-side: delta (bad - baseline) for VP vs non-VP")
    print("=" * 90)
    print()
    comparison = pd.DataFrame({
        "vp_delta": vp_summary["delta_bad_minus_baseline"],
        "non_vp_delta": non_vp_summary["delta_bad_minus_baseline"],
    }).dropna()
    comparison["difference"] = comparison["vp_delta"] - comparison["non_vp_delta"]
    comparison = comparison.sort_values("difference")
    print(comparison.to_string(float_format="%.3f"))
