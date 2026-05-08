#!/usr/bin/env python3
"""Good persona recovery analysis for vulnerable populations.

How much does a humane system prompt improve VP treatment, and does it
close the gap with non-VP performance? Reports:
1. Per-VP summary: baseline, good_persona, and the good-persona delta (good - baseline)
2. Per-principle VP vs non-VP comparison under good persona
"""

import pandas as pd
from pathlib import Path

TABLES = Path(__file__).resolve().parent.parent / "tables"

df = pd.read_csv(TABLES / "vp_sample_scores.csv")
vp = df[df["vulnerable_population"].notna() & (df["vulnerable_population"] != "")].copy()
non_vp = df[df["vulnerable_population"].isna() | (df["vulnerable_population"] == "")].copy()

# ---------- Part 1: Per-VP good persona lift ----------

rows = []
for vp_name, group in vp.groupby("vulnerable_population"):
    bl = group[group["persona"] == "baseline"]["score"]
    gp = group[group["persona"] == "good_persona"]["score"]
    bp = group[group["persona"] == "bad_persona"]["score"]
    rows.append({
        "vulnerable_population": vp_name,
        "n_principles": group["principle"].nunique(),
        "baseline_mean": bl.mean(),
        "good_persona_mean": gp.mean(),
        "delta_good": gp.mean() - bl.mean(),
        "bad_persona_mean": bp.mean(),
        "n_scenarios": group["sample_id"].nunique(),
    })

summary = pd.DataFrame(rows).sort_values("delta_good", ascending=False)

print("=" * 110)
print("Per-VP good persona recovery (good_persona - baseline), pooled across models")
print("=" * 110)
print()
print(summary.to_string(index=False, float_format="%.3f"))

# ---------- Part 2: Per-principle VP vs non-VP under good persona ----------

vp_good = vp[vp["persona"] == "good_persona"].groupby("principle")["score"].mean()
non_vp_good = non_vp[non_vp["persona"] == "good_persona"].groupby("principle")["score"].mean()
vp_bl = vp[vp["persona"] == "baseline"].groupby("principle")["score"].mean()
non_vp_bl = non_vp[non_vp["persona"] == "baseline"].groupby("principle")["score"].mean()

comparison = pd.DataFrame({
    "vp_baseline": vp_bl,
    "non_vp_baseline": non_vp_bl,
    "baseline_gap": vp_bl - non_vp_bl,
    "vp_good": vp_good,
    "non_vp_good": non_vp_good,
    "good_gap": vp_good - non_vp_good,
    "vp_delta_good": vp_good - vp_bl,
    "non_vp_delta_good": non_vp_good - non_vp_bl,
}).dropna()

comparison["gap_closed"] = comparison["good_gap"] - comparison["baseline_gap"]
comparison = comparison.sort_values("good_gap")

print()
print()
print("=" * 110)
print("Per-principle: VP vs non-VP under good persona")
print("baseline_gap = VP baseline - non-VP baseline")
print("good_gap     = VP good_persona - non-VP good_persona")
print("gap_closed   = how much of the baseline gap was closed by the good persona")
print("=" * 110)
print()
print(comparison.to_string(float_format="%.3f"))

# ---------- Part 3: Focus on floor-effect principles ----------

print()
print()
print("=" * 110)
print("Focus: Be Transparent and Respect User Attention (floor-effect principles)")
print("=" * 110)
print()

focus = ["be-transparent-and-honest", "respect-user-attention"]
for p in focus:
    print(f"--- {p} ---")
    vp_sub = vp[vp["principle"] == p]
    non_vp_sub = non_vp[non_vp["principle"] == p]

    for persona in ["baseline", "good_persona", "bad_persona"]:
        vp_mean = vp_sub[vp_sub["persona"] == persona]["score"].mean()
        non_vp_mean = non_vp_sub[non_vp_sub["persona"] == persona]["score"].mean()
        print(f"  {persona:15s}  VP: {vp_mean:+.3f}   non-VP: {non_vp_mean:+.3f}   gap: {vp_mean - non_vp_mean:+.3f}")
    print()
