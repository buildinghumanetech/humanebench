#!/usr/bin/env python3
"""Generate tables and figure for Section 4.5 (Vulnerable Populations).

Outputs:
  tables/vp_table5_by_principle.csv   — VP vs non-VP steerability by principle
  tables/vp_table6_age_gradient.csv   — age-group summary (children/teenagers/elderly)
  figures/vp_age_gradient.png         — grouped dot chart for age groups
  figures/vp_age_gradient.pdf         — PDF version for LaTeX
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(TABLES / "vp_sample_scores.csv")
vp = df[df["vulnerable_population"].notna() & (df["vulnerable_population"] != "")].copy()
non_vp = df[df["vulnerable_population"].isna() | (df["vulnerable_population"] == "")].copy()

PRINCIPLE_SHORT = {
    "respect-user-attention": "Respect User Attention",
    "be-transparent-and-honest": "Be Transparent and Honest",
    "enable-meaningful-choices": "Enable Meaningful Choices",
    "enhance-human-capabilities": "Enhance Human Capabilities",
    "foster-healthy-relationships": "Foster Healthy Relationships",
    "prioritize-long-term-wellbeing": "Prioritize Long-term Wellbeing",
    "protect-dignity-and-safety": "Protect Dignity and Safety",
    "design-for-equity-and-inclusion": "Design for Equity and Inclusion",
}

COLORS = {
    "baseline": "#000000",
    "good": "#16A34A",
    "bad": "#DC2626",
    "ref_line": "#9CA3AF",
    "background": "#FFFFFF",
    "grid": "#E5E7EB",
}


# ── Table 5: VP vs non-VP by principle ──────────────────────────────

def make_table5():
    rows = []
    for principle in PRINCIPLE_SHORT:
        vp_sub = vp[vp["principle"] == principle]
        non_vp_sub = non_vp[non_vp["principle"] == principle]
        n_vp = vp_sub["sample_id"].nunique()

        vp_bl = vp_sub[vp_sub["persona"] == "baseline"]["score"].mean()
        vp_good = vp_sub[vp_sub["persona"] == "good_persona"]["score"].mean()
        vp_bad = vp_sub[vp_sub["persona"] == "bad_persona"]["score"].mean()
        vp_delta = vp_bad - vp_bl

        non_vp_bl = non_vp_sub[non_vp_sub["persona"] == "baseline"]["score"].mean()
        non_vp_delta = (
            non_vp_sub[non_vp_sub["persona"] == "bad_persona"]["score"].mean() - non_vp_bl
            if len(non_vp_sub) > 0 else float("nan")
        )
        baseline_gap = vp_bl - non_vp_bl if not np.isnan(non_vp_bl) else float("nan")

        rows.append({
            "principle": PRINCIPLE_SHORT[principle],
            "n_vp": n_vp,
            "vp_baseline": vp_bl,
            "vp_good": vp_good,
            "vp_bad": vp_bad,
            "vp_delta_bad": vp_delta,
            "non_vp_baseline": non_vp_bl,
            "non_vp_delta_bad": non_vp_delta,
            "baseline_gap": baseline_gap,
        })

    t5 = pd.DataFrame(rows)
    t5.to_csv(TABLES / "vp_table5_by_principle.csv", index=False, float_format="%.3f")
    print(f"Wrote {TABLES / 'vp_table5_by_principle.csv'}")
    print(t5.to_string(index=False, float_format="%.2f"))
    return t5


# ── Table 6: Age gradient ──────────────────────────────────────────

def make_table6():
    age_groups = ["children", "teenagers", "elderly"]
    rows = []
    for ag in age_groups:
        sub = vp[vp["vulnerable_population"] == ag]
        bl = sub[sub["persona"] == "baseline"]["score"]
        bad = sub[sub["persona"] == "bad_persona"]["score"]
        good = sub[sub["persona"] == "good_persona"]["score"]
        rows.append({
            "group": ag.capitalize(),
            "n_principles": sub["principle"].nunique(),
            "n_scenarios": sub["sample_id"].nunique(),
            "baseline": bl.mean(),
            "good_persona": good.mean(),
            "bad_persona": bad.mean(),
            "delta_bad": bad.mean() - bl.mean(),
            "delta_good": good.mean() - bl.mean(),
        })

    t6 = pd.DataFrame(rows)
    t6.to_csv(TABLES / "vp_table6_age_gradient.csv", index=False, float_format="%.3f")
    print(f"\nWrote {TABLES / 'vp_table6_age_gradient.csv'}")
    print(t6.to_string(index=False, float_format="%.3f"))
    return t6


# ── Figure: Age gradient dot chart ─────────────────────────────────

def make_figure(t6):
    groups = t6["group"].tolist()
    n = len(groups)

    non_vp_bl = non_vp[non_vp["persona"] == "baseline"]["score"].mean()
    non_vp_good = non_vp[non_vp["persona"] == "good_persona"]["score"].mean()
    non_vp_bad = non_vp[non_vp["persona"] == "bad_persona"]["score"].mean()

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    x = np.arange(n)

    for i, (_, row) in enumerate(t6.iterrows()):
        ax.plot(
            [x[i], x[i]],
            [row["bad_persona"], row["good_persona"]],
            color=COLORS["grid"], linewidth=1.5, zorder=1,
        )

    ax.scatter(x, t6["baseline"], color=COLORS["baseline"],
               marker="o", s=50, zorder=3, label="Baseline")
    ax.scatter(x, t6["good_persona"], color=COLORS["good"],
               marker="^", s=50, zorder=3, label="Good persona")
    ax.scatter(x, t6["bad_persona"], color=COLORS["bad"],
               marker="v", s=50, zorder=3, label="Bad persona")

    ax.axhline(y=non_vp_bl, color=COLORS["ref_line"], linestyle="--",
               linewidth=0.8, zorder=0)
    ax.text((n - 1) / 2.0, non_vp_bl + 0.02, "non-VP baseline",
            fontsize=6.5, color=COLORS["ref_line"], ha="center", va="bottom")

    ax.axhline(y=0.0, color="#000000", linewidth=0.5, zorder=0)
    ax.axhline(y=0.5, color=COLORS["ref_line"], linewidth=0.5,
               linestyle=":", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Mean HumaneScore", fontsize=9)
    ax.set_ylim(-0.15, 1.05)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7, loc="upper center", ncol=3, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, zorder=0)

    plt.tight_layout()
    fig.savefig(FIGURES / "vp_age_gradient.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "vp_age_gradient.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIGURES / 'vp_age_gradient.pdf'}")
    print(f"Wrote {FIGURES / 'vp_age_gradient.png'}")


if __name__ == "__main__":
    t5 = make_table5()
    t6 = make_table6()
    make_figure(t6)
