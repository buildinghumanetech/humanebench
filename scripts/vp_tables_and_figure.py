#!/usr/bin/env python3
"""Generate the tables and figure for the Vulnerable Populations analysis.

Outputs:
  tables/vp_table5_by_principle.csv   — VP vs non-VP steerability by principle
  tables/vp_table6_age_gradient.csv   — age-group summary (children/teenagers/elderly)
  figures/vp_age_gradient.png         — horizontal candlestick chart for age groups
  figures/vp_age_gradient.pdf         — PDF version for LaTeX
"""
# Paper: produces tables/vp_table5_by_principle.csv -- the per-principle VP
#   vs non-VP HumaneScore table -- and figures/vp_age_gradient.{png,pdf}, the
#   age-group figure, both in the supplement's "Vulnerable-Population
#   Analysis"; the headline reading of them is the main paper's "Vulnerable
#   Populations" section.
# Paper: also produces tables/vp_table6_age_gradient.csv, the children /
#   teenagers / elderly summary the figure plots.
# Paper: "table5" / "table6" in the filenames are draft-era numbers. The
#   submitted paper numbers only Table 1, in the main text; supplement tables
#   are cited by section name.

from pathlib import Path

import matplotlib.patches as mpatches
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
    order = ["Children", "Teenagers", "Elderly"]
    t6 = t6.set_index("group").loc[order].reset_index()
    groups = t6["group"].tolist()
    n = len(groups)

    non_vp_bl = non_vp[non_vp["persona"] == "baseline"]["score"].mean()

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])

    bar_lw = 3
    cap_size = 9
    dot_size = 6

    for i, (_, row) in enumerate(t6.iterrows()):
        y_pos = n - i - 1

        baseline = row["baseline"]
        good = row["good_persona"]
        bad = row["bad_persona"]

        ax.plot([baseline, good], [y_pos, y_pos],
                color=COLORS["good"], linewidth=bar_lw,
                solid_capstyle="butt", zorder=2)
        ax.plot([good], [y_pos], marker="|", markersize=cap_size,
                color=COLORS["good"], markeredgewidth=2, zorder=2)

        ax.plot([bad, baseline], [y_pos, y_pos],
                color=COLORS["bad"], linewidth=bar_lw,
                solid_capstyle="butt", zorder=2)
        ax.plot([bad], [y_pos], marker="|", markersize=cap_size,
                color=COLORS["bad"], markeredgewidth=2, zorder=2)

        ax.plot([baseline], [y_pos], marker="o", markersize=dot_size,
                color=COLORS["baseline"], markeredgecolor="white",
                markeredgewidth=1.5, zorder=3)

    ax.axvline(x=0, color=COLORS["baseline"], linewidth=1.5,
               linestyle="--", alpha=0.7, zorder=1)
    ax.axvline(x=non_vp_bl, color=COLORS["ref_line"], linewidth=1.0,
               linestyle="--", alpha=0.5, zorder=1)
    ax.text(non_vp_bl, n - 0.3, "non-VP\nbaseline",
            fontsize=5.5, color=COLORS["ref_line"],
            ha="center", va="bottom")

    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(-0.5, n - 0.5)

    ax.set_xlabel("HumaneScore", fontsize=9, fontweight="bold")
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["-0.2", "0.0", "+0.2", "+0.4", "+0.6", "+0.8", "+1.0"])
    ax.tick_params(axis="x", labelsize=7)

    y_labels = groups[::-1]
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.tick_params(axis="y", length=0)

    ax.grid(True, axis="x", alpha=0.2, color=COLORS["grid"],
            linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["good"], label="Good persona"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["baseline"],
               markeredgecolor="white", markersize=6, label="Baseline"),
        mpatches.Patch(facecolor=COLORS["bad"], label="Bad persona"),
        Line2D([0], [0], color=COLORS["baseline"], linewidth=1.5,
               linestyle="--", alpha=0.7, label="Harmful threshold"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6.5,
              frameon=True, fancybox=False, shadow=False)

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
