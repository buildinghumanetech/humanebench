#!/usr/bin/env python3
"""
Render the HELM Safety × HumaneBench Δ_bad scatter.

Single panel: HELM Safety aggregate (x) vs Δ_bad = bad − baseline (y).
Matches the y-axis convention of scripts/create_aaai_helm_scatter.py so
the safety and capability scatters pair directly.
Family-colored markers, all points labeled, regression line, and stats box
with Spearman ρ, ρ_partial, permutation p, n. Mirrors the styling of
scripts/create_aaai_helm_scatter.py so the two figures pair visually.

Inputs:
  - tables/helm_safety_robustness_merged.csv   (from compute_helm_safety_partial_corr.py)
  - tables/helm_safety_robustness_stats.csv    (same)
  - figures/model_display_names.json

Output:
  - figures/helm_safety_vs_drop_scatter.{png,svg,pdf}

Run from the repo root:
    python scripts/create_helm_safety_robustness_scatter.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_CSV = REPO_ROOT / "tables" / "helm_safety_robustness_merged.csv"
STATS_CSV = REPO_ROOT / "tables" / "helm_safety_robustness_stats.csv"
NAMES_JSON = REPO_ROOT / "figures" / "model_display_names.json"
OUT_DIR = REPO_ROOT / "figures"
OUT_STEM = "helm_safety_vs_drop_scatter"

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 8
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.labelsize"] = 8
mpl.rcParams["axes.titlesize"] = 9
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7

FAMILY_COLORS = {
    "OpenAI":    "#3B82F6",
    "Anthropic": "#F97316",
    "Google":    "#10B981",
    "Meta":      "#8B5CF6",
    "xAI":       "#EF4444",
    "DeepSeek":  "#14B8A6",
    "Other":     "#6B7280",
}

ABOVE       = ("center", "bottom",  0,   7)
BELOW       = ("center", "top",     0,  -7)
ABOVE_CLOSE = ("center", "bottom",  0,   3)
BELOW_CLOSE = ("center", "top",     0,  -3)
# Diagonal placements: the (ha, va) anchor is the corner of the text closest
# to the dot, and (dx, dy) shifts that corner so a short leader line angles
# from the dot to the label.
BELOW_LEFT  = ("right",  "top",    -4,  -7)   # label down-and-to-the-left
BELOW_RIGHT = ("left",   "top",     4,  -7)   # label down-and-to-the-right

# Inherits from create_aaai_helm_scatter.py's overrides (same cohort, same
# y-axis convention) plus two more for the safety x-axis layout, where
# GPT-4.1 ↔ GPT-4o ↔ Gemini 3 Pro Preview cluster tightly in the bottom-middle
# and need to alternate to avoid label merge.
LABEL_DIRECTION = {
    "claude-sonnet-4":      BELOW,
    "gpt-5":                ABOVE_CLOSE,
    "gpt-5.1":              BELOW_CLOSE,
    "gpt-4o-2024-11-20":    BELOW,
    "gemini-3-pro-preview": BELOW,
    # Gemini 2.5 Flash and 2.0 Flash share x ≈ 0.91 — alternate ABOVE/BELOW
    # to keep their labels from stacking on the same row.
    "gemini-2.0-flash-001": BELOW_LEFT,
    "gemini-2.5-pro":       BELOW_RIGHT,
}

LABEL_OVERRIDES = {
    "gpt-4o-2024-11-20": "GPT-4o",
    "gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "llama-4-maverick": "Llama 4 Maverick",
    "llama-3.1-405b-instruct": "Llama 3.1 405B",
}


def significance(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_display_names() -> dict[str, str]:
    with open(NAMES_JSON) as f:
        return json.load(f)


def stat_lookup(stats: pd.DataFrame, name: str) -> dict | None:
    row = stats[stats["name"] == name]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    return r


def render(merged: pd.DataFrame, stats: pd.DataFrame, out_stem: Path) -> None:
    cohort = merged.dropna(subset=["safety_score", "delta_bad"]).copy()
    display_names = load_display_names()
    cohort["display"] = cohort["eval_model"].map(
        lambda s: LABEL_OVERRIDES.get(s, display_names.get(s, s))
    )
    cohort["color"] = cohort["family"].map(
        lambda f: FAMILY_COLORS.get(f, FAMILY_COLORS["Other"])
    )

    fig, ax = plt.subplots(figsize=(3.6, 3.0))

    seen = set()
    for _, row in cohort.iterrows():
        fam = row["family"]
        label = fam if fam not in seen else None
        seen.add(fam)
        ax.scatter(row["safety_score"], row["delta_bad"],
                   c=row["color"], s=28, zorder=3,
                   edgecolors="white", linewidths=0.5, label=label)
        ha, va, dx, dy = LABEL_DIRECTION.get(row["eval_model"], ABOVE)
        ax.annotate(row["display"],
                    xy=(row["safety_score"], row["delta_bad"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=5.5, ha=ha, va=va,
                    color="#1f2937", zorder=4,
                    arrowprops=dict(arrowstyle="-", color="#6B7280",
                                    lw=0.5, alpha=0.85,
                                    shrinkA=2.5, shrinkB=1.0))

    x = cohort["safety_score"].to_numpy(dtype=float)
    y = cohort["delta_bad"].to_numpy(dtype=float)
    if len(x) >= 3 and np.std(x) > 0:
        slope, intercept, _, _, _ = sp_stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, "k--", lw=0.8, alpha=0.5, zorder=2)

    ax.axhline(0, color="#9CA3AF", lw=0.5, ls=":", alpha=0.7, zorder=1)

    rho = stat_lookup(stats, "spearman_rho")
    partial = stat_lookup(stats, "spearman_rho_partial")
    box_lines = []
    if rho is not None:
        sig = significance(float(rho["p_value_perm"]))
        box_lines.append(
            f"$\\rho = {float(rho['point_estimate']):+.2f}$ "
            f"[{float(rho['ci_lower']):+.2f}, {float(rho['ci_upper']):+.2f}], "
            f"$p_{{\\mathrm{{perm}}}} = {float(rho['p_value_perm']):.3f}$ ({sig})"
        )
    if partial is not None:
        box_lines.append(
            f"$\\rho_{{\\mathrm{{partial}}}} = {float(partial['point_estimate']):+.2f}$ "
            f"[{float(partial['ci_lower']):+.2f}, {float(partial['ci_upper']):+.2f}]"
        )
    box_lines.append(f"$n = {len(cohort)}$")

    ax.text(0.04, 0.96, "\n".join(box_lines),
            transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#9CA3AF", linewidth=0.5))

    ax.set_xlabel("HELM Safety Aggregate (higher = safer)")
    ax.set_ylabel(r"$\Delta_{\mathrm{bad}} = S^{(\mathrm{bad})} - S^{(\mathrm{base})}$")
    ax.grid(True, alpha=0.18, lw=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    span_x = max(x.max() - x.min(), 1e-9)
    ax.set_xlim(x.min() - span_x * 0.12, x.max() + span_x * 0.18)
    span_y = max(y.max() - y.min(), 1e-9)
    ax.set_ylim(y.min() - span_y * 0.18, y.max() + span_y * 0.20)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels),
               frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("png", "svg", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> int:
    if not MERGED_CSV.exists() or not STATS_CSV.exists():
        print("ERROR: run scripts/compute_helm_safety_partial_corr.py first.",
              file=sys.stderr)
        return 1
    merged = pd.read_csv(MERGED_CSV)
    stats = pd.read_csv(STATS_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = OUT_DIR / OUT_STEM
    render(merged, stats, out_stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
