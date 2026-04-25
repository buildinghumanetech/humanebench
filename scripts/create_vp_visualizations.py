#!/usr/bin/env python3
"""Generate VP-filtered visualizations from the flat score CSV.

Reads tables/vp_sample_scores.csv (produced by compute_vp_breakdown.py)
and generates heatmaps, drill-down bar charts, and a minor VP summary table.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURE_DIR = "figures"
TABLE_DIR = "tables"
INPUT_CSV = os.path.join(TABLE_DIR, "vp_sample_scores.csv")
MODEL_MAP_PATH = os.path.join(FIGURE_DIR, "model_display_names.json")

PERSONAS = ["baseline", "good_persona", "bad_persona"]
PERSONA_LABELS = {
    "baseline": "Baseline",
    "good_persona": "Good Persona",
    "bad_persona": "Bad Persona",
}

TARGET_VPS = ["children", "teenagers", "elderly"]

PRINCIPLES = [
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
]

PRINCIPLE_LABELS = {
    "respect-user-attention": "Respect Attention",
    "enable-meaningful-choices": "Enable Choices",
    "enhance-human-capabilities": "Enhance Capabilities",
    "protect-dignity-and-safety": "Protect Safety",
    "foster-healthy-relationships": "Foster Relationships",
    "prioritize-long-term-wellbeing": "Long-term Wellbeing",
    "be-transparent-and-honest": "Be Transparent",
    "design-for-equity-and-inclusion": "Equity & Inclusion",
}

MODEL_ORDER = [
    "claude-opus-4.1",
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "deepseek-v3.1-terminus",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gpt-4.1",
    "gpt-4o-2024-11-20",
    "gpt-5",
    "gpt-5.1",
    "grok-4",
    "llama-3.1-405b-instruct",
    "llama-4-maverick",
]


def load_model_map():
    with open(MODEL_MAP_PATH, "r") as f:
        return json.load(f)


def create_heatmap(df, vp, persona, model_map):
    """Create a single VP heatmap for a given VP and persona."""
    subset = df[(df["vulnerable_population"] == vp) & (df["persona"] == persona)]

    means = subset.groupby(["model", "principle"])["score"].mean().unstack(fill_value=np.nan)
    counts = subset.groupby(["model", "principle"])["score"].count().unstack(fill_value=0)

    means = means.reindex(index=MODEL_ORDER, columns=PRINCIPLES)
    counts = counts.reindex(index=MODEL_ORDER, columns=PRINCIPLES).fillna(0).astype(int)

    display_models = [model_map.get(m, m) for m in MODEL_ORDER]
    display_principles = [PRINCIPLE_LABELS[p] for p in PRINCIPLES]

    annot = np.empty_like(means.values, dtype=object)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            val = means.iloc[i, j]
            n = counts.iloc[i, j]
            if pd.isna(val) or n == 0:
                annot[i, j] = "n/a"
            else:
                annot[i, j] = f"{val:.3f}\n(n={n})"

    vmax = max(abs(means.min().min()), abs(means.max().max()))
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        means.values,
        annot=annot,
        fmt="",
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=display_principles,
        yticklabels=display_models,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Mean Score"},
    )
    ax.set_title(
        f"VP: {vp.title()} | Condition: {PERSONA_LABELS[persona]}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Principle", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(FIGURE_DIR, f"vp_heatmap_{vp}_{persona}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def generate_all_heatmaps(df, model_map):
    """Generate 9 heatmaps: 3 VPs x 3 personas."""
    print("Generating VP heatmaps...")
    for vp in TARGET_VPS:
        for persona in PERSONAS:
            create_heatmap(df, vp, persona, model_map)


def format_score(val):
    if pd.isna(val):
        return "N/A"
    return f"{val:.3f}"


def create_minor_vp_table(df):
    """Create summary table for VPs outside the big three."""
    print("Generating minor VP summary table...")
    equity_principle = "design-for-equity-and-inclusion"
    minor = df[
        (df["vulnerable_population"] != "")
        & (~df["vulnerable_population"].isin(TARGET_VPS))
        & (df["principle"] == equity_principle)
    ]

    rows = []
    for vp, group in minor.groupby("vulnerable_population"):
        n_scenarios = group["sample_id"].nunique()
        row = {"VP": vp, "n_scenarios": n_scenarios}
        for persona in PERSONAS:
            p_group = group[group["persona"] == persona]
            row[f"{PERSONA_LABELS[persona]} Mean"] = p_group["score"].mean()
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("n_scenarios", ascending=False)

    csv_path = os.path.join(TABLE_DIR, "vp_minor_summary.csv")
    result.to_csv(csv_path, index=False)

    md_path = os.path.join(TABLE_DIR, "vp_minor_summary.md")
    display = result.copy()
    for col in ["Baseline Mean", "Good Persona Mean", "Bad Persona Mean"]:
        display[col] = display[col].apply(format_score)
    with open(md_path, "w") as f:
        f.write("# Minor Vulnerable Population Summary\n\n")
        f.write("Scores for Design for Equity and Inclusion principle only. ")
        f.write("These VPs lack cross-principle coverage for full heatmaps.\n\n")
        f.write(display.to_markdown(index=False))
        f.write("\n")

    print(f"  Saved {csv_path}")
    print(f"  Saved {md_path}")


PERSONA_COLORS = {
    "baseline": "#9CA3AF",
    "good_persona": "#3B82F6",
    "bad_persona": "#DC2626",
}


def create_drilldown_chart(df, vp, principle, model_map):
    """Create a grouped bar chart for a specific VP x principle combination."""
    print(f"Generating drill-down chart: {vp} x {principle}...")
    subset = df[
        (df["vulnerable_population"] == vp) & (df["principle"] == principle)
    ]

    if subset.empty:
        print(f"  No data for VP={vp}, principle={principle}")
        return

    means = subset.groupby(["model", "persona"])["score"].mean().unstack(fill_value=np.nan)
    counts = subset.groupby(["model", "persona"])["score"].count().unstack(fill_value=0)

    means = means.reindex(index=MODEL_ORDER, columns=PERSONAS)
    counts = counts.reindex(index=MODEL_ORDER, columns=PERSONAS).fillna(0).astype(int)

    display_models = [model_map.get(m, m) for m in MODEL_ORDER]

    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 8))
    for i, persona in enumerate(PERSONAS):
        vals = means[persona].values
        ns = counts[persona].values
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=PERSONA_LABELS[persona],
            color=PERSONA_COLORS[persona],
            alpha=0.85,
        )
        for bar, n in zip(bars, ns):
            if n > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title(
        f"Drill-Down: {vp.title()} + {PRINCIPLE_LABELS.get(principle, principle)}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_models, rotation=40, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(FIGURE_DIR, f"vp_drilldown_{vp}_{principle}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VP-filtered visualizations"
    )
    parser.add_argument(
        "--drilldown",
        nargs=2,
        metavar=("VP", "PRINCIPLE"),
        help="Generate a drill-down bar chart for a specific VP and principle",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(INPUT_CSV)
    model_map = load_model_map()

    if args.drilldown:
        vp, principle = args.drilldown
        create_drilldown_chart(df, vp, principle, model_map)
    else:
        generate_all_heatmaps(df, model_map)
        create_minor_vp_table(df)


if __name__ == "__main__":
    main()
