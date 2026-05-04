"""
Compute the binarized robustness gap as an AI-side robustness check for §3.5.

The good-persona stratified α_ord on the AI side was 0.29 (judges noisier at
the positive end of the scale). A reviewer will rightly ask whether the headline
robustness gap (good HumaneScore − bad HumaneScore) is an artifact of
ordinal-end disagreements rather than real prosocial-flipping.

This script answers that by recomputing the per-model robustness gap using only
the binarized prosocial-flip signal — fraction of items where the ensemble's
mean severity is positive — and checking whether the gap survives in BOTH
ordering AND magnitude.

Inputs (read-only):
  - tables/inter_judge_raw.csv (35,956 rows: sample_uid, persona, model,
    principle, judge_name, severity)

Outputs (written to tables/):
  - robustness_gap_binarized.csv  — per-model gaps (ordinal + binarized)
  - robustness_gap_binarized.md    — Spearman ρ + Pearson r + mean ratio
                                     headline + warnings if thresholds tripped
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_BOOTSTRAP_DEFAULT,
    binarize_long,
    bootstrap_persona_deltas,
    load_long_scores,
)

# Same warning thresholds as the plan.
SPEARMAN_WARN = 0.85    # below this → "rank ordering not preserved"
MEAN_RATIO_WARN = 0.7   # below this → "binarized gaps systematically smaller"


def build_per_item_ensemble(judge_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long judge table to one row per (sample_uid, persona, model)."""
    grouped = judge_raw_df.groupby(
        ["sample_uid", "persona", "model"], as_index=False
    ).agg(
        mean_severity=("severity", "mean"),
        n_judges=("severity", "count"),
    )
    grouped["prosocial"] = (grouped["mean_severity"] >= 0).astype(int)
    return grouped


def compute_per_model_gaps(per_item: pd.DataFrame) -> pd.DataFrame:
    """For each model, compute prosocial rate and mean severity per persona, plus gaps."""
    rows = []
    for model, sub in per_item.groupby("model"):
        cell = {"model": model}
        for persona in ["baseline", "good_persona", "bad_persona"]:
            cell_sub = sub[sub["persona"] == persona]
            cell[f"n_items_{persona}"] = int(len(cell_sub))
            cell[f"prosocial_rate_{persona}"] = (
                float(cell_sub["prosocial"].mean()) if len(cell_sub) else float("nan")
            )
            cell[f"mean_severity_{persona}"] = (
                float(cell_sub["mean_severity"].mean()) if len(cell_sub) else float("nan")
            )
        # benevolent − adversarial
        cell["binarized_gap"] = (
            cell["prosocial_rate_good_persona"] - cell["prosocial_rate_bad_persona"]
        )
        cell["ordinal_gap"] = (
            cell["mean_severity_good_persona"] - cell["mean_severity_bad_persona"]
        )
        cell["gap_ratio"] = (
            cell["binarized_gap"] / cell["ordinal_gap"]
            if cell["ordinal_gap"] != 0
            else float("nan")
        )
        rows.append(cell)
    df = pd.DataFrame(rows)
    df = df.sort_values("ordinal_gap", ascending=False).reset_index(drop=True)
    df["rank_ordinal"] = df["ordinal_gap"].rank(ascending=False, method="min").astype(int)
    df["rank_binarized"] = df["binarized_gap"].rank(ascending=False, method="min").astype(int)
    return df


def add_gap_cis(
    df: pd.DataFrame,
    judge_raw_csv: Path,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Attach 95% paired bootstrap CIs to ordinal_gap and binarized_gap.

    Both gaps are paired-scenario deltas (`good_persona - bad_persona`); CIs
    use the same per-(model) paired resampling as
    `humanebench.bootstrap.bootstrap_persona_deltas` (HumaneScore-level), so
    they're directly comparable to the headline persona-delta CIs.
    """
    long = load_long_scores(judge_raw_csv)

    ord_deltas = bootstrap_persona_deltas(
        long,
        baseline_persona="bad_persona",
        contrast_personas=["good_persona"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    ord_h = ord_deltas[ord_deltas["principle"] == "HumaneScore"].set_index("model")

    bin_deltas = bootstrap_persona_deltas(
        binarize_long(long),
        baseline_persona="bad_persona",
        contrast_personas=["good_persona"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    bin_h = bin_deltas[bin_deltas["principle"] == "HumaneScore"].set_index("model")

    df = df.copy()
    df["ordinal_gap_ci_lower"] = df["model"].map(ord_h["ci_lower"])
    df["ordinal_gap_ci_upper"] = df["model"].map(ord_h["ci_upper"])
    df["binarized_gap_ci_lower"] = df["model"].map(bin_h["ci_lower"])
    df["binarized_gap_ci_upper"] = df["model"].map(bin_h["ci_upper"])
    return df


def write_outputs(df: pd.DataFrame, tables_dir: Path) -> None:
    out_cols = [
        "model",
        "n_items_baseline", "n_items_good_persona", "n_items_bad_persona",
        "prosocial_rate_baseline", "prosocial_rate_good_persona", "prosocial_rate_bad_persona",
        "binarized_gap", "binarized_gap_ci_lower", "binarized_gap_ci_upper",
        "mean_severity_baseline", "mean_severity_good_persona", "mean_severity_bad_persona",
        "ordinal_gap", "ordinal_gap_ci_lower", "ordinal_gap_ci_upper",
        "gap_ratio",
        "rank_ordinal", "rank_binarized",
    ]
    # Tolerate the script being run before CI columns are attached (legacy path).
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(tables_dir / "robustness_gap_binarized.csv", index=False)

    spearman_rho, spearman_p = spearmanr(df["ordinal_gap"], df["binarized_gap"])
    pearson_r, pearson_p = pearsonr(df["ordinal_gap"], df["binarized_gap"])
    mean_ratio = float(df["gap_ratio"].mean())

    warnings = []
    if spearman_rho < SPEARMAN_WARN:
        warnings.append(
            f"**RANK ORDERING NOT PRESERVED** — Spearman ρ = {spearman_rho:.3f} "
            f"is below the {SPEARMAN_WARN} threshold. The model ordering shifts "
            "meaningfully when we binarize. The robustness story is not insulated "
            "from the ordinal-end-noise concern."
        )
    if mean_ratio < MEAN_RATIO_WARN:
        warnings.append(
            f"**MAGNITUDE SHRINKS** — mean(binarized_gap / ordinal_gap) = "
            f"{mean_ratio:.3f} is below the {MEAN_RATIO_WARN} threshold. The "
            "binarized gaps are systematically smaller than the ordinal gaps. "
            "The robustness story shrinks materially under binarization."
        )

    print("=" * 72)
    print("BINARIZED ROBUSTNESS GAP — SUMMARY")
    print("=" * 72)
    print(f"Models: {len(df)}")
    print(f"Spearman ρ (rank preservation): {spearman_rho:+.3f}  (p={spearman_p:.4f})")
    print(f"Pearson r  (magnitude correlation): {pearson_r:+.3f}  (p={pearson_p:.4f})")
    print(f"Mean ratio (binarized / ordinal): {mean_ratio:+.3f}")
    print()
    print(df[["model", "ordinal_gap", "binarized_gap", "gap_ratio",
              "rank_ordinal", "rank_binarized"]].to_string(index=False))
    print()
    if warnings:
        print("!" * 72)
        for w in warnings:
            print(w)
            print()
        print("!" * 72)
    else:
        print("All thresholds passed — the robustness story survives binarization "
              "in both ordering and magnitude.")
    print()

    md = [
        "# Binarized robustness gap (AI-side robustness check)",
        "",
        "This is a robustness check for the headline result that bad-persona "
        "system prompts erode HumaneScores. The good-persona stratified α_ord "
        "on the AI side is only 0.29 (judges are noisier at the positive end "
        "of the scale), so a reviewer will reasonably ask whether the gap is "
        "an artifact of ordinal disagreements rather than real prosocial flips. "
        "We answer that by recomputing the per-model gap using only the "
        "binarized prosocial-flip signal (fraction of items with ensemble mean "
        "severity >= 0) and checking whether the gap survives in both ordering "
        "and magnitude.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| Models | {len(df)} |",
        f"| Spearman ρ (rank preservation) | {spearman_rho:+.3f} (p={spearman_p:.4f}) |",
        f"| Pearson r (magnitude correlation) | {pearson_r:+.3f} (p={pearson_p:.4f}) |",
        f"| Mean ratio (binarized / ordinal) | {mean_ratio:+.3f} |",
        "",
    ]
    if warnings:
        md += ["## ⚠ Warnings", ""]
        for w in warnings:
            md.append(f"- {w}")
        md.append("")
    else:
        md += [
            "## Verdict",
            "",
            "The robustness story **survives binarization** in both ordering "
            f"(Spearman ρ = {spearman_rho:+.3f}) and magnitude "
            f"(mean ratio = {mean_ratio:+.3f}). The headline gap is not an "
            "artifact of ordinal-end disagreements between judges.",
            "",
        ]
    md += [
        "## Per-model gaps",
        "",
        "Sorted by ordinal gap (descending). Higher gap = more behavioral drift "
        "between benevolent and adversarial system prompts.",
        "",
        "| model | n_good | n_bad | ord_gap | ord_gap 95% CI | bin_gap | bin_gap 95% CI | ratio | rank_ord | rank_bin |",
        "| --- | ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: | ---: |",
    ]

    def _ci(row, lo_col, hi_col):
        lo = row.get(lo_col)
        hi = row.get(hi_col)
        if pd.isna(lo) or pd.isna(hi):
            return "—"
        return f"[{lo:+.3f}, {hi:+.3f}]"

    for _, row in df.iterrows():
        md.append(
            f"| {row['model']} | {row['n_items_good_persona']} | "
            f"{row['n_items_bad_persona']} | {row['ordinal_gap']:+.3f} | "
            f"{_ci(row, 'ordinal_gap_ci_lower', 'ordinal_gap_ci_upper')} | "
            f"{row['binarized_gap']:+.3f} | "
            f"{_ci(row, 'binarized_gap_ci_lower', 'binarized_gap_ci_upper')} | "
            f"{row['gap_ratio']:+.3f} | "
            f"{row['rank_ordinal']} | {row['rank_binarized']} |"
        )
    md += [
        "",
        "## How to use this table",
        "",
        "- **Ordinal gap** uses mean severity on `{-1, -0.5, +0.5, +1}` "
        "(the same scale as HumaneScore).",
        "- **Binarized gap** uses prosocial rate (fraction of items with "
        "ensemble mean >= 0).",
        "- **Ratio** is `binarized_gap / ordinal_gap`. Values near 1 indicate "
        "the binarized signal preserves the magnitude; lower values mean the "
        "ordinal signal carries information beyond the sign.",
        "",
        "If both Spearman ρ ≥ 0.85 and mean ratio ≥ 0.7, this table goes to "
        "the appendix as a robustness check and the headline result stands. "
        "If either threshold fails, this becomes a main-text finding worth "
        "discussing in §3.5 or §4.",
    ]
    (tables_dir / "robustness_gap_binarized.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-raw-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables/inter_judge_raw.csv",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables",
    )
    args = parser.parse_args()

    print(f"Loading inter-judge raw scores from: {args.judge_raw_csv}")
    judge_raw_df = pd.read_csv(args.judge_raw_csv)
    print(f"  loaded {len(judge_raw_df)} per-judge rows")

    per_item = build_per_item_ensemble(judge_raw_df)
    print(f"  collapsed to {len(per_item)} per-item ensemble rows")

    gaps = compute_per_model_gaps(per_item)
    print(f"  computed gaps for {len(gaps)} models")

    print(f"  bootstrapping ordinal/binarized gap CIs (n={N_BOOTSTRAP_DEFAULT}, "
          f"seed={BOOTSTRAP_SEED}) ...")
    gaps = add_gap_cis(gaps, args.judge_raw_csv)
    print()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(gaps, args.tables_dir)
    print(f"Outputs written to {args.tables_dir}/robustness_gap_binarized.{{csv,md}}")


if __name__ == "__main__":
    main()
