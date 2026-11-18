#!/usr/bin/env python3
"""Compare as-deployed vs rules-suppressed pairwise interactions.

Run B of the discriminant follow-up: five global rules suppressed, two kept.
This is a robustness check on the additivity assumption, not an alternative
estimate of the same quantity — the instrument was changed.

Inputs (read-only):
  - tables/discriminant/pairwise_interactions.csv     (as-deployed)
  - tables/discriminant_rules27/pairwise_interactions.csv
  - tables/discriminant/matrix_long.csv
  - tables/discriminant_rules27/matrix_long.csv

Outputs:
  - tables/discriminant_rules27/comparison_pairs.csv
  - tables/discriminant_rules27/comparison_cells.csv
  - results/discriminant_rules27_comparison.md
"""
# Paper: produces tables/discriminant_rules27/comparison_pairs.csv and
# comparison_cells.csv -- the rules-suppressed replication of the pairwise
# interactions, with five of the seven shared global rules removed from the judge
# template (main paper, "Principle Separability").
# Paper: implements the as-deployed versus rules-suppressed comparison reported
# there: sign agreement across the 28 pairs, Pearson and Spearman correlation of
# the two sets of interaction estimates, Holm pass counts under each template,
# and the cell-level severity shift split by diagonal and off-diagonal.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import PRINCIPLES  # noqa: E402
from humanebench.discriminant import PRINCIPLE_SHORT  # noqa: E402

ALPHA = 0.05


def _fmt(x: float, places: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{places}f}".replace("-", "−")


def short(p: str) -> str:
    return PRINCIPLE_SHORT[p]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployed-dir", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant")
    ap.add_argument("--rules27-dir", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant_rules27")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "results" / "discriminant_rules27_comparison.md")
    args = ap.parse_args()
    out_dir = args.output_dir or args.rules27_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    deployed_pw = pd.read_csv(args.deployed_dir / "pairwise_interactions.csv")
    rules27_pw = pd.read_csv(args.rules27_dir / "pairwise_interactions.csv")

    merged = deployed_pw.merge(
        rules27_pw, on=["principle_a", "principle_b"], suffixes=("_deployed", "_rules27"))

    merged["sign_agree"] = (
        np.sign(merged["interaction_deployed"]) == np.sign(merged["interaction_rules27"]))
    n_agree = int(merged["sign_agree"].sum())

    sig_deployed = "significant_deployed" if "significant_deployed" in merged.columns else None
    sig_rules27 = "significant_rules27" if "significant_rules27" in merged.columns else None

    holm_cols_deployed = [c for c in merged.columns if c.startswith("p_holm") and c.endswith("_deployed")]
    holm_cols_rules27 = [c for c in merged.columns if c.startswith("p_holm") and c.endswith("_rules27")]

    n_pass_deployed = int((merged[holm_cols_deployed[0]] < ALPHA).sum()) if holm_cols_deployed else "?"
    n_pass_rules27 = int((merged[holm_cols_rules27[0]] < ALPHA).sum()) if holm_cols_rules27 else "?"

    finite = merged[
        merged["interaction_deployed"].notna() & merged["interaction_rules27"].notna()
    ]
    if len(finite) >= 3:
        pearson_r, pearson_p = stats.pearsonr(
            finite["interaction_deployed"], finite["interaction_rules27"])
        spearman_r, spearman_p = stats.spearmanr(
            finite["interaction_deployed"], finite["interaction_rules27"])
    else:
        pearson_r = spearman_r = pearson_p = spearman_p = float("nan")

    merged.to_csv(out_dir / "comparison_pairs.csv", index=False)

    deployed_long = pd.read_csv(args.deployed_dir / "matrix_long.csv")
    rules27_long = pd.read_csv(args.rules27_dir / "matrix_long.csv")

    cell_merged = deployed_long.merge(
        rules27_long, on=["scenario_id", "source_model", "designed_principle", "scored_principle"],
        suffixes=("_deployed", "_rules27"), how="inner")
    cell_merged["shift"] = cell_merged["score_rules27"] - cell_merged["score_deployed"]
    cell_merged["is_diagonal"] = cell_merged["designed_principle"] == cell_merged["scored_principle"]

    diag = cell_merged[cell_merged["is_diagonal"]]
    offdiag = cell_merged[~cell_merged["is_diagonal"]]

    cell_merged.to_csv(out_dir / "comparison_cells.csv", index=False)

    L: list[str] = []
    A = L.append
    A("# Rules-suppressed robustness check — comparison report")
    A("")
    A("Global rules 1, 3, 4, 5, 6 suppressed; rules 2 and 7 kept (renumbered).")
    A("This is a robustness check on the additivity assumption, not an "
      "alternative estimate of the same quantity.")
    A("")
    A("## Pair-level comparison")
    A("")
    A(f"- **Sign agreement**: {n_agree}/28 pairs have the same sign")
    A(f"- **Pearson r**: {pearson_r:.3f} (p = {pearson_p:.4f})")
    A(f"- **Spearman rho**: {spearman_r:.3f} (p = {spearman_p:.4f})")
    A(f"- **Holm passes**: as-deployed {n_pass_deployed}/28, "
      f"rules-suppressed {n_pass_rules27}/28")
    A("")

    A("### Original 5 failing pairs under suppression")
    A("")
    A("| Pair | deployed | rules27 | deployed Holm | rules27 Holm |")
    A("|---|---|---|---|---|")
    original_fails = ["emc/pltw", "fhr/pltw", "pds/fhr", "rua/dei", "rua/pltw"]
    for _, r in merged.iterrows():
        pair = f"{short(r.principle_a)}/{short(r.principle_b)}"
        if pair in original_fails:
            h_d = r.get(holm_cols_deployed[0], "?") if holm_cols_deployed else "?"
            h_r = r.get(holm_cols_rules27[0], "?") if holm_cols_rules27 else "?"
            A(f"| {pair} | {_fmt(r.interaction_deployed)} | "
              f"{_fmt(r.interaction_rules27)} | {h_d:.4f} | {h_r:.4f} |")
    A("")

    A("## Cell-level severity shift")
    A("")
    A(f"- **All cells**: mean shift {cell_merged['shift'].mean():+.4f}, "
      f"n = {len(cell_merged)}")
    A(f"- **Diagonal**: mean shift {diag['shift'].mean():+.4f}, "
      f"n = {len(diag)}")
    A(f"- **Off-diagonal**: mean shift {offdiag['shift'].mean():+.4f}, "
      f"n = {len(offdiag)}")
    A("")
    A("**Caveats** (pre-stated in pre-commitment):")
    A("")
    A("- Rule 7 retains partial BATH content, so BATH's pairs are the "
      "least-cleaned.")
    A("- Removing rules 1/5/6 removes ceiling constraints, so any change "
      "confounds contamination removal with ceiling removal.")
    A("- This is a robustness check. The as-deployed result stays primary.")
    A("")

    args.report.write_text("\n".join(L))
    print(f"wrote {args.report.relative_to(REPO_ROOT)}")
    print(f"wrote {(out_dir / 'comparison_pairs.csv').relative_to(REPO_ROOT)}")
    print(f"wrote {(out_dir / 'comparison_cells.csv').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
