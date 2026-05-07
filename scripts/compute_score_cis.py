"""Compute bootstrap CIs for headline HumaneBench scores.

Wraps `humanebench.bootstrap` to produce two long-format tables that every
downstream script joins on:

  - tables/score_cis_long.csv          — per-(model, persona, principle) marginals
  - tables/persona_delta_cis_long.csv  — paired good/bad persona deltas

Resampling design: 95% percentile CIs from `n_bootstrap=1000` paired
scenario-resamples stratified by principle. Seed and conventions match
`scripts/compute_inter_judge_agreement.py` and
`scripts/compute_ensemble_vs_human_cis.py` so every CI in the paper is
generated under one bootstrap design.

Inputs (read-only):
  - tables/inter_judge_raw.csv  (produced by compute_inter_judge_agreement.py;
    already exclusion-filtered upstream)

Outputs (written to --output-dir):
  - score_cis_long.csv
  - persona_delta_cis_long.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_BOOTSTRAP_DEFAULT,
    bootstrap_cell_scores,
    bootstrap_persona_deltas,
    load_long_scores,
)

DEFAULT_RAW = REPO_ROOT / "tables" / "inter_judge_raw.csv"
DEFAULT_OUT = REPO_ROOT / "tables"

# Headline CSVs in the repo root that we can augment in-place without
# re-reading the .eval logs.
PRINCIPLES_TUPLE = (
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
)


def _augment_persona_csv(path: Path, persona: str, cells_df) -> None:
    """Add *_ci_lower / *_ci_upper columns to {persona}_scores.csv in place.

    Joins by (model, persona, principle). Leaves existing point estimates
    untouched.
    """
    import pandas as pd
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = cells_df[cells_df["persona"] == persona].copy()
    sub = sub.rename(columns={"principle": "_principle"})
    for principle in PRINCIPLES_TUPLE:
        ci = sub[sub["_principle"] == principle][["model", "ci_lower", "ci_upper"]]
        ci = ci.rename(columns={
            "ci_lower": f"{principle}_ci_lower",
            "ci_upper": f"{principle}_ci_upper",
        })
        df = df.merge(ci, on="model", how="left")
    hs = sub[sub["_principle"] == "HumaneScore"][["model", "ci_lower", "ci_upper"]]
    hs = hs.rename(columns={
        "ci_lower": "overall_ci_lower",
        "ci_upper": "overall_ci_upper",
    })
    df = df.merge(hs, on="model", how="left")

    # Re-order columns to keep CIs adjacent to their point estimates.
    base = ["model", "total_samples", "scored_samples"]
    principle_cols = []
    for p in PRINCIPLES_TUPLE:
        principle_cols.extend([p, f"{p}_ci_lower", f"{p}_ci_upper"])
    tail = ["overall", "overall_ci_lower", "overall_ci_upper", "negative_rate"]
    ordered = [c for c in base + principle_cols + tail if c in df.columns]
    df = df[ordered]
    df.to_csv(path, index=False)


def _augment_steerability_csv(path: Path, cells_df, deltas_df) -> None:
    """Add CI columns for marginal scores and paired deltas to steerability_comparison.csv."""
    import pandas as pd
    if not path.exists():
        return
    df = pd.read_csv(path)
    for persona in ("baseline", "good_persona", "bad_persona"):
        ci = cells_df[(cells_df["persona"] == persona)
                      & (cells_df["principle"] == "HumaneScore")][
            ["model", "ci_lower", "ci_upper"]
        ].rename(columns={
            "ci_lower": f"{persona}_score_ci_lower",
            "ci_upper": f"{persona}_score_ci_upper",
        })
        df = df.merge(ci, on="model", how="left")
    for contrast in ("good_persona", "bad_persona"):
        kind = "good_delta" if contrast == "good_persona" else "bad_delta"
        d = deltas_df[(deltas_df["contrast_persona"] == contrast)
                      & (deltas_df["principle"] == "HumaneScore")][
            ["model", "ci_lower", "ci_upper"]
        ].rename(columns={
            "ci_lower": f"{kind}_ci_lower",
            "ci_upper": f"{kind}_ci_upper",
        })
        df = df.merge(d, on="model", how="left")

    ordered = [
        "model",
        "baseline_score", "baseline_score_ci_lower", "baseline_score_ci_upper",
        "good_persona_score", "good_persona_score_ci_lower", "good_persona_score_ci_upper",
        "good_delta", "good_delta_ci_lower", "good_delta_ci_upper",
        "bad_persona_score", "bad_persona_score_ci_lower", "bad_persona_score_ci_upper",
        "bad_delta", "bad_delta_ci_lower", "bad_delta_ci_upper",
        "robustness_status",
        "baseline_negative_rate", "good_persona_negative_rate", "bad_persona_negative_rate",
    ]
    df = df[[c for c in ordered if c in df.columns]]
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW,
                        help="Path to inter_judge_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="Directory to write CI CSVs into")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--augment-existing-csvs",
        action="store_true",
        default=True,
        help="Also patch CI columns into existing baseline_scores.csv / "
             "good_persona_scores.csv / bad_persona_scores.csv / "
             "steerability_comparison.csv in the repo root (default on; lets "
             "you refresh CIs without re-running extract_all_scores.py).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.raw_csv} ...")
    long = load_long_scores(args.raw_csv)
    print(f"  {len(long):,} per-sample rows across "
          f"{long['model'].nunique()} models, {long['persona'].nunique()} personas")

    print(f"\nBootstrapping cell scores (n={args.n_bootstrap}, seed={args.seed}) ...")
    cells = bootstrap_cell_scores(long, n_bootstrap=args.n_bootstrap, seed=args.seed)
    out_cells = args.output_dir / "score_cis_long.csv"
    cells.to_csv(out_cells, index=False)
    print(f"  wrote {len(cells):,} rows -> {out_cells.relative_to(REPO_ROOT)}")

    print(f"\nBootstrapping paired persona deltas "
          f"(n={args.n_bootstrap}, seed={args.seed}) ...")
    deltas = bootstrap_persona_deltas(long, n_bootstrap=args.n_bootstrap, seed=args.seed)
    out_deltas = args.output_dir / "persona_delta_cis_long.csv"
    deltas.to_csv(out_deltas, index=False)
    print(f"  wrote {len(deltas):,} rows -> {out_deltas.relative_to(REPO_ROOT)}")

    if args.augment_existing_csvs:
        print("\nAugmenting existing repo-root CSVs with CI columns ...")
        for persona in ("baseline", "good_persona", "bad_persona"):
            target = REPO_ROOT / f"{persona}_scores.csv"
            if target.exists():
                _augment_persona_csv(target, persona, cells)
                print(f"  patched {target.relative_to(REPO_ROOT)}")
        steer = REPO_ROOT / "steerability_comparison.csv"
        if steer.exists():
            _augment_steerability_csv(steer, cells, deltas)
            print(f"  patched {steer.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
