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
  - tables/inter_judge_raw_regenerated.csv  (produced by compute_inter_judge_agreement.py;
    already exclusion-filtered upstream)

Outputs (written to --output-dir):
  - score_cis_long.csv
  - persona_delta_cis_long.csv
"""
# Paper: produces tables/score_cis_long.csv and persona_delta_cis_long.csv --
#   the per-(model, persona) HumaneScore point estimates and 95% CIs printed in
#   Table 1 ("Overall Performance"), and the baseline-positive / bad-negative
#   comparison behind the flip count in "The Anti-Humane Flip".
# Paper: implements the shared bootstrap protocol -- 95% percentile CIs from
#   1,000 paired scenario-level resamples stratified by principle, one seed and
#   one design for every CI in the paper (see humanebench/bootstrap.py).
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
from humanebench.tables import resolve_table  # noqa: E402

DEFAULT_RAW = REPO_ROOT / "tables" / "inter_judge_raw_regenerated.csv"
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


def _shown(path: Path) -> str:
    """Repo-relative path for display, or the full path if it lies outside.

    `--output-dir` may point anywhere; a progress message is no reason to
    abort a run that has already written its first file.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _without_ci_columns(df):
    """Drop any CI columns a previous run left behind.

    The augment functions merge freshly computed `*_ci_lower` / `*_ci_upper`
    columns in by model. If the target already carries columns of those names,
    pandas resolves the collision by suffixing both sides `_x` / `_y`, the
    canonical names vanish, and the column reorder at the end -- which keeps
    only names it recognises -- silently drops every confidence interval in
    the file. Running the script twice therefore used to strip the CIs it
    added the first time. Clearing them first makes the operation idempotent.
    """
    return df.drop(columns=[c for c in df.columns
                            if c.endswith(("_ci_lower", "_ci_upper"))])


def _require_full_cohort(path: Path, df, cells_df) -> None:
    """Refuse to augment a published CSV from a raw table that covers less of it.

    Clearing the stale CI columns is only safe when the incoming data can
    repopulate all of them. Merged `how="left"` from a narrower raw table --
    one decomposition condition, a single-persona export, a debugging slice --
    every model outside that table keeps its point estimate and silently loses
    its interval to NaN. These four CSVs are tracked published artifacts, and
    `--augment-existing-csvs` is on by default, so this needs no flag to fire.
    """
    have = set(cells_df["model"].unique())
    want = set(df["model"].unique())
    missing = sorted(want - have)
    if missing:
        raise SystemExit(
            f"{path.name} covers {len(want)} models but the supplied raw table "
            f"has only {len(have)}; {len(missing)} would lose their confidence "
            f"intervals to empty cells while keeping their point estimates "
            f"({', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}).\n"
            f"Re-run against the full-cohort table, or pass "
            f"--no-augment-existing-csvs to compute the CI tables without "
            f"touching the published CSVs."
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
    _require_full_cohort(path, df, cells_df)
    df = _without_ci_columns(df)
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
    _require_full_cohort(path, df, cells_df)
    df = _without_ci_columns(df)
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
                        help="Path to the per-judge long table")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="Directory to write CI CSVs into")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--augment-existing-csvs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also patch CI columns into existing baseline_scores.csv / "
             "good_persona_scores.csv / bad_persona_scores.csv / "
             "steerability_comparison.csv in the repo root (default on; lets "
             "you refresh CIs without re-running extract_all_scores.py).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {resolve_table(args.raw_csv)} ...")
    raw_csv = resolve_table(args.raw_csv)
    long = load_long_scores(raw_csv)
    print(f"  {len(long):,} per-sample rows across "
          f"{long['model'].nunique()} models, {long['persona'].nunique()} personas")

    print(f"\nBootstrapping cell scores (n={args.n_bootstrap}, seed={args.seed}) ...")
    cells = bootstrap_cell_scores(long, n_bootstrap=args.n_bootstrap, seed=args.seed)
    out_cells = args.output_dir / "score_cis_long.csv"
    cells.to_csv(out_cells, index=False)
    print(f"  wrote {len(cells):,} rows -> {_shown(out_cells)}")

    print(f"\nBootstrapping paired persona deltas "
          f"(n={args.n_bootstrap}, seed={args.seed}) ...")
    deltas = bootstrap_persona_deltas(long, n_bootstrap=args.n_bootstrap, seed=args.seed)
    out_deltas = args.output_dir / "persona_delta_cis_long.csv"
    deltas.to_csv(out_deltas, index=False)
    print(f"  wrote {len(deltas):,} rows -> {_shown(out_deltas)}")

    if args.augment_existing_csvs:
        print("\nAugmenting existing repo-root CSVs with CI columns ...")
        for persona in ("baseline", "good_persona", "bad_persona"):
            target = REPO_ROOT / f"{persona}_scores.csv"
            if target.exists():
                _augment_persona_csv(target, persona, cells)
                print(f"  patched {_shown(target)}")
        steer = REPO_ROOT / "steerability_comparison.csv"
        if steer.exists():
            _augment_steerability_csv(steer, cells, deltas)
            print(f"  patched {_shown(steer)}")


if __name__ == "__main__":
    main()
