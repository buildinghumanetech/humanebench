"""Compute cohort-mean CIs for the per-principle Table 4 cells.

Wraps `humanebench.bootstrap.bootstrap_cohort_principle_means` to produce two
long-format CSVs of across-model cohort means with 95% percentile bootstrap
CIs, plus the paired bad-persona delta CIs. Same resampling design and seed
as `compute_score_cis.py`.

Inputs (read-only):
  - tables/inter_judge_raw_regenerated.csv  (produced by compute_inter_judge_agreement.py)

Outputs (written to --output-dir):
  - cohort_principle_cis.csv         (one row per (principle, persona))
  - cohort_principle_delta_cis.csv   (one row per (principle, delta pair))
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
    bootstrap_cohort_principle_means,
    load_long_scores,
)
from scripts.generate_tables import MODEL_ORDER  # noqa: E402
from humanebench.tables import resolve_table  # noqa: E402

DEFAULT_RAW = REPO_ROOT / "tables" / "inter_judge_raw_regenerated.csv"
DEFAULT_OUT = REPO_ROOT / "tables"


def _shown(path: Path) -> str:
    """Repo-relative path for display, or the full path if it lies outside.

    `--output-dir` may point anywhere; a progress message is no reason to
    abort a run that has already written its first file.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW,
                        help="Path to the per-judge long table")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="Directory to write CI CSVs into")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {resolve_table(args.raw_csv)} ...")
    raw_csv = resolve_table(args.raw_csv)
    long = load_long_scores(raw_csv)
    available_models = set(long["model"].unique())
    missing = [m for m in MODEL_ORDER if m not in available_models]
    if missing:
        raise SystemExit(
            f"raw_csv missing scores for paper models: {missing}\n"
            f"Available models: {sorted(available_models)}"
        )
    print(f"  {len(long):,} per-sample rows; using {len(MODEL_ORDER)} paper models")

    print(f"\nBootstrapping cohort principle means (n={args.n_bootstrap}, seed={args.seed}) ...")
    cells, deltas = bootstrap_cohort_principle_means(
        long,
        models=MODEL_ORDER,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    out_cells = args.output_dir / "cohort_principle_cis.csv"
    cells.to_csv(out_cells, index=False)
    print(f"  wrote {len(cells):,} rows -> {_shown(out_cells)}")

    out_deltas = args.output_dir / "cohort_principle_delta_cis.csv"
    deltas.to_csv(out_deltas, index=False)
    print(f"  wrote {len(deltas):,} rows -> {_shown(out_deltas)}")

    # Print a human-readable summary table mirroring section_4.tex Table 4.
    print("\nCohort-mean per-principle scores (95% percentile bootstrap CIs):")
    pivot = cells.pivot(index="principle", columns="persona", values="point_estimate")
    pivot_lo = cells.pivot(index="principle", columns="persona", values="ci_lower")
    pivot_hi = cells.pivot(index="principle", columns="persona", values="ci_upper")
    bad_delta = deltas.set_index("principle")
    header = f"{'principle':<32} {'baseline':>22} {'good_persona':>22} {'bad_persona':>22} {'bad - base':>22}  n"
    print(header)
    print("-" * len(header))

    def cell(principle, persona):
        return (
            f"{pivot.loc[principle, persona]:+.3f} "
            f"[{pivot_lo.loc[principle, persona]:+.3f},{pivot_hi.loc[principle, persona]:+.3f}]"
        )

    for principle in pivot.index:
        bd = bad_delta.loc[principle]
        delta_str = (
            f"{bd['point_estimate']:+.3f} [{bd['ci_lower']:+.3f},{bd['ci_upper']:+.3f}]"
        )
        n = int(bd["n_scenarios"])
        print(
            f"{principle:<32} {cell(principle, 'baseline'):>22} "
            f"{cell(principle, 'good_persona'):>22} "
            f"{cell(principle, 'bad_persona'):>22} {delta_str:>22}  {n}"
        )


if __name__ == "__main__":
    main()
