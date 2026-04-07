"""
Compute bootstrap CIs for ensemble-vs-human agreement on the curated 24-item
validation set.

The existing `scripts/human_judging_analysis/output/judge_vs_human_agreement_analysis.md`
reports point estimates for weighted Cohen's κ (0.905), Spearman's ρ (0.847),
Pearson's r (0.962), and direction-match rate (95.8%, 23/24) against the
ensemble scores that were computed against the pre-generated AI outputs in
`data/golden_questions.jsonl` (via `src/golden_questions_task.py`'s
`use_pregenerated_output()` solver). That document explicitly flags the lack
of confidence intervals:

  "Judge-vs-human metrics lack confidence intervals. The point estimates
  (kappa 0.905, Spearman 0.847) look strong, but at n=24 the confidence
  intervals would be wide."

This script closes that gap by bootstrap-resampling the 24 rows in
`judge_vs_human_comparison.csv` and computing 95% percentile CIs for all four
metrics. Bootstrap seed and resample count match
`scripts/compute_inter_judge_agreement.py` exactly so the numbers are directly
comparable to every other statistic in the §3.5 paragraph.

Inputs (read-only):
  - scripts/human_judging_analysis/output/judge_vs_human_comparison.csv
    (24 rows: observation_id, human_consensus, ai_judge_score, ...)

Outputs (written to tables/):
  - ensemble_vs_human_curated_24_cis.csv — one row with point estimates + CIs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

# bootstrap seed must match scripts/compute_inter_judge_agreement.py:158
BOOTSTRAP_SEED = 20260407
N_BOOTSTRAP_DEFAULT = 1000

ORDINAL_LEVELS = [-1.0, -0.5, 0.5, 1.0]
ORDINAL_TO_CAT = {v: i for i, v in enumerate(ORDINAL_LEVELS)}


def discretize_to_ordinal(score: float) -> float:
    """Snap a continuous score to the nearest of {-1, -0.5, 0.5, 1}.

    Mirrors scripts/compare_judge_vs_human.py:168-179.
    """
    return min(ORDINAL_LEVELS, key=lambda x: abs(x - score))


def weighted_kappa(human: np.ndarray, ai: np.ndarray) -> float:
    """Quadratic-weighted Cohen's κ with values snapped to the 4-point scale."""
    h_cats = [ORDINAL_TO_CAT[discretize_to_ordinal(x)] for x in human]
    a_cats = [ORDINAL_TO_CAT[discretize_to_ordinal(x)] for x in ai]
    return float(cohen_kappa_score(h_cats, a_cats, weights="quadratic",
                                   labels=list(range(len(ORDINAL_LEVELS)))))


def direction_match_rate(human: np.ndarray, ai: np.ndarray) -> float:
    """Fraction of items where sign(human) == sign(ai). Mirrors
    compare_judge_vs_human.py:142-143 (strict > 0 on each side)."""
    h_dir = np.where(human > 0, 1, np.where(human < 0, -1, 0))
    a_dir = np.where(ai > 0, 1, np.where(ai < 0, -1, 0))
    return float((h_dir == a_dir).mean())


def bootstrap_metric(human: np.ndarray, ai: np.ndarray, metric_fn,
                     n_bootstrap: int) -> tuple[float, float, int]:
    """Return (ci_lower, ci_upper, n_successful) for a metric under resampling.

    Resamples rows with replacement. Skips resamples where the metric returns
    NaN (can happen for κ if the resampled subset collapses to one category).
    """
    rng = np.random.default_rng(seed=BOOTSTRAP_SEED)
    n = len(human)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(human[idx], ai[idx])
        except Exception:  # noqa: BLE001
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        values.append(v)
    if not values:
        return float("nan"), float("nan"), 0
    values = np.asarray(values)
    return (float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
            len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "scripts/human_judging_analysis/output/judge_vs_human_comparison.csv",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    args = parser.parse_args()

    print(f"Loading: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"  {len(df)} rows")

    # Sanity asserts: the provenance of these 24 rows
    assert len(df) == 24, f"expected 24 rows, got {len(df)}"
    assert df["is_golden"].all(), "all 24 rows should be is_golden=True"

    human = df["human_consensus"].to_numpy(dtype=float)
    ai = df["ai_judge_score"].to_numpy(dtype=float)

    # Sanity check 1: reproduce the 23/24 direction-match headline
    direction_rate = direction_match_rate(human, ai)
    direction_matches = int(round(direction_rate * len(df)))
    print(f"\nDirection match (Rule A, sign > 0): "
          f"{direction_matches}/{len(df)} = {direction_rate:.3%}")
    assert direction_matches == 23, (
        f"Expected 23/24 match rate to reproduce; got {direction_matches}/24")
    print("  ✓ reproduces the existing 23/24 in "
          "judge_vs_human_agreement_analysis.md")

    # Sanity check 2: reproduce the published point estimates
    kappa_point = weighted_kappa(human, ai)
    spearman_point = float(spearmanr(human, ai).correlation)
    pearson_point = float(pearsonr(human, ai).statistic)
    print(f"\nPoint estimates (expected from judge_vs_human_agreement_analysis.md):")
    print(f"  Weighted Cohen's κ (quadratic): {kappa_point:.3f}  "
          f"(published: 0.905)")
    print(f"  Spearman's ρ:                   {spearman_point:.3f}  "
          f"(published: 0.847)")
    print(f"  Pearson's r:                    {pearson_point:.3f}  "
          f"(published: 0.962)")
    assert abs(kappa_point - 0.905) < 0.005, (
        f"weighted kappa {kappa_point:.4f} does not reproduce 0.905")
    assert abs(spearman_point - 0.847) < 0.005, (
        f"spearman {spearman_point:.4f} does not reproduce 0.847")
    assert abs(pearson_point - 0.962) < 0.005, (
        f"pearson {pearson_point:.4f} does not reproduce 0.962")
    print("  ✓ all three reproduce the published point estimates")

    # Bootstrap CIs
    print(f"\nBootstrapping (n={args.n_bootstrap}, seed={BOOTSTRAP_SEED}):")
    k_lo, k_hi, k_n = bootstrap_metric(human, ai, weighted_kappa,
                                       args.n_bootstrap)
    s_lo, s_hi, s_n = bootstrap_metric(
        human, ai,
        lambda h, a: float(spearmanr(h, a).correlation),
        args.n_bootstrap)
    p_lo, p_hi, p_n = bootstrap_metric(
        human, ai,
        lambda h, a: float(pearsonr(h, a).statistic),
        args.n_bootstrap)
    d_lo, d_hi, d_n = bootstrap_metric(human, ai, direction_match_rate,
                                       args.n_bootstrap)

    print(f"  Weighted κ:     {kappa_point:.3f} "
          f"[95% CI: {k_lo:.3f}, {k_hi:.3f}]  "
          f"({k_n}/{args.n_bootstrap} resamples successful)")
    print(f"  Spearman ρ:     {spearman_point:.3f} "
          f"[95% CI: {s_lo:.3f}, {s_hi:.3f}]  ({s_n}/{args.n_bootstrap})")
    print(f"  Pearson r:      {pearson_point:.3f} "
          f"[95% CI: {p_lo:.3f}, {p_hi:.3f}]  ({p_n}/{args.n_bootstrap})")
    print(f"  Direction rate: {direction_rate:.3%} "
          f"[95% CI: {d_lo:.3%}, {d_hi:.3%}]  ({d_n}/{args.n_bootstrap})")

    # Sanity check 3: point estimates inside CIs
    for name, pt, lo, hi in [
        ("κ", kappa_point, k_lo, k_hi),
        ("Spearman ρ", spearman_point, s_lo, s_hi),
        ("Pearson r", pearson_point, p_lo, p_hi),
        ("direction", direction_rate, d_lo, d_hi),
    ]:
        assert lo <= pt <= hi, (
            f"{name} point estimate {pt:.3f} outside CI [{lo:.3f}, {hi:.3f}]")
    print("  ✓ all point estimates fall inside their CIs")

    # Write output
    out_row = {
        "n": len(df),
        "weighted_kappa": kappa_point,
        "weighted_kappa_ci_lower": k_lo,
        "weighted_kappa_ci_upper": k_hi,
        "spearman_rho": spearman_point,
        "spearman_rho_ci_lower": s_lo,
        "spearman_rho_ci_upper": s_hi,
        "pearson_r": pearson_point,
        "pearson_r_ci_lower": p_lo,
        "pearson_r_ci_upper": p_hi,
        "direction_match_rate": direction_rate,
        "direction_match_rate_ci_lower": d_lo,
        "direction_match_rate_ci_upper": d_hi,
        "direction_matches": direction_matches,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.tables_dir / "ensemble_vs_human_curated_24_cis.csv"
    pd.DataFrame([out_row]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
