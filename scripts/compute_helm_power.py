#!/usr/bin/env python3
"""Minimum detectable effect for the HELM capability x robustness correlation.

Section 4.6 claims "no significant correlation between capability and robust
humaneness" from n = 13 HELM-matched models, but prints no r and no p, and
never states what effect size that n could actually have detected. A null at
n = 13 is only informative if the study could have found a real effect of a
plausible size; otherwise "no significant correlation" is a statement about
the sample size, not about capability.

This script reports both halves:

  1. The **observed** correlation (Pearson and Spearman) between HELM aggregate
     score and Delta_bad, with CIs, recomputed from the logs rather than read
     off a stale table.
  2. The **minimum detectable effect**: the smallest |r| that a two-tailed test
     at alpha = 0.05 would reject the null for, with 80% power, at n = 13.

MDE method -- Fisher z. Under H1 the transform z = arctanh(r) is approximately
normal with SD 1/sqrt(n-3), so the smallest detectable |z| is
(z_{1-alpha/2} + z_{power}) / sqrt(n-3) and the MDE is its tanh. The
approximation is mildly conservative at small n, so the value is cross-checked
against an exact Monte-Carlo power simulation: draw bivariate-normal samples at
the candidate rho, apply the exact t-test on r, and solve for the rho whose
rejection rate hits the target power.

Inputs (read-only):
  - helm_integration/data/helm_aggregate_scores.json
  - tables/loo_model_scores.csv   (ensemble3 rows; regenerated from the logs
    by scripts/compute_loo_sensitivity.py -- NOT table1, which is stale)

Outputs (written to --output-dir, default tables/):
  - helm_power.csv    observed correlations + MDE by method
  - helm_power.md     the narrative

Run from repo root:
    python scripts/compute_helm_power.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_aaai_helm_scatter import HELM_TO_EVAL  # noqa: E402
from humanebench.bootstrap import BOOTSTRAP_SEED  # noqa: E402

ALPHA = 0.05
POWER = 0.80
N_SIM = 200_000


def fisher_z_mde(n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest |r| detectable at `power` with a two-tailed test of size `alpha`."""
    if n <= 3:
        return float("nan")
    z_a = sp.norm.ppf(1 - alpha / 2)
    z_b = sp.norm.ppf(power)
    return float(np.tanh((z_a + z_b) / np.sqrt(n - 3)))


def exact_power(rho: float, n: int, alpha: float, n_sim: int, seed: int) -> float:
    """Monte-Carlo rejection rate of the exact t-test on r at true `rho`.

    Draws (x, y) bivariate normal with correlation rho, computes the sample r,
    and applies t = r*sqrt((n-2)/(1-r^2)) against t_{n-2}. This is the test the
    reported p-value comes from, so it is the right reference for the Fisher-z
    approximation.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_sim, n))
    e = rng.standard_normal((n_sim, n))
    y = rho * x + np.sqrt(max(0.0, 1 - rho**2)) * e

    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r**2))
    crit = sp.t.ppf(1 - alpha / 2, n - 2)
    return float((np.abs(t) > crit).mean())


def exact_mde(n: int, alpha: float, power: float, n_sim: int, seed: int) -> float:
    """Bisect on rho for the value whose exact-test power equals `power`."""
    lo, hi = 0.0, 0.999
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if exact_power(mid, n, alpha, n_sim, seed) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def r_ci(r: float, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Fisher-z confidence interval for a Pearson r."""
    if n <= 3:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = sp.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def load_matched(helm_path: Path, scores_path: Path) -> pd.DataFrame:
    """Join HELM aggregate scores to regenerated Delta_bad on the matched cohort."""
    helm = json.loads(helm_path.read_text())["models"]
    helm_scores = {r["model_name"]: r["mean_score"] for r in helm}

    scores = pd.read_csv(scores_path)
    scores = scores[scores.config == "ensemble3"].set_index("model")

    rows = []
    unmatched = []
    for helm_name, eval_model in HELM_TO_EVAL.items():
        hs = helm_scores.get(helm_name)
        if hs is None or eval_model not in scores.index:
            unmatched.append((helm_name, eval_model, hs is None))
            continue
        r = scores.loc[eval_model]
        rows.append({
            "helm_model": helm_name,
            "eval_model": eval_model,
            "helm_score": float(hs),
            "s_baseline": r.s_baseline,
            "s_bad": r.s_bad_persona,
            "delta_bad": r.delta_bad,
        })
    if unmatched:
        for h, e, missing_helm in unmatched:
            why = "no HELM score" if missing_helm else "no eval row"
            print(f"[warn] unmatched: {h} -> {e} ({why})")
    return pd.DataFrame(rows).sort_values("helm_score", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--helm", type=Path,
                    default=REPO_ROOT / "helm_integration" / "data"
                    / "helm_aggregate_scores.json")
    ap.add_argument("--scores", type=Path,
                    default=REPO_ROOT / "tables" / "loo_model_scores.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_matched(args.helm, args.scores)
    n = len(df)
    print(f"Matched cohort: n = {n}")

    x = df.helm_score.to_numpy()
    out_rows = []
    for target, label in [("delta_bad", "Delta_bad"), ("s_bad", "S_bad"),
                          ("s_baseline", "S_baseline")]:
        y = df[target].to_numpy()
        pr, pp = sp.pearsonr(x, y)
        srho, sp_p = sp.spearmanr(x, y)
        lo, hi = r_ci(pr, n)
        out_rows.append({
            "outcome": label, "n": n,
            "pearson_r": pr, "pearson_p": pp,
            "pearson_ci_lower": lo, "pearson_ci_upper": hi,
            "spearman_rho": srho, "spearman_p": sp_p,
        })
        print(f"  HELM x {label:11} r = {pr:+.3f} (p = {pp:.3f}), "
              f"rho = {srho:+.3f} (p = {sp_p:.3f})")

    mde_fisher = fisher_z_mde(n)
    print(f"\nFisher-z MDE at n={n}, power={POWER}, alpha={ALPHA}: r = {mde_fisher:.4f}")
    mde_exact = exact_mde(n, ALPHA, POWER, args.n_sim, args.seed)
    print(f"Exact Monte-Carlo MDE ({args.n_sim:,} sims): r = {mde_exact:.4f}")

    obs_r = out_rows[0]["pearson_r"]
    power_at_obs = exact_power(abs(obs_r), n, ALPHA, args.n_sim, args.seed)
    print(f"Power to detect the observed |r| = {abs(obs_r):.3f}: {power_at_obs:.3f}")

    pd.DataFrame(out_rows).to_csv(args.output_dir / "helm_power.csv", index=False)

    L = ["# HELM correlation: observed effect and minimum detectable effect\n"]
    L.append(f"Matched cohort: **n = {n}** models. HELM aggregate score vs "
             "HumaneBench outcomes. Delta_bad regenerated from the logs via "
             "`compute_loo_sensitivity.py` (`ensemble3`), not read from the "
             "stale `table1_steerability_summary.csv`.\n")
    L.append("## Observed correlations\n")
    L.append("| outcome | Pearson r | 95% CI | p | Spearman rho | p |")
    L.append("| --- | ---: | :---: | ---: | ---: | ---: |")
    for r in out_rows:
        L.append(f"| {r['outcome']} | {r['pearson_r']:+.3f} | "
                 f"[{r['pearson_ci_lower']:+.3f}, {r['pearson_ci_upper']:+.3f}] | "
                 f"{r['pearson_p']:.3f} | {r['spearman_rho']:+.3f} | "
                 f"{r['spearman_p']:.3f} |")
    L.append("")
    L.append("## Minimum detectable effect\n")
    L.append(f"Two-tailed, alpha = {ALPHA}, power = {POWER}, n = {n}.\n")
    L.append("| method | MDE (|r|) |")
    L.append("| --- | ---: |")
    L.append(f"| Fisher z approximation | {mde_fisher:.3f} |")
    L.append(f"| Exact t-test, Monte-Carlo ({args.n_sim:,} sims) | {mde_exact:.3f} |")
    L.append("")
    L.append(
        f"With {n} models this design could only have detected a correlation of "
        f"about **|r| >= {mde_exact:.2f}** -- a very large effect. Its power to "
        f"detect the correlation actually observed "
        f"(|r| = {abs(obs_r):.3f}) is **{power_at_obs:.2f}**.\n"
    )
    L.append(
        "This bounds what the null can be read to mean. The result rules out a "
        "*strong* capability-robustness relationship; it does not establish the "
        "absence of a moderate one. Section 4.6's phrasing should say so "
        "explicitly rather than leaving \"no significant correlation\" to carry "
        "the weight, and should print the observed r and p, which the current "
        "text does not.\n"
    )
    (args.output_dir / "helm_power.md").write_text("\n".join(L))
    df.to_csv(args.output_dir / "helm_matched_cohort.csv", index=False)
    print(f"\nWrote {args.output_dir / 'helm_power.md'}")


if __name__ == "__main__":
    main()
