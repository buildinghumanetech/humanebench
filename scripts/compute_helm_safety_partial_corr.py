#!/usr/bin/env python3
"""
Correlate HELM Safety against HumaneBench adversarial robustness Δ_bad.

Δ_bad := bad-persona HumaneScore − baseline HumaneScore
       (matches scripts/create_aaai_helm_scatter.py; more negative = larger
        adversarial degradation; 0 = no degradation)

The "necessary but not sufficient" hypothesis predicts:
  * Spearman ρ(Safety, Δ_bad) > 0  (more safety → Δ_bad closer to zero)
  * ρ_partial (controlling for HELM Capabilities) closer to zero than ρ

Inputs (read-only):
  - helm_integration/data/helm_safety_aggregate_scores.json
  - helm_integration/data/helm_aggregate_scores.json
  - helm_integration/helm_safety_mapping.json
  - tables/table1_steerability_summary.csv

Outputs (written to tables/):
  - helm_safety_robustness_merged.csv
  - helm_safety_robustness_stats.csv
  - helm_safety_robustness_stats.md

Usage:
    python scripts/compute_helm_safety_partial_corr.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent

SAFETY_AGG = REPO_ROOT / "helm_integration" / "data" / "helm_safety_aggregate_scores.json"
CAPABILITY_AGG = REPO_ROOT / "helm_integration" / "data" / "helm_aggregate_scores.json"
MAPPING_FILE = REPO_ROOT / "helm_integration" / "helm_safety_mapping.json"
SUMMARY_CSV = REPO_ROOT / "tables" / "table1_steerability_summary.csv"

MERGED_CSV = REPO_ROOT / "tables" / "helm_safety_robustness_merged.csv"
STATS_CSV = REPO_ROOT / "tables" / "helm_safety_robustness_stats.csv"
STATS_MD = REPO_ROOT / "tables" / "helm_safety_robustness_stats.md"

N_BOOT = 10_000
N_PERM = 10_000
SEED = 0


def load_mapping() -> pd.DataFrame:
    with open(MAPPING_FILE) as f:
        rows = json.load(f)["mappings"]
    df = pd.DataFrame(rows)
    df = df.rename(columns={"your_model_name": "eval_model",
                            "helm_model_name": "helm_safety_model_name"})
    return df[["eval_model", "helm_safety_model_name", "notes"]]


def load_safety() -> pd.DataFrame:
    with open(SAFETY_AGG) as f:
        rows = json.load(f)["models"]
    df = pd.DataFrame(rows)
    return df.rename(columns={"model_name": "helm_safety_model_name",
                              "mean_score": "safety_score"})[
        ["helm_safety_model_name", "safety_score"]
    ]


def load_capability() -> pd.DataFrame:
    with open(CAPABILITY_AGG) as f:
        rows = json.load(f)["models"]
    df = pd.DataFrame(rows)
    return df.rename(columns={"model_name": "helm_capability_model_name",
                              "mean_score": "capability_score"})[
        ["helm_capability_model_name", "capability_score"]
    ]


def load_humanescores() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df = df.rename(columns={
        "Model": "eval_model",
        "Baseline HumaneScore": "baseline_humanescore",
        "Bad Persona HumaneScore": "bad_humanescore",
    })
    df["delta_bad"] = df["bad_humanescore"] - df["baseline_humanescore"]
    return df[["eval_model", "baseline_humanescore", "bad_humanescore", "delta_bad"]]


FAMILY_PREFIXES = {
    "gpt": "OpenAI", "claude": "Anthropic", "gemini": "Google",
    "llama": "Meta", "grok": "xAI", "deepseek": "DeepSeek",
}


def family_for(eval_model: str) -> str:
    low = eval_model.lower()
    for prefix, fam in FAMILY_PREFIXES.items():
        if prefix in low:
            return fam
    return "Other"


def build_merged() -> pd.DataFrame:
    mapping = load_mapping()
    safety = load_safety()
    capability = load_capability()
    hs = load_humanescores()

    # Heuristic capability-mapping reuse: the safety mapping shares the same
    # eval_model slugs as the capability mapping in create_aaai_helm_scatter.py.
    # Build a capability-name lookup using the same naming if present in HELM.
    cap_mapping_path = REPO_ROOT / "helm_integration" / "model_mapping.json"
    cap_name_by_eval: dict[str, str] = {}
    if cap_mapping_path.exists():
        with open(cap_mapping_path) as f:
            for row in json.load(f).get("mappings", []):
                slug = row.get("your_model_name", "")
                helm = row.get("helm_model_name", "")
                # capabilities mapping uses display-style slugs; the AAAI scatter
                # bridges them in HELM_TO_EVAL. We reuse that bridge directly:
                cap_name_by_eval.setdefault(slug, helm)
    # Direct bridge from create_aaai_helm_scatter.py:
    AAAI_HELM_TO_EVAL = {
        "OpenAI GPT 5 (2025-08-07)": "gpt-5",
        "OpenAI GPT 5.1": "gpt-5.1",
        "OpenAI GPT 4 1 (2025-04-14)": "gpt-4.1",
        "OpenAI GPT 4O (2024-11-20)": "gpt-4o-2024-11-20",
        "Anthropic CLAUDE 4 5 SONNET (2025-09-29)": "claude-sonnet-4.5",
        "Anthropic CLAUDE 4 SONNET (2025-05-14)": "claude-sonnet-4",
        "Google Gemini 3 Pro (Preview)": "gemini-3-pro-preview",
        "Google Gemini 2 5 Pro": "gemini-2.5-pro",
        "Google Gemini 2.5 Flash": "gemini-2.5-flash",
        "Google Gemini 2.0 Flash 001": "gemini-2.0-flash-001",
        "Meta Llama 4 Maverick (17Bx128E) Instruct FP8": "llama-4-maverick",
        "Meta Llama 3.1 405B Instruct Turbo": "llama-3.1-405b-instruct",
        "Grok 4 (0709)": "grok-4",
    }
    cap_name_by_eval = {v: k for k, v in AAAI_HELM_TO_EVAL.items()}

    mapping = mapping[mapping["helm_safety_model_name"].astype(bool)].copy()
    df = mapping.merge(safety, on="helm_safety_model_name", how="left")
    df["helm_capability_model_name"] = df["eval_model"].map(cap_name_by_eval)
    df = df.merge(capability, on="helm_capability_model_name", how="left")
    df = df.merge(hs, on="eval_model", how="left")
    df["family"] = df["eval_model"].apply(family_for)

    # Diagnostics.
    missing_safety = df[df["safety_score"].isna()]["eval_model"].tolist()
    missing_cap = df[df["capability_score"].isna()]["eval_model"].tolist()
    missing_hs = df[df["delta_bad"].isna()]["eval_model"].tolist()
    if missing_safety:
        print(f"WARNING: safety_score missing for: {missing_safety}")
        print("  (mapping has a name but the safety aggregate JSON did not "
              "match it — check helm_safety_mapping.json against "
              "helm_integration/data/helm_safety_raw_data.json)")
    if missing_cap:
        print(f"NOTE: capability_score missing for: {missing_cap}")
    if missing_hs:
        print(f"NOTE: HumaneBench scores missing for: {missing_hs}")

    return df


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman ρ of (x, y) controlling for z, via residual ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    # Regress rx ~ rz, ry ~ rz; correlate residuals.
    A = np.column_stack([np.ones_like(rz, dtype=float), rz.astype(float)])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    res_x = rx - A @ bx
    res_y = ry - A @ by
    if np.std(res_x) == 0 or np.std(res_y) == 0:
        return float("nan")
    return float(pearsonr(res_x, res_y).statistic)


def bootstrap_ci(stat_fn, *vectors, n_boot: int = N_BOOT, seed: int = SEED,
                 ) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(vectors[0])
    estimates = np.empty(n_boot)
    idx_pool = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        sampled = [v[idx] for v in vectors]
        try:
            estimates[i] = stat_fn(*sampled)
        except Exception:
            estimates[i] = np.nan
    estimates = estimates[~np.isnan(estimates)]
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def permutation_p(stat_fn, x: np.ndarray, y: np.ndarray, *extras,
                  n_perm: int = N_PERM, seed: int = SEED) -> Tuple[float, float]:
    """Two-sided permutation p; shuffles y against x (and any extras stay fixed)."""
    rng = np.random.default_rng(seed)
    observed = stat_fn(x, y, *extras)
    null = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        try:
            null[i] = stat_fn(x, y_perm, *extras)
        except Exception:
            null[i] = np.nan
    null = null[~np.isnan(null)]
    p = float(np.mean(np.abs(null) >= abs(observed)))
    return observed, p


def loo_range(stat_fn, *vectors) -> Tuple[float, float, float]:
    """Min / median / max of stat_fn over single-model leave-one-out folds."""
    n = len(vectors[0])
    vals = []
    for i in range(n):
        kept = [np.delete(v, i) for v in vectors]
        try:
            vals.append(stat_fn(*kept))
        except Exception:
            continue
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(vals.min()), float(np.median(vals)), float(vals.max())


def spearman_stat(x, y):
    rho, _ = spearmanr(x, y)
    return float(rho)


def pearson_stat(x, y):
    r, _ = pearsonr(x, y)
    return float(r)


def main() -> int:
    merged = build_merged()
    merged.to_csv(MERGED_CSV, index=False)
    print(f"✓ Wrote {MERGED_CSV.relative_to(REPO_ROOT)}")

    cohort = merged.dropna(subset=["safety_score", "delta_bad"]).copy()
    n = len(cohort)
    if n < 3:
        print(f"✗ Cohort too small (n={n}) for correlation. Check merged CSV.")
        return 1

    x = cohort["safety_score"].to_numpy(dtype=float)
    y = cohort["delta_bad"].to_numpy(dtype=float)

    # Primary + secondary.
    rho_obs, p_rho = permutation_p(spearman_stat, x, y)
    r_obs, p_r = permutation_p(pearson_stat, x, y)
    rho_lo, rho_hi = bootstrap_ci(spearman_stat, x, y)
    r_lo, r_hi = bootstrap_ci(pearson_stat, x, y)

    rho_min, rho_med, rho_max = loo_range(spearman_stat, x, y)
    r_min, r_med, r_max = loo_range(pearson_stat, x, y)

    # Partial (subset with capability).
    partial_cohort = cohort.dropna(subset=["capability_score"]).copy()
    n_partial = len(partial_cohort)
    has_partial = n_partial >= 4
    rows = [
        ("spearman_rho", rho_obs, rho_lo, rho_hi, p_rho, n),
        ("pearson_r",    r_obs,   r_lo,   r_hi,   p_r,   n),
        ("spearman_rho_loo_min",    rho_min, np.nan, np.nan, np.nan, n),
        ("spearman_rho_loo_median", rho_med, np.nan, np.nan, np.nan, n),
        ("spearman_rho_loo_max",    rho_max, np.nan, np.nan, np.nan, n),
        ("pearson_r_loo_min",    r_min, np.nan, np.nan, np.nan, n),
        ("pearson_r_loo_median", r_med, np.nan, np.nan, np.nan, n),
        ("pearson_r_loo_max",    r_max, np.nan, np.nan, np.nan, n),
    ]

    partial_summary = None
    if has_partial:
        xp = partial_cohort["safety_score"].to_numpy(dtype=float)
        yp = partial_cohort["delta_bad"].to_numpy(dtype=float)
        zp = partial_cohort["capability_score"].to_numpy(dtype=float)
        partial_obs, p_partial = permutation_p(partial_spearman, xp, yp, zp)
        partial_lo, partial_hi = bootstrap_ci(partial_spearman, xp, yp, zp)
        partial_min, partial_med, partial_max = loo_range(
            partial_spearman, xp, yp, zp
        )
        partial_summary = (partial_obs, partial_lo, partial_hi, p_partial,
                           partial_min, partial_med, partial_max, n_partial)
        rows += [
            ("spearman_rho_partial",            partial_obs, partial_lo, partial_hi, p_partial, n_partial),
            ("spearman_rho_partial_loo_min",    partial_min, np.nan, np.nan, np.nan, n_partial),
            ("spearman_rho_partial_loo_median", partial_med, np.nan, np.nan, np.nan, n_partial),
            ("spearman_rho_partial_loo_max",    partial_max, np.nan, np.nan, np.nan, n_partial),
        ]
    else:
        print(f"NOTE: skipping partial correlation (only {n_partial} models "
              "have both safety + capability + drop).")

    stats_df = pd.DataFrame(rows, columns=[
        "name", "point_estimate", "ci_lower", "ci_upper", "p_value_perm", "n"
    ])
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"✓ Wrote {STATS_CSV.relative_to(REPO_ROOT)}")

    # Markdown report.
    lines = [
        "# HELM Safety × HumaneBench adversarial robustness (Δ_bad)",
        "",
        f"Cohort: n = {n} (DeepSeek V3.1 and Claude Opus 4.1 expected-absent "
        f"per the HELM Safety release).",
        "",
        "Δ_bad = bad-persona HumaneScore − baseline HumaneScore "
        "(more negative = larger adversarial degradation; matches "
        "scripts/create_aaai_helm_scatter.py). The hypothesis predicts a "
        "**positive** ρ.",
        "",
        "## Primary",
        "",
        f"| statistic | estimate | 95% CI (bootstrap, n={N_BOOT:,}) | p (perm, n={N_PERM:,}) |",
        "|-----------|----------|----------------------------------|------------------------|",
        f"| Spearman ρ | {rho_obs:+.3f} | [{rho_lo:+.3f}, {rho_hi:+.3f}] | {p_rho:.4f} |",
        f"| Pearson r  | {r_obs:+.3f}   | [{r_lo:+.3f}, {r_hi:+.3f}]     | {p_r:.4f} |",
        "",
        "## Leave-one-out robustness (Spearman ρ)",
        "",
        f"min / median / max across {n} LOO folds: "
        f"{rho_min:+.3f} / {rho_med:+.3f} / {rho_max:+.3f}",
        "",
    ]
    if partial_summary is not None:
        (partial_obs, partial_lo, partial_hi, p_partial,
         partial_min, partial_med, partial_max, n_partial) = partial_summary
        lines += [
            "## Partial Spearman ρ(Safety, Δ_bad | Capability)",
            "",
            f"n = {n_partial} (subset with HELM Capability score).",
            "",
            f"| statistic | estimate | 95% CI | p (perm) |",
            "|-----------|----------|--------|----------|",
            f"| ρ_partial | {partial_obs:+.3f} | "
            f"[{partial_lo:+.3f}, {partial_hi:+.3f}] | {p_partial:.4f} |",
            "",
            f"LOO range: {partial_min:+.3f} / {partial_med:+.3f} / "
            f"{partial_max:+.3f}",
            "",
        ]
        weakened = abs(partial_obs) < abs(rho_obs)
        marker = "✓" if weakened else "✗"
        lines.append(
            f"{marker} |ρ_partial| {'<' if weakened else '≥'} |ρ| — "
            f"capability {'absorbs some of the signal (consistent with ' if weakened else 'does not weaken (inconsistent with '}"
            f"'necessary but not sufficient').")
        lines.append("")
    caveats = cohort[cohort["notes"].astype(bool) &
                     ~cohort["notes"].str.contains("expected-absent", na=False)]
    if not caveats.empty:
        lines.append("## Caveats")
        lines.append("")
        for _, row in caveats.iterrows():
            lines.append(f"- **{row['eval_model']}** — {row['notes']}")
        lines.append("")

    lines += [
        "## Cohort",
        "",
        "See `tables/helm_safety_robustness_merged.csv` for per-model values.",
        "",
    ]
    STATS_MD.write_text("\n".join(lines))
    print(f"✓ Wrote {STATS_MD.relative_to(REPO_ROOT)}")

    print()
    print(f"Spearman ρ(Safety, Δ_bad) = {rho_obs:+.3f}  "
          f"[{rho_lo:+.3f}, {rho_hi:+.3f}]  p_perm={p_rho:.4f}  n={n}")
    if partial_summary is not None:
        print(f"Partial ρ | Capability  = {partial_summary[0]:+.3f}  "
              f"[{partial_summary[1]:+.3f}, {partial_summary[2]:+.3f}]  "
              f"p_perm={partial_summary[3]:.4f}  n={partial_summary[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
