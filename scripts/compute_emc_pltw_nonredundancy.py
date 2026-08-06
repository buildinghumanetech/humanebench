#!/usr/bin/env python3
"""Non-redundant measurement test for enable-meaningful-choices / prioritize-long-term-wellbeing.

emc/pltw is the sole failing pair of 28 in the pairwise discriminant test,
TOST-classified equivalent.  This script asks: do the two benchmark *columns*
carry non-redundant information about *models*, even though scenarios do not
differentiate them?

The primary statistic is the cross-model correlation of the two columns under
scenario-cluster bootstrap (models fixed — inference target is this 15-model
cohort).  An attenuation-corrected Pearson r accounts for scenario-sampling
noise; shared-judge method variance inflates the correlation, so finding r < 1
despite it is conservative.

No API calls.

Inputs (read-only):
  - tables/inter_judge_raw_regenerated.csv  (or --raw-csv)
  - tables/discriminant_pooled/pairwise_interactions.csv  (or --pairwise-csv)
  - data/humane_bench.jsonl  (exclusion flags)

Outputs (written to --output-dir, default tables/):
  - emc_pltw_model_principle_scores.csv
  - emc_pltw_model_divergence.csv
  - emc_pltw_divergence_summary.csv
  - emc_pltw_pair_correlations.csv
  - emc_pltw_persona_did.csv
  - results/emc_pltw_differential_validity.md

Run from repo root:
    python scripts/compute_emc_pltw_nonredundancy.py
"""
# Paper: produces tables/emc_pltw_pair_correlations.csv and the other emc_pltw_*
# tables -- the cross-model correlation evidence on the Enable Meaningful Choices
# / Prioritize Long-term Wellbeing columns (supplement, "Cross-Model Correlation
# and Construct Redundancy").
# Paper: implements the attenuation-corrected cross-model Pearson r over the
# 15-model cohort under scenario-cluster bootstrap, joined to the Holm-corrected
# and TOST verdicts from the pairwise interaction test (main paper, "Principle
# Separability").
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_BOOTSTRAP_DEFAULT,
    PERSONAS,
    PRINCIPLES,
    bootstrap_cohort_principle_means,
    bootstrap_model_principle_grid,
    load_long_scores,
)
from humanebench.excluded import load_excluded_ids  # noqa: E402
from humanebench.tables import resolve_table  # noqa: E402

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

SHORT = {
    "respect-user-attention": "rua",
    "enable-meaningful-choices": "emc",
    "enhance-human-capabilities": "ehc",
    "protect-dignity-and-safety": "pds",
    "foster-healthy-relationships": "fhr",
    "prioritize-long-term-wellbeing": "pltw",
    "be-transparent-and-honest": "bath",
    "design-for-equity-and-inclusion": "dei",
}

FOCAL_A = "enable-meaningful-choices"
FOCAL_B = "prioritize-long-term-wellbeing"


def _fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.{nd}f}"


# ---------------------------------------------------------------------------
# Errors-in-variables correction
# ---------------------------------------------------------------------------


def eiv_corrected_pearson(
    x: np.ndarray, y: np.ndarray,
    se2_x: np.ndarray, se2_y: np.ndarray,
) -> dict:
    """Attenuation-corrected Pearson r for errors-in-variables.

    x, y: (n_models,) point estimates.
    se2_x, se2_y: (n_models,) squared SEs from scenario-cluster bootstrap.
    Errors independent between principles (disjoint scenario sets).
    """
    # Paper: the attenuation correction behind the corrected cross-model
    # correlations (supplement, "Cross-Model Correlation and Construct
    # Redundancy").
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    mean_se2_x = float(np.mean(se2_x))
    mean_se2_y = float(np.mean(se2_y))

    rel_x = 1.0 - mean_se2_x / var_x if var_x > 0 else 0.0
    rel_y = 1.0 - mean_se2_y / var_y if var_y > 0 else 0.0

    denom_x = var_x - mean_se2_x
    denom_y = var_y - mean_se2_y

    if denom_x <= 0 or denom_y <= 0:
        return {
            "r_corrected": float("nan"),
            "reliability_x": rel_x,
            "reliability_y": rel_y,
            "var_x": var_x,
            "var_y": var_y,
            "mean_se2_x": mean_se2_x,
            "mean_se2_y": mean_se2_y,
            "valid": False,
        }

    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    r_corr = cov_xy / np.sqrt(denom_x * denom_y)

    return {
        "r_corrected": r_corr,
        "reliability_x": rel_x,
        "reliability_y": rel_y,
        "var_x": var_x,
        "var_y": var_y,
        "mean_se2_x": mean_se2_x,
        "mean_se2_y": mean_se2_y,
        "valid": True,
    }


# ---------------------------------------------------------------------------
# Pair statistics
# ---------------------------------------------------------------------------


def pair_stats(
    grid, persona_idx: int, p_a_idx: int, p_b_idx: int,
) -> dict:
    """Cross-model correlation + EIV correction for one principle pair, one persona."""
    n_models = len(grid.models)
    x_point = grid.point[:, persona_idx, p_a_idx]
    y_point = grid.point[:, persona_idx, p_b_idx]

    r_raw, _ = sp_stats.pearsonr(x_point, y_point)
    rho_raw, _ = sp_stats.spearmanr(x_point, y_point)

    # Per-model SE^2 from replicate variance
    se2_x = np.var(grid.replicates[:, :, persona_idx, p_a_idx], axis=0, ddof=1)
    se2_y = np.var(grid.replicates[:, :, persona_idx, p_b_idx], axis=0, ddof=1)

    eiv = eiv_corrected_pearson(x_point, y_point, se2_x, se2_y)

    # Bootstrap CI for correlations (scenario resampling, models fixed)
    n_boot = grid.replicates.shape[0]
    r_reps = np.empty(n_boot)
    rho_reps = np.empty(n_boot)
    r_corr_reps = np.empty(n_boot)
    n_valid = 0

    mean_se2_x_point = float(np.mean(se2_x))
    mean_se2_y_point = float(np.mean(se2_y))

    for b in range(n_boot):
        xb = grid.replicates[b, :, persona_idx, p_a_idx]
        yb = grid.replicates[b, :, persona_idx, p_b_idx]
        if np.std(xb) < 1e-15 or np.std(yb) < 1e-15:
            r_reps[b] = np.nan
            rho_reps[b] = np.nan
            r_corr_reps[b] = np.nan
            continue
        r_reps[b], _ = sp_stats.pearsonr(xb, yb)
        rho_reps[b], _ = sp_stats.spearmanr(xb, yb)
        n_valid += 1

        # Corrected r per replicate: var_rep includes both signal + resample noise
        # E[var_rep] ≈ var_signal + 2·mean(se²), so subtract 2·mean(se²)
        var_xb = float(np.var(xb, ddof=1))
        var_yb = float(np.var(yb, ddof=1))
        dx = var_xb - 2 * mean_se2_x_point
        dy = var_yb - 2 * mean_se2_y_point
        if dx > 0 and dy > 0:
            cov_xy_b = float(np.cov(xb, yb, ddof=1)[0, 1])
            r_corr_reps[b] = cov_xy_b / np.sqrt(dx * dy)
        else:
            r_corr_reps[b] = np.nan

    r_ci = np.nanpercentile(r_reps, [2.5, 97.5]) if n_valid > 0 else (np.nan, np.nan)
    rho_ci = np.nanpercentile(rho_reps, [2.5, 97.5]) if n_valid > 0 else (np.nan, np.nan)

    n_corr_valid = int(np.sum(np.isfinite(r_corr_reps)))
    if n_corr_valid > n_boot * 0.5:
        r_corr_ci = np.nanpercentile(r_corr_reps, [2.5, 97.5])
        ci_method = "plugin_approx"
    else:
        r_corr_ci = (np.nan, np.nan)
        ci_method = "unstable"

    return {
        "n_models": n_models,
        "pearson_r": r_raw,
        "pearson_ci_lower": float(r_ci[0]),
        "pearson_ci_upper": float(r_ci[1]),
        "spearman_rho": rho_raw,
        "spearman_ci_lower": float(rho_ci[0]),
        "spearman_ci_upper": float(rho_ci[1]),
        "n_zero_variance_reps": n_boot - n_valid,
        **eiv,
        "r_corrected_ci_lower": float(r_corr_ci[0]),
        "r_corrected_ci_upper": float(r_corr_ci[1]),
        "ci_method": ci_method,
    }


# ---------------------------------------------------------------------------
# Divergence table
# ---------------------------------------------------------------------------


def divergence_table(grid, persona_idx: int, p_a_idx: int, p_b_idx: int) -> pd.DataFrame:
    rows = []
    for mi, model in enumerate(grid.models):
        xa = grid.point[mi, persona_idx, p_a_idx]
        xb = grid.point[mi, persona_idx, p_b_idx]
        d = xa - xb

        d_reps = grid.replicates[:, mi, persona_idx, p_a_idx] - grid.replicates[:, mi, persona_idx, p_b_idx]
        lo, hi = np.percentile(d_reps, [2.5, 97.5])
        se = float(np.std(d_reps, ddof=1))

        rows.append({
            "model": model,
            "s_emc": float(xa),
            "s_pltw": float(xb),
            "d_point": float(d),
            "d_ci_lower": float(lo),
            "d_ci_upper": float(hi),
            "d_se": se,
            "abs_d_gt_0.5": abs(d) > 0.5,
        })

    return pd.DataFrame(rows)


def divergence_summary(grid, persona_idx: int, p_a_idx: int, p_b_idx: int, persona_name: str) -> dict:
    d_models = grid.point[:, persona_idx, p_a_idx] - grid.point[:, persona_idx, p_b_idx]
    d_reps = grid.replicates[:, :, persona_idx, p_a_idx] - grid.replicates[:, :, persona_idx, p_b_idx]

    max_idx = int(np.argmax(d_models))
    min_idx = int(np.argmin(d_models))

    contrast_reps = d_reps[:, max_idx] - d_reps[:, min_idx]
    contrast_point = float(d_models[max_idx] - d_models[min_idx])
    lo, hi = np.percentile(contrast_reps, [2.5, 97.5])

    return {
        "persona": persona_name,
        "n_models": len(grid.models),
        "max_model": grid.models[max_idx],
        "max_d": float(d_models[max_idx]),
        "min_model": grid.models[min_idx],
        "min_d": float(d_models[min_idx]),
        "maxmin_contrast": contrast_point,
        "maxmin_ci_lower": float(lo),
        "maxmin_ci_upper": float(hi),
        "selection_conditional": True,
    }


# ---------------------------------------------------------------------------
# Persona DiD
# ---------------------------------------------------------------------------


def persona_did(long, models, n_bootstrap, seed) -> pd.DataFrame:
    reps_out: dict = {}
    cells, deltas = bootstrap_cohort_principle_means(
        long, models=models, personas=list(PERSONAS),
        delta_personas=(("good_persona", "baseline"), ("bad_persona", "baseline")),
        n_bootstrap=n_bootstrap, seed=seed,
        replicates_out=reps_out,
    )

    rows = []
    for contrast in ["good_persona", "bad_persona"]:
        emc_delta = reps_out.get((FOCAL_A, contrast, "baseline"))
        pltw_delta = reps_out.get((FOCAL_B, contrast, "baseline"))
        if emc_delta is None or pltw_delta is None:
            continue

        did_reps = emc_delta - pltw_delta

        emc_row = deltas[(deltas["principle"] == FOCAL_A) & (deltas["contrast_persona"] == contrast)].iloc[0]
        pltw_row = deltas[(deltas["principle"] == FOCAL_B) & (deltas["contrast_persona"] == contrast)].iloc[0]
        did_point = float(emc_row["point_estimate"]) - float(pltw_row["point_estimate"])

        lo, hi = np.percentile(did_reps, [2.5, 97.5])

        # Ceiling confound for good persona
        ceiling_flag = contrast == "good_persona"

        emc_cell = cells[(cells["principle"] == FOCAL_A) & (cells["persona"] == "baseline")].iloc[0]
        pltw_cell = cells[(cells["principle"] == FOCAL_B) & (cells["persona"] == "baseline")].iloc[0]

        rows.append({
            "contrast_persona": contrast,
            "delta_emc": float(emc_row["point_estimate"]),
            "delta_emc_ci_lower": float(emc_row["ci_lower"]),
            "delta_emc_ci_upper": float(emc_row["ci_upper"]),
            "delta_pltw": float(pltw_row["point_estimate"]),
            "delta_pltw_ci_lower": float(pltw_row["ci_lower"]),
            "delta_pltw_ci_upper": float(pltw_row["ci_upper"]),
            "did": did_point,
            "did_ci_lower": float(lo),
            "did_ci_upper": float(hi),
            "ceiling_confound_flag": ceiling_flag,
            "n_scenarios_emc": int(emc_row["n_scenarios"]),
            "n_scenarios_pltw": int(pltw_row["n_scenarios"]),
            "n_models": int(emc_row["n_models"]),
            "emc_baseline_headroom": 1.0 - float(emc_cell["point_estimate"]),
            "pltw_baseline_headroom": 1.0 - float(pltw_cell["point_estimate"]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Context: all 28 pairs
# ---------------------------------------------------------------------------


def all_pair_correlations(grid, pairwise_csv: Path) -> pd.DataFrame:
    """Correlation statistics for all 28 unordered principle pairs × 3 personas."""
    pairwise = pd.read_csv(pairwise_csv) if pairwise_csv.exists() else None

    rows = []
    for persona_idx, persona in enumerate(grid.personas):
        for i in range(len(grid.principles)):
            for j in range(i + 1, len(grid.principles)):
                p_a, p_b = grid.principles[i], grid.principles[j]
                ps = pair_stats(grid, persona_idx, i, j)

                is_focal = {p_a, p_b} == {FOCAL_A, FOCAL_B}

                # Join discriminant test results
                sig = p_holm = equiv = None
                if pairwise is not None:
                    match = pairwise[
                        ((pairwise["principle_a"] == p_a) & (pairwise["principle_b"] == p_b))
                        | ((pairwise["principle_a"] == p_b) & (pairwise["principle_b"] == p_a))
                    ]
                    if len(match) == 1:
                        sig = bool(match.iloc[0]["significant"])
                        p_holm = float(match.iloc[0]["p_holm"])
                        equiv = bool(match.iloc[0]["equivalent"]) if "equivalent" in match.columns else None

                row = {
                    "persona": persona,
                    "principle_a": p_a,
                    "principle_b": p_b,
                    **ps,
                    "is_focal_pair": is_focal,
                    "scenario_level_significant": sig,
                    "scenario_level_p_holm": p_holm,
                    "tost_equivalent": equiv,
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Rank by |r_raw| and |r_corrected| per persona
    for persona in grid.personas:
        mask = df["persona"] == persona
        sub = df.loc[mask].copy()
        sub["rank_abs_r_raw"] = sub["pearson_r"].abs().rank(ascending=False).astype(int)
        r_corr_abs = sub["r_corrected"].abs().copy()
        r_corr_abs[~sub["valid"]] = np.nan
        sub["rank_abs_r_corrected"] = r_corr_abs.rank(ascending=False, na_option="bottom").astype(int)
        df.loc[mask, "rank_abs_r_raw"] = sub["rank_abs_r_raw"].values
        df.loc[mask, "rank_abs_r_corrected"] = sub["rank_abs_r_corrected"].values

    return df


# ---------------------------------------------------------------------------
# Gate assertions
# ---------------------------------------------------------------------------


def gate_cohort_cis(
    grid, cells_path: Path,
    tol: float = 1e-9,
) -> None:
    """Assert grid-derived cohort values match published tables."""
    if not cells_path.exists():
        print("[warn] cohort CI tables not found; skipping gate.")
        return

    pub_cells = pd.read_csv(cells_path)

    mismatches = []
    for pi, principle in enumerate(grid.principles):
        for pei, persona in enumerate(grid.personas):
            cohort_point = float(grid.point[:, pei, pi].mean())
            pub_row = pub_cells[(pub_cells["principle"] == principle) & (pub_cells["persona"] == persona)]
            if pub_row.empty:
                continue
            pub_val = float(pub_row.iloc[0]["point_estimate"])
            if abs(cohort_point - pub_val) > tol:
                mismatches.append(
                    f"  {principle}/{persona}: published={pub_val:.12f}  got={cohort_point:.12f}"
                )

    if mismatches:
        raise AssertionError(
            "GATE FAILED: grid cohort means do not match cohort_principle_cis.csv:\n"
            + "\n".join(mismatches[:10])
        )
    print("  GATE: grid cohort means match cohort_principle_cis.csv ✓")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    out: Path,
    model_scores: pd.DataFrame,
    divergence_df: pd.DataFrame,
    div_summary: pd.DataFrame,
    pair_corrs: pd.DataFrame,
    did_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> None:
    L: list[str] = []
    A = L.append

    A("# Non-redundant measurement: enable-meaningful-choices vs prioritize-long-term-wellbeing\n")
    A(
        "emc/pltw is the sole failing pair of 28 in the pairwise discriminant "
        "test (TOST-classified equivalent at bound 0.5). That result stands — "
        "scenarios designed for one principle do not differentially engage the "
        "other's rubric. This analysis asks a different question: do the two "
        "benchmark *columns* carry non-redundant information about *models*?\n"
    )
    A(
        "**Inference target:** this 15-model cohort, models fixed. CIs come "
        "from scenario-cluster bootstrap — they answer 'how much would this "
        "statistic move under scenario re-draw with these 15 models fixed.' "
        "This is explicitly distinct from `interprinciple_correlation.md`, "
        "which resamples *models* (n=15, CI width ~±0.5, deliberately weak). "
        "The two analyses answer different questions and should not be compared "
        "by placing their CIs side by side.\n"
    )

    # §1: Confounded quantities
    A("## 1. Confounded quantities — reported, not argued\n")
    A("Two divergences between the columns are confounded and cannot serve as evidence of non-redundancy:\n")

    # Get baseline means
    emc_baseline = model_scores[(model_scores["principle"] == FOCAL_A) & (model_scores["persona"] == "baseline")]
    pltw_baseline = model_scores[(model_scores["principle"] == FOCAL_B) & (model_scores["persona"] == "baseline")]
    emc_base_mean = float(emc_baseline["point"].mean())
    pltw_base_mean = float(pltw_baseline["point"].mean())

    A(f"- **Baseline mean difference** ({_fmt(emc_base_mean)} vs {_fmt(pltw_base_mean)}): "
      "a column main effect confounded with rubric calibration. One construct "
      "measured through two differently-calibrated rubrics produces different means.")
    A(f"- **Good-persona divergence**: pltw baseline {_fmt(pltw_base_mean)} leaves "
      f"{1.0 - pltw_base_mean:.3f} of headroom on a scale capped at +1.0 "
      f"(vs {1.0 - emc_base_mean:.3f} for emc). Structurally, pltw *cannot* gain as "
      "much. 'At ceiling and unresponsive' is compatible with one construct behind "
      "a more lenient rubric.\n")

    # §2: Model-profile divergence (primary)
    A("## 2. Model-profile divergence (primary)\n")
    A(
        "If emc and pltw were one construct measured through any two monotone "
        "rubric calibrations, the 15 models' true orderings on the two columns "
        "would be identical, and per-model differences d_m = emc_m − pltw_m "
        "would be constant across models. Between-model contrasts are 2×2 "
        "model × principle interactions — rubric calibration and scenario-set "
        "difficulty both cancel.\n"
    )

    A("### Cross-model correlations\n")
    A("| persona | Pearson r | 95% CI | Spearman ρ | 95% CI | r corrected | 95% CI | reliability emc | reliability pltw |")
    A("| --- | ---: | :---: | ---: | :---: | ---: | :---: | ---: | ---: |")
    focal = pair_corrs[pair_corrs["is_focal_pair"]]
    for _, r in focal.iterrows():
        r_corr_str = f"{r.r_corrected:+.3f}" if r.valid else "--"
        r_corr_ci = f"[{_fmt(r.r_corrected_ci_lower)}, {_fmt(r.r_corrected_ci_upper)}]" if r.ci_method != "unstable" else "--"
        rel_x = f"{r.reliability_x:.3f}" if r.valid else "--"
        rel_y = f"{r.reliability_y:.3f}" if r.valid else "--"
        A(f"| {r.persona} | {r.pearson_r:+.3f} | [{_fmt(r.pearson_ci_lower)}, {_fmt(r.pearson_ci_upper)}] | "
          f"{r.spearman_rho:+.3f} | [{_fmt(r.spearman_ci_lower)}, {_fmt(r.spearman_ci_upper)}] | "
          f"{r_corr_str} | {r_corr_ci} | {rel_x} | {rel_y} |")
    A("")

    A(
        "**Note on replicate correlations.** Cluster-bootstrap CIs on correlations "
        "can sit below the point estimate. Each replicate's model-level means carry "
        "resampling noise that attenuates the correlation. The corrected r is the "
        "antidote, not a bug — it removes the scenario-noise component from the "
        "signal variance.\n"
    )

    # Max-min contrast
    A("### Model divergence\n")
    A("| persona | max model | min model | max−min contrast | 95% CI |")
    A("| --- | --- | --- | ---: | :---: |")
    for _, r in div_summary.iterrows():
        A(f"| {r.persona} | {r.max_model} ({_fmt(r.max_d)}) | {r.min_model} ({_fmt(r.min_d)}) | "
          f"{_fmt(r.maxmin_contrast)} | [{_fmt(r.maxmin_ci_lower)}, {_fmt(r.maxmin_ci_upper)}] |")
    A("")
    A("Selection-conditional: the max and min models were chosen from the observed data, "
      "so the CI does not account for selection. The contrast itself (a model × principle "
      "interaction) is clean.\n")

    # §3: 28-pair context
    A("## 3. Context across all 28 pairs\n")
    A(
        "Same correlation statistics for every principle pair; the question is "
        "where emc/pltw ranks. If its model-level r is comparable to pairs that "
        "passed the scenario-level discrimination test, 'scenarios don't "
        "differentiate them but models do' is directly supported.\n"
    )

    for persona in ["baseline", "bad_persona"]:
        p_focal = focal[focal["persona"] == persona]
        if p_focal.empty:
            continue
        rank_r = int(p_focal.iloc[0]["rank_abs_r_raw"])
        rank_corr = int(p_focal.iloc[0]["rank_abs_r_corrected"])
        A(f"**{persona}:** emc/pltw ranks {rank_r}/28 by |Pearson r|, "
          f"{rank_corr}/28 by |corrected r| (1 = strongest correlation).\n")

    # Top / bottom pairs table for baseline
    base_pairs = pair_corrs[pair_corrs["persona"] == "baseline"].sort_values("pearson_r", ascending=False)
    A("### All 28 pairs ranked by Pearson r (baseline)\n")
    A("| rank | pair | r | Spearman ρ | r corrected | scenario-level | focal |")
    A("| ---: | --- | ---: | ---: | ---: | :---: | :---: |")
    for rank, (_, r) in enumerate(base_pairs.iterrows(), 1):
        pair_str = f"{SHORT.get(r.principle_a, r.principle_a[:4])}/{SHORT.get(r.principle_b, r.principle_b[:4])}"
        sig_str = "pass" if r.scenario_level_significant else ("fail" if r.scenario_level_significant is not None else "--")
        focal_str = "**focal**" if r.is_focal_pair else ""
        r_corr_str = f"{r.r_corrected:+.3f}" if r.valid else "--"
        A(f"| {rank} | {pair_str} | {r.pearson_r:+.3f} | {r.spearman_rho:+.3f} | {r_corr_str} | {sig_str} | {focal_str} |")
    A("")

    # §4: Persona-response DiD
    A("## 4. Persona-response divergence\n")
    A("| contrast | Δ emc | 95% CI | Δ pltw | 95% CI | DiD | 95% CI | ceiling flag |")
    A("| --- | ---: | :---: | ---: | :---: | ---: | :---: | :---: |")
    for _, r in did_df.iterrows():
        ceil = "yes" if r.ceiling_confound_flag else "no"
        A(f"| {r.contrast_persona} | {_fmt(r.delta_emc)} | [{_fmt(r.delta_emc_ci_lower)}, {_fmt(r.delta_emc_ci_upper)}] | "
          f"{_fmt(r.delta_pltw)} | [{_fmt(r.delta_pltw_ci_lower)}, {_fmt(r.delta_pltw_ci_upper)}] | "
          f"{_fmt(r['did'])} | [{_fmt(r.did_ci_lower)}, {_fmt(r.did_ci_upper)}] | {ceil} |")
    A("")

    good_did = did_df[did_df["contrast_persona"] == "good_persona"]
    if not good_did.empty:
        gd = good_did.iloc[0]
        A(f"Good-persona DiD: {_fmt(gd['did'])} [{_fmt(gd.did_ci_lower)}, {_fmt(gd.did_ci_upper)}]. "
          f"emc headroom = {gd.emc_baseline_headroom:.3f}, pltw headroom = {gd.pltw_baseline_headroom:.3f}. "
          "Ceiling-confounded — pltw structurally cannot gain as much.\n")

    bad_did = did_df[did_df["contrast_persona"] == "bad_persona"]
    if not bad_did.empty:
        bd = bad_did.iloc[0]
        A(f"Bad-persona DiD: {_fmt(bd['did'])} [{_fmt(bd.did_ci_lower)}, {_fmt(bd.did_ci_upper)}]. "
          "Away from the ceiling — this is the clean comparison.\n")

    # §5: Summary
    A("## 5. Strongest defensible statement\n")
    A(
        "> **Superseded in part.** `results/emc_pltw_correlation_check.md` "
        "examines whether the correlation statistics in §2–§3 are informative "
        "about construct redundancy at all, and concludes they are not, in "
        "either direction: a general performance factor dominates between-model "
        "variance (PC1 = 73–97% by persona) and the corrected correlations do "
        "not track the scenario-level verdicts (the rank-1 pair by corrected r "
        "passed the interaction test; the rank-2 pair is the sole failure). "
        "Read the sentence below as a description of the numbers, not as "
        "evidence for or against redundancy; the retention argument rests on "
        "the scenario-level tests and the leave-one-principle-out analysis.\n"
    )

    base_focal = focal[focal["persona"] == "baseline"]
    bad_focal = focal[focal["persona"] == "bad_persona"]

    if not base_focal.empty:
        r_base = float(base_focal.iloc[0]["pearson_r"])
        if r_base < 0.9:
            A(
                f"The cross-model Pearson r under baseline is {r_base:+.3f} — "
                "clearly below 1.0. Models that score relatively well on emc do not "
                "necessarily score relatively well on pltw, and vice versa. The two "
                "columns carry non-redundant information about model behaviour.\n"
            )
        elif r_base < 0.99:
            A(
                f"The cross-model Pearson r under baseline is {r_base:+.3f} — "
                "high but not 1.0. There is modest evidence that the two columns "
                "carry partially non-redundant information about models, but the "
                "divergence is small.\n"
            )
        else:
            A(
                f"The cross-model Pearson r under baseline is {r_base:+.3f} — "
                "essentially 1.0. The two columns do not carry detectably non-redundant "
                "information about models in this cohort. Retention rests on the "
                "differential-engagement argument alone.\n"
            )

    A("---\n")
    A(f"Generated by `scripts/compute_emc_pltw_nonredundancy.py`. "
      f"{n_bootstrap:,} bootstrap replicates, seed {seed}.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--raw-csv", type=Path, default=None)
    ap.add_argument("--pairwise-csv", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "results" / "emc_pltw_differential_validity.md")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    raw_csv = args.raw_csv or REPO_ROOT / "tables" / "inter_judge_raw_regenerated.csv"
    raw_csv = resolve_table(raw_csv.expanduser())
    pairwise_csv = args.pairwise_csv or (
        REPO_ROOT / "tables" / "discriminant_pooled" / "pairwise_interactions.csv"
    )

    print(f"Loading {raw_csv} ...")
    long = load_long_scores(raw_csv)
    excluded = load_excluded_ids()
    if excluded:
        long = long[~long["sample_id"].isin(excluded)]
    print(f"  {len(long):,} rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding model-principle grid ...")
    grid = bootstrap_model_principle_grid(
        long, models=MODEL_ORDER, personas=list(PERSONAS),
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    print(f"  {len(grid.models)} models, {len(grid.personas)} personas, "
          f"{len(grid.principles)} principles")

    # Gate: cohort means match published
    cells_path = args.output_dir / "cohort_principle_cis.csv"
    gate_cohort_cis(grid, cells_path)

    # Model-principle scores (all 8 principles)
    score_rows = []
    for pi, principle in enumerate(grid.principles):
        for mi, model in enumerate(grid.models):
            for pei, persona in enumerate(grid.personas):
                rep = grid.replicates[:, mi, pei, pi]
                lo, hi = np.percentile(rep, [2.5, 97.5])
                score_rows.append({
                    "model": model,
                    "persona": persona,
                    "principle": principle,
                    "point": float(grid.point[mi, pei, pi]),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "se_boot": float(np.std(rep, ddof=1)),
                    "n_scenarios": int(grid.n_scenarios[pi]),
                })
    model_scores_df = pd.DataFrame(score_rows)
    model_scores_df.to_csv(args.output_dir / "emc_pltw_model_principle_scores.csv", index=False)
    print(f"wrote {args.output_dir / 'emc_pltw_model_principle_scores.csv'}")

    # Focal pair indices
    a_idx = grid.principles.index(FOCAL_A)
    b_idx = grid.principles.index(FOCAL_B)

    # Divergence tables per persona
    div_rows = []
    div_summary_rows = []
    for pei, persona in enumerate(grid.personas):
        dt = divergence_table(grid, pei, a_idx, b_idx)
        dt["persona"] = persona
        div_rows.append(dt)
        div_summary_rows.append(divergence_summary(grid, pei, a_idx, b_idx, persona))

    div_df = pd.concat(div_rows, ignore_index=True)
    div_df.to_csv(args.output_dir / "emc_pltw_model_divergence.csv", index=False)
    print(f"wrote {args.output_dir / 'emc_pltw_model_divergence.csv'}")

    div_summary_df = pd.DataFrame(div_summary_rows)
    div_summary_df.to_csv(args.output_dir / "emc_pltw_divergence_summary.csv", index=False)
    print(f"wrote {args.output_dir / 'emc_pltw_divergence_summary.csv'}")

    # All 28 pair correlations
    print("\nComputing 28-pair correlations x 3 personas ...")
    pair_corrs = all_pair_correlations(grid, pairwise_csv)
    pair_corrs.to_csv(args.output_dir / "emc_pltw_pair_correlations.csv", index=False)
    print(f"wrote {args.output_dir / 'emc_pltw_pair_correlations.csv'}")

    # Persona-response DiD
    print("\nPersona-response DiD ...")
    did_df = persona_did(long, MODEL_ORDER, args.n_bootstrap, args.seed)
    did_df.to_csv(args.output_dir / "emc_pltw_persona_did.csv", index=False)
    print(f"wrote {args.output_dir / 'emc_pltw_persona_did.csv'}")

    # Report
    write_report(
        args.report, model_scores_df, div_df, div_summary_df,
        pair_corrs, did_df, args.n_bootstrap, args.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
