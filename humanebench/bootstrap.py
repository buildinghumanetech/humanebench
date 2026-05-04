"""Bootstrap confidence intervals for HumaneBench headline scores.

Single source of truth for the resampling design used in the paper:

- Resampling unit: scenario_id (`sample_id`), stratified by principle within
  each (model, persona) cell. Within a model, the *same resampled scenario
  ids* are used across personas, so persona-delta CIs preserve the within-
  prompt pairing.
- Per-sample score: mean severity across the 3-judge ensemble.
- Per-cell HumaneScore: mean of the 8 principle means (matches scorer).
- Replicates: 1000. Seed: 20260407. CI: 2.5 / 97.5 percentile.

These conventions are deliberately identical to
`scripts/compute_inter_judge_agreement.py` and
`scripts/compute_ensemble_vs_human_cis.py` so every CI in the paper shares
one bootstrap design.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 20260407
N_BOOTSTRAP_DEFAULT = 1000
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5

PRINCIPLES: tuple[str, ...] = (
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
)
HUMANESCORE_KEY = "HumaneScore"
PERSONAS: tuple[str, ...] = ("baseline", "good_persona", "bad_persona")


def load_long_scores(raw_csv: Path | str) -> pd.DataFrame:
    """Read `tables/inter_judge_raw.csv` and collapse 3 judge rows per sample.

    The raw CSV is already exclusion-filtered upstream by
    `scripts/compute_inter_judge_agreement.py` (it loads
    `humanebench.excluded.load_excluded_ids` before building the long table).
    Returns one row per (persona, model, sample_id) with the ensemble-mean
    severity. Samples where any judge failed are dropped.
    """
    df = pd.read_csv(raw_csv)
    needed = {"persona", "model", "principle", "sample_id", "judge_name", "severity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"raw_csv missing required columns: {sorted(missing)}")

    grouped = (
        df.groupby(["persona", "model", "principle", "sample_id"], as_index=False)
        .agg(score=("severity", "mean"), n_judges=("severity", "size"))
    )
    # Strict-ensemble: drop any sample without the full 3-judge complement.
    grouped = grouped[grouped["n_judges"] == 3].drop(columns=["n_judges"])
    grouped = grouped.dropna(subset=["score"]).reset_index(drop=True)
    return grouped


# ---------------------------------------------------------------------------
# Internal: build numpy lookups for fast resampling
# ---------------------------------------------------------------------------


@dataclass
class _PrincipleScores:
    """Scores for one (model, persona, principle) cell, ordered by sample_id."""

    sample_ids: np.ndarray  # shape (n,)
    scores: np.ndarray  # shape (n,)


@dataclass
class _ModelPersonaCell:
    by_principle: dict[str, _PrincipleScores]


def _build_cells(
    long: pd.DataFrame,
) -> dict[tuple[str, str], _ModelPersonaCell]:
    """Group long-format scores into (model, persona) → principle → arrays."""
    cells: dict[tuple[str, str], _ModelPersonaCell] = {}
    for (model, persona), sub in long.groupby(["model", "persona"]):
        by_p: dict[str, _PrincipleScores] = {}
        for principle, p_sub in sub.groupby("principle"):
            p_sub = p_sub.sort_values("sample_id")
            by_p[principle] = _PrincipleScores(
                sample_ids=p_sub["sample_id"].to_numpy(),
                scores=p_sub["score"].to_numpy(dtype=float),
            )
        cells[(model, persona)] = _ModelPersonaCell(by_principle=by_p)
    return cells


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    return (
        float(np.percentile(samples, CI_LOW_PCT)),
        float(np.percentile(samples, CI_HIGH_PCT)),
    )


# ---------------------------------------------------------------------------
# Public API: marginal per-cell CIs
# ---------------------------------------------------------------------------


def bootstrap_cell_scores(
    long: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap per-(model, persona) marginal CIs.

    For each (model, persona) cell, draw `n_bootstrap` resamples stratified
    by principle (sample with replacement to the same per-principle n), then
    compute per-principle means and HumaneScore = mean of the 8 principle
    means.

    Returns long-format rows:
        (model, persona, principle, point_estimate, ci_lower, ci_upper, n_eff)
    where principle ∈ PRINCIPLES ∪ {"HumaneScore"}.
    """
    cells = _build_cells(long)
    rows: list[dict] = []

    for (model, persona), cell in cells.items():
        principles_present = [p for p in PRINCIPLES if p in cell.by_principle]
        if not principles_present:
            continue

        # Pre-allocate replicate matrix: rows = principles, cols = replicates.
        rng = np.random.default_rng(seed)
        rep_matrix = np.empty((len(principles_present), n_bootstrap), dtype=float)
        n_per_principle: dict[str, int] = {}

        for i, principle in enumerate(principles_present):
            arr = cell.by_principle[principle].scores
            n = arr.shape[0]
            n_per_principle[principle] = n
            if n == 0:
                rep_matrix[i, :] = np.nan
                continue
            # Vectorized: sample (n_bootstrap, n) indices, take row means.
            idx = rng.integers(0, n, size=(n_bootstrap, n))
            rep_matrix[i, :] = arr[idx].mean(axis=1)

        # Per-principle CIs (point estimate = mean of original cell, no resample)
        for i, principle in enumerate(principles_present):
            arr = cell.by_principle[principle].scores
            point = float(arr.mean()) if arr.size else float("nan")
            lo, hi = _percentile_ci(rep_matrix[i])
            rows.append({
                "model": model,
                "persona": persona,
                "principle": principle,
                "point_estimate": point,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_eff": n_per_principle[principle],
            })

        # HumaneScore: mean across the 8 principle means per replicate.
        humane_reps = rep_matrix.mean(axis=0)
        humane_point = float(
            np.mean([cell.by_principle[p].scores.mean() for p in principles_present])
        )
        lo, hi = _percentile_ci(humane_reps)
        rows.append({
            "model": model,
            "persona": persona,
            "principle": HUMANESCORE_KEY,
            "point_estimate": humane_point,
            "ci_lower": lo,
            "ci_upper": hi,
            "n_eff": int(sum(n_per_principle.values())),
        })

    return pd.DataFrame(rows).sort_values(
        ["model", "persona", "principle"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API: paired persona-delta CIs
# ---------------------------------------------------------------------------


def bootstrap_persona_deltas(
    long: pd.DataFrame,
    baseline_persona: str = "baseline",
    contrast_personas: Iterable[str] = ("good_persona", "bad_persona"),
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap paired persona deltas (`contrast - baseline`).

    For each model, restricts to scenarios present in *all* of the relevant
    personas, then resamples scenarios stratified by principle. Within each
    replicate the same resampled `sample_id`s are looked up under every
    persona, preserving the paired structure.

    Returns long-format rows:
        (model, contrast_persona, principle, point_estimate, ci_lower,
         ci_upper, n_eff)
    where principle ∈ PRINCIPLES ∪ {"HumaneScore"}.
    """
    contrast_personas = list(contrast_personas)
    rows: list[dict] = []

    for model, sub in long.groupby("model"):
        # Pivot to (persona, principle, sample_id) -> score
        wide = sub.pivot_table(
            index=["principle", "sample_id"],
            columns="persona",
            values="score",
            aggfunc="first",
        )
        required = [baseline_persona] + contrast_personas
        wide = wide.dropna(subset=required)  # paired intersection
        if wide.empty:
            continue
        wide = wide.reset_index()

        # Build per-principle index map for stratified resampling.
        principle_groups: dict[str, np.ndarray] = {}
        for principle, p_sub in wide.groupby("principle"):
            principle_groups[principle] = p_sub.index.to_numpy()
        principles_present = [p for p in PRINCIPLES if p in principle_groups]
        if not principles_present:
            continue

        # Pre-extract per-persona score columns as numpy arrays for speed.
        persona_arrays = {p: wide[p].to_numpy(dtype=float) for p in required}

        rng = np.random.default_rng(seed)

        # Stratified resample of row-indices, shared across personas.
        # Shape: (n_bootstrap, total_n_paired).
        idx_chunks_per_rep: list[np.ndarray] = []
        principle_slice_bounds: list[tuple[str, int, int]] = []
        cursor = 0
        # We'll generate one big idx matrix per principle, then concatenate
        # along axis=1 so each replicate's indices are contiguous in column
        # order (principle-major) — that's what we'll average over.
        per_principle_idx: list[np.ndarray] = []
        for principle in principles_present:
            row_pool = principle_groups[principle]
            n = row_pool.shape[0]
            picks = rng.integers(0, n, size=(n_bootstrap, n))
            per_principle_idx.append(row_pool[picks])  # (n_bootstrap, n)
            principle_slice_bounds.append((principle, cursor, cursor + n))
            cursor += n
        full_idx = np.concatenate(per_principle_idx, axis=1)  # (n_bootstrap, total)

        # For each persona, replicate × column matrix of looked-up scores.
        persona_rep_scores = {
            p: persona_arrays[p][full_idx] for p in required
        }

        for contrast in contrast_personas:
            diff = persona_rep_scores[contrast] - persona_rep_scores[baseline_persona]
            # Per-principle replicate means (n_bootstrap,) for each principle.
            principle_rep_means = []
            for principle, lo_c, hi_c in principle_slice_bounds:
                rep_means = diff[:, lo_c:hi_c].mean(axis=1)
                principle_rep_means.append(rep_means)
                # Point estimate for this principle's delta.
                base_mean = persona_arrays[baseline_persona][principle_groups[principle]].mean()
                contrast_mean = persona_arrays[contrast][principle_groups[principle]].mean()
                lo, hi = _percentile_ci(rep_means)
                rows.append({
                    "model": model,
                    "contrast_persona": contrast,
                    "principle": principle,
                    "point_estimate": float(contrast_mean - base_mean),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_eff": int(hi_c - lo_c),
                })

            # HumaneScore delta: mean of the 8 per-principle replicate-means.
            stacked = np.stack(principle_rep_means, axis=0)  # (n_principles, n_boot)
            humane_reps = stacked.mean(axis=0)
            base_humane = float(np.mean([
                persona_arrays[baseline_persona][principle_groups[p]].mean()
                for p in principles_present
            ]))
            contrast_humane = float(np.mean([
                persona_arrays[contrast][principle_groups[p]].mean()
                for p in principles_present
            ]))
            lo, hi = _percentile_ci(humane_reps)
            rows.append({
                "model": model,
                "contrast_persona": contrast,
                "principle": HUMANESCORE_KEY,
                "point_estimate": contrast_humane - base_humane,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_eff": int(full_idx.shape[1]),
            })

    return pd.DataFrame(rows).sort_values(
        ["model", "contrast_persona", "principle"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: binarized variant for the robustness-gap script
# ---------------------------------------------------------------------------


def binarize_long(long: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `long` with `score` binarized as `1.0` if `score >= 0`.

    Matches the threshold used in `scripts/compute_binarized_robustness_gap.py`
    (>=0). The ensemble mean of {-1, -0.5, 0.5, 1.0} can equal 0 when judges
    split, which we treat as the "acceptable" side per the existing script's
    convention.
    """
    out = long.copy()
    out["score"] = (out["score"] >= 0).astype(float)
    return out
