"""Unit tests for `humanebench.bootstrap`.

Verifies:
1. Coverage: synthetic data with known mean, CI covers truth ~95% over many runs.
2. Pairing tightens delta CIs vs. an unpaired naive computation when persona
   arms share a positive within-prompt correlation.
3. Seed reproducibility: same seed → same CI bounds bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from humanebench.bootstrap import (
    BOOTSTRAP_SEED,
    HUMANESCORE_KEY,
    PRINCIPLES,
    bootstrap_cell_scores,
    bootstrap_cohort_principle_means,
    bootstrap_persona_deltas,
)


def _synth_long(
    rng: np.random.Generator,
    *,
    model: str = "m1",
    personas: list[str] | None = None,
    persona_means: dict[str, float] | None = None,
    n_per_principle: int = 100,
    pair_correlation: float = 0.0,
) -> pd.DataFrame:
    """Build a synthetic long-format scores frame.

    Each scenario draws a per-prompt latent ε; persona scores are
    `persona_mean + pair_correlation * ε + (1 - pair_correlation) * noise`.
    With pair_correlation=0 the personas are independent.
    """
    personas = personas or ["baseline", "bad_persona"]
    persona_means = persona_means or {p: 0.0 for p in personas}

    rows = []
    for principle in PRINCIPLES:
        for k in range(n_per_principle):
            sample_id = f"{principle}-{k:03d}"
            eps = rng.normal()
            for persona in personas:
                noise = rng.normal()
                score = (
                    persona_means[persona]
                    + pair_correlation * eps
                    + (1.0 - pair_correlation) * noise
                )
                rows.append({
                    "persona": persona,
                    "model": model,
                    "principle": principle,
                    "sample_id": sample_id,
                    "score": score,
                })
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_humanescore_ci_covers_truth():
    """Across many simulated datasets, the CI should contain the truth ~95% of the time."""
    rng = np.random.default_rng(0)
    truth = 0.5
    n_runs = 80
    covered = 0
    for run in range(n_runs):
        df = _synth_long(
            rng,
            persona_means={"baseline": truth},
            personas=["baseline"],
            n_per_principle=80,
        )
        cis = bootstrap_cell_scores(df, n_bootstrap=400, seed=BOOTSTRAP_SEED + run)
        hs = cis[(cis["principle"] == HUMANESCORE_KEY)
                 & (cis["persona"] == "baseline")].iloc[0]
        if hs["ci_lower"] <= truth <= hs["ci_upper"]:
            covered += 1
    rate = covered / n_runs
    # 95% nominal coverage; allow slack for 80 runs and synthetic tails.
    assert 0.85 <= rate <= 1.0, f"coverage {rate} outside expected band"


@pytest.mark.unit
def test_pairing_tightens_delta_ci():
    """With positive within-prompt correlation, paired delta CI should be tighter than unpaired."""
    rng = np.random.default_rng(42)
    df = _synth_long(
        rng,
        personas=["baseline", "bad_persona"],
        persona_means={"baseline": 0.5, "bad_persona": 0.2},
        n_per_principle=80,
        pair_correlation=0.7,
    )

    # Paired delta via the production routine (uses shared resampled scenario_ids).
    paired = bootstrap_persona_deltas(
        df,
        baseline_persona="baseline",
        contrast_personas=["bad_persona"],
        n_bootstrap=600,
    )
    paired_humane = paired[(paired["principle"] == HUMANESCORE_KEY)].iloc[0]
    paired_width = paired_humane["ci_upper"] - paired_humane["ci_lower"]

    # Unpaired baseline: bootstrap each arm independently, take difference of
    # marginal HumaneScore replicate distributions.
    rng_a = np.random.default_rng(BOOTSTRAP_SEED)
    rng_b = np.random.default_rng(BOOTSTRAP_SEED + 1)
    n_boot = 600

    def _marginal_humane(persona: str, rng_: np.random.Generator) -> np.ndarray:
        sub = df[df["persona"] == persona]
        principle_groups = {
            p: sub[sub["principle"] == p]["score"].to_numpy()
            for p in PRINCIPLES
        }
        per_principle_reps = []
        for p in PRINCIPLES:
            arr = principle_groups[p]
            idx = rng_.integers(0, arr.size, size=(n_boot, arr.size))
            per_principle_reps.append(arr[idx].mean(axis=1))
        return np.stack(per_principle_reps, axis=0).mean(axis=0)

    base_reps = _marginal_humane("baseline", rng_a)
    bad_reps = _marginal_humane("bad_persona", rng_b)
    unpaired_diff = bad_reps - base_reps
    unpaired_width = float(
        np.percentile(unpaired_diff, 97.5) - np.percentile(unpaired_diff, 2.5)
    )

    # Paired must be meaningfully tighter when correlation is high.
    assert paired_width < unpaired_width * 0.85, (
        f"paired width {paired_width:.4f} not tighter than unpaired {unpaired_width:.4f}"
    )


@pytest.mark.unit
def test_seed_reproducibility():
    """Identical seeds must give identical CI bounds."""
    rng = np.random.default_rng(123)
    df = _synth_long(rng, n_per_principle=40)
    a = bootstrap_cell_scores(df, n_bootstrap=200, seed=BOOTSTRAP_SEED)
    b = bootstrap_cell_scores(df, n_bootstrap=200, seed=BOOTSTRAP_SEED)
    pd.testing.assert_frame_equal(a, b)

    da = bootstrap_persona_deltas(
        df, contrast_personas=["bad_persona"], n_bootstrap=200, seed=BOOTSTRAP_SEED,
    )
    db = bootstrap_persona_deltas(
        df, contrast_personas=["bad_persona"], n_bootstrap=200, seed=BOOTSTRAP_SEED,
    )
    pd.testing.assert_frame_equal(da, db)


def _multi_model_long(
    rng: np.random.Generator,
    *,
    models: list[str],
    personas: list[str],
    persona_means: dict[str, float],
    n_per_principle: int = 40,
    pair_correlation: float = 0.0,
) -> pd.DataFrame:
    """Synthetic long table spanning multiple models, paired across personas.

    Each (principle, sample_id) has one latent ε shared across models and
    personas; each (model, persona) draws independent noise on top. With
    pair_correlation > 0 the same scenarios are easy / hard across models too.
    """
    rows = []
    for principle in PRINCIPLES:
        for k in range(n_per_principle):
            sample_id = f"{principle}-{k:03d}"
            eps = rng.normal()
            for model in models:
                for persona in personas:
                    noise = rng.normal()
                    score = (
                        persona_means[persona]
                        + pair_correlation * eps
                        + (1.0 - pair_correlation) * noise
                    )
                    rows.append({
                        "persona": persona,
                        "model": model,
                        "principle": principle,
                        "sample_id": sample_id,
                        "score": score,
                    })
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_cohort_point_estimate_matches_mean_of_model_means():
    """Cohort point estimate equals the simple mean across models of per-model means."""
    rng = np.random.default_rng(2026)
    models = ["m1", "m2", "m3"]
    personas = ["baseline", "good_persona", "bad_persona"]
    df = _multi_model_long(
        rng,
        models=models,
        personas=personas,
        persona_means={"baseline": 0.5, "good_persona": 0.8, "bad_persona": -0.3},
        n_per_principle=30,
        pair_correlation=0.4,
    )

    cells, deltas = bootstrap_cohort_principle_means(
        df, models=models, personas=personas, n_bootstrap=50,
    )

    # Hand-compute one cell: respect-user-attention, baseline.
    principle = "respect-user-attention"
    sub = df[(df["principle"] == principle) & (df["persona"] == "baseline")]
    per_model = sub.groupby("model")["score"].mean()
    expected = float(per_model.loc[models].mean())
    got = float(
        cells[(cells["principle"] == principle) & (cells["persona"] == "baseline")][
            "point_estimate"
        ].iloc[0]
    )
    assert got == pytest.approx(expected, abs=1e-10)

    # And the bad - baseline delta point estimate equals the difference of cohort means.
    bad_pt = float(
        cells[(cells["principle"] == principle) & (cells["persona"] == "bad_persona")][
            "point_estimate"
        ].iloc[0]
    )
    base_pt = float(
        cells[(cells["principle"] == principle) & (cells["persona"] == "baseline")][
            "point_estimate"
        ].iloc[0]
    )
    delta_pt = float(
        deltas[(deltas["principle"] == principle) & (deltas["contrast_persona"] == "bad_persona")][
            "point_estimate"
        ].iloc[0]
    )
    assert delta_pt == pytest.approx(bad_pt - base_pt, abs=1e-10)

    # n_models in output matches len(models) for every row in both frames.
    assert (cells["n_models"] == len(models)).all()
    assert (deltas["n_models"] == len(models)).all()

    # CI bounds are well-ordered around the point estimate. Use <= between
    # bound and point because a zero-variance bootstrap sample can pin a
    # percentile to the mean; strict < between the bounds themselves is fine
    # given the synthetic noise.
    assert (cells["ci_lower"] <= cells["point_estimate"]).all()
    assert (cells["point_estimate"] <= cells["ci_upper"]).all()
    assert (cells["ci_lower"] < cells["ci_upper"]).all()
    assert (deltas["ci_lower"] <= deltas["point_estimate"]).all()
    assert (deltas["point_estimate"] <= deltas["ci_upper"]).all()
    assert (deltas["ci_lower"] < deltas["ci_upper"]).all()


@pytest.mark.unit
def test_cohort_bootstrap_seed_reproducibility():
    """Identical seeds give identical cohort CIs (cells and deltas)."""
    rng = np.random.default_rng(7)
    models = ["m1", "m2", "m3"]
    personas = ["baseline", "bad_persona"]
    df = _multi_model_long(
        rng,
        models=models,
        personas=personas,
        persona_means={"baseline": 0.4, "bad_persona": -0.1},
        n_per_principle=25,
    )
    a_cells, a_deltas = bootstrap_cohort_principle_means(
        df, models=models, personas=personas, n_bootstrap=150, seed=BOOTSTRAP_SEED,
    )
    b_cells, b_deltas = bootstrap_cohort_principle_means(
        df, models=models, personas=personas, n_bootstrap=150, seed=BOOTSTRAP_SEED,
    )
    pd.testing.assert_frame_equal(a_cells, b_cells)
    pd.testing.assert_frame_equal(a_deltas, b_deltas)


@pytest.mark.unit
def test_cohort_resampling_unit_is_scenario_not_model():
    """When per-scenario noise dominates and model noise is zero, cohort-replicate
    variance should equal the variance of the per-scenario mean across models —
    i.e. scenarios drive bootstrap variation, models do not. We test this by
    constructing data where all models give identical scores for each
    (principle, sample_id) and confirming the cohort CI width matches the
    per-scenario-mean CI width from a one-model bootstrap.
    """
    rng = np.random.default_rng(11)
    models = ["m1", "m2", "m3", "m4"]
    personas = ["baseline"]
    # Build one base table with shared scenario scores; replicate it per model.
    base = _synth_long(
        rng,
        personas=personas,
        persona_means={"baseline": 0.5},
        n_per_principle=60,
    )
    base = base.drop(columns=["model"])
    rows = []
    for m in models:
        sub = base.copy()
        sub["model"] = m
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)

    cohort_cells, _ = bootstrap_cohort_principle_means(
        df, models=models, personas=personas, delta_personas=(),
        n_bootstrap=400, seed=BOOTSTRAP_SEED,
    )
    # Compare to the single-model marginal CI (which is the per-scenario-mean CI).
    one_model = df[df["model"] == "m1"].copy()
    one_cells = bootstrap_cell_scores(one_model, n_bootstrap=400, seed=BOOTSTRAP_SEED)

    principle = "respect-user-attention"
    cohort = cohort_cells[(cohort_cells["principle"] == principle)
                          & (cohort_cells["persona"] == "baseline")].iloc[0]
    single = one_cells[(one_cells["principle"] == principle)
                       & (one_cells["persona"] == "baseline")].iloc[0]
    cohort_width = cohort["ci_upper"] - cohort["ci_lower"]
    single_width = single["ci_upper"] - single["ci_lower"]
    # Widths must agree closely — any large gap means models are contributing
    # bootstrap variation, which would mean the resampling unit is wrong.
    assert abs(cohort_width - single_width) < 0.05 * single_width, (
        f"cohort width {cohort_width:.4f} differs from single-model width "
        f"{single_width:.4f} by more than 5% — resampling unit suspect"
    )
