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
    DesignedMeasuredMatrix,
    bootstrap_cell_scores,
    bootstrap_cohort_grid,
    bootstrap_cohort_principle_means,
    bootstrap_designed_measured_matrix,
    bootstrap_naive_grid,
    bootstrap_persona_deltas,
    cohort_flip_stats,
    diagonal_ranks,
    discriminant_contrasts,
    holm_adjust,
    pairwise_interactions,
)
from humanebench.bootstrap import _bootstrap_two_sided_p


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


# ---------------------------------------------------------------------------
# bootstrap_cohort_grid / cohort_flip_stats
#
# These back the cohort-level statistics in the paper (the anti-humane flip
# count and the size of the robust set). They are separate from
# bootstrap_cohort_principle_means above: the defining property here is that ONE
# scenario draw is carried across every (model, persona) cell, so that a count
# computed across models inherits the correlation a shared scenario induces.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cohort_grid_point_matches_mean_of_principle_means():
    """Grid point estimates equal the mean of the 8 principle means per cell."""
    rng = np.random.default_rng(4242)
    models = ["m1", "m2"]
    personas = ["baseline", "good_persona", "bad_persona"]
    df = _multi_model_long(
        rng, models=models, personas=personas,
        persona_means={"baseline": 0.5, "good_persona": 0.8, "bad_persona": -0.3},
        n_per_principle=20,
    )
    grid = bootstrap_cohort_grid(df, n_bootstrap=0)

    expected = (
        df.groupby(["model", "persona", "principle"])["score"].mean()
        .groupby(["model", "persona"]).mean()
    )
    for i, model in enumerate(grid.models):
        for j, persona in enumerate(grid.personas):
            assert grid.point[i, j] == pytest.approx(expected[(model, persona)]), (
                f"{model}/{persona} point estimate is not the mean of principle means"
            )


@pytest.mark.unit
def test_cohort_grid_shares_one_scenario_draw_across_cells():
    """The defining property: every cell sees the SAME resampled scenarios.

    Two models with byte-identical scores must therefore produce byte-identical
    replicates. Independent per-cell resampling (`bootstrap_naive_grid`) must
    not, which is what makes it the wrong design for a cohort statistic.
    """
    rng = np.random.default_rng(7)
    base = _synth_long(rng, personas=["baseline"],
                       persona_means={"baseline": 0.4}, n_per_principle=25)
    base = base.drop(columns=["model"])
    twin = pd.concat([base.assign(model=m) for m in ("m1", "m2")], ignore_index=True)

    shared = bootstrap_cohort_grid(twin, n_bootstrap=50)
    i1, i2 = shared.models.index("m1"), shared.models.index("m2")
    diff = shared.replicates[:, i1, 0] - shared.replicates[:, i2, 0]
    assert np.allclose(diff, 0.0), (
        "identical models diverged under a supposedly shared scenario draw — "
        "the resample is not being carried across cells"
    )

    naive = bootstrap_naive_grid(twin, n_bootstrap=50)
    j1, j2 = naive.models.index("m1"), naive.models.index("m2")
    naive_diff = naive.replicates[:, j1, 0] - naive.replicates[:, j2, 0]
    assert not np.allclose(naive_diff, 0.0), (
        "naive per-cell resampling produced identical replicates; the two "
        "designs are no longer distinguishable and the design-effect "
        "comparison is meaningless"
    )


@pytest.mark.unit
def test_cohort_grid_seed_reproducibility():
    """Same seed → identical replicates; different seed → different replicates."""
    rng = np.random.default_rng(99)
    df = _multi_model_long(
        rng, models=["m1", "m2"], personas=["baseline", "bad_persona"],
        persona_means={"baseline": 0.5, "bad_persona": -0.4}, n_per_principle=15,
    )
    a = bootstrap_cohort_grid(df, n_bootstrap=40, seed=BOOTSTRAP_SEED)
    b = bootstrap_cohort_grid(df, n_bootstrap=40, seed=BOOTSTRAP_SEED)
    c = bootstrap_cohort_grid(df, n_bootstrap=40, seed=BOOTSTRAP_SEED + 1)
    assert np.array_equal(a.replicates, b.replicates)
    assert not np.array_equal(a.replicates, c.replicates)


@pytest.mark.unit
def test_cohort_grid_tolerates_ragged_cells():
    """A scenario missing from one cell must not corrupt any other cell.

    Real logs are ragged: 23 of the 45 (model, persona) cells hold fewer than
    788 scenarios after judge-failure cascades.
    """
    rng = np.random.default_rng(123)
    df = _multi_model_long(
        rng, models=["m1", "m2"], personas=["baseline", "bad_persona"],
        persona_means={"baseline": 0.5, "bad_persona": -0.4}, n_per_principle=12,
    )
    victim = df["sample_id"].iloc[0]
    ragged = df.drop(df[(df.model == "m1") & (df.persona == "baseline")
                        & (df.sample_id == victim)].index)

    grid = bootstrap_cohort_grid(ragged, n_bootstrap=0)
    expected = (
        ragged.groupby(["model", "persona", "principle"])["score"].mean()
        .groupby(["model", "persona"]).mean()
    )
    for i, model in enumerate(grid.models):
        for j, persona in enumerate(grid.personas):
            assert grid.point[i, j] == pytest.approx(expected[(model, persona)])

    i = grid.models.index("m1")
    j = grid.personas.index("baseline")
    assert grid.n_missing[i, j] == 1, "raggedness not reported in n_missing"
    assert grid.n_missing.sum() == 1, "raggedness leaked into other cells"


@pytest.mark.unit
def test_cohort_flip_stats_counts_and_membership():
    """Flip = S_base > 0 AND S_bad < 0, counted across models."""
    rows = []
    # m_flip flips; m_robust stays positive; m_low is negative at baseline too.
    spec = {
        "m_flip": {"baseline": 0.6, "bad_persona": -0.6},
        "m_robust": {"baseline": 0.7, "bad_persona": 0.6},
        "m_low": {"baseline": -0.2, "bad_persona": -0.8},
    }
    for model, means in spec.items():
        for persona, mean in means.items():
            for principle in PRINCIPLES:
                for k in range(10):
                    rows.append({"persona": persona, "model": model,
                                 "principle": principle,
                                 "sample_id": f"{principle}-{k:03d}",
                                 "score": mean})
    df = pd.DataFrame(rows)

    stats = cohort_flip_stats(bootstrap_cohort_grid(df, n_bootstrap=20))
    assert stats["flip_sign"]["point"] == 1
    assert stats["flip_sign"]["models"] == ("m_flip",)
    # Constant scores → no resampling variation → degenerate CI.
    assert stats["flip_sign"]["ci"] == (1.0, 1.0)
    # m_robust is the only cell at or above the 0.5 robustness threshold.
    assert stats["robust_sbad"]["models"] == ("m_robust",)


@pytest.mark.unit
def test_cohort_flip_stats_adversarial_persona_is_selectable():
    """`adversarial_persona` must default to bad_persona and be overridable.

    Guards the reported numbers against a default drift when additional
    adversarial conditions are added to the logs. Skips cleanly if the
    parameter is not present yet, so this test can land before the
    parameterisation it guards.
    """
    import inspect as _inspect
    if "adversarial_persona" not in _inspect.signature(cohort_flip_stats).parameters:
        pytest.skip("cohort_flip_stats has no adversarial_persona parameter yet")

    rows = []
    # Under bad_persona m1 flips; under decoy_persona it does not.
    spec = {"baseline": 0.6, "bad_persona": -0.5, "decoy_persona": 0.4}
    for persona, mean in spec.items():
        for principle in PRINCIPLES:
            for k in range(8):
                rows.append({"persona": persona, "model": "m1",
                             "principle": principle,
                             "sample_id": f"{principle}-{k:03d}", "score": mean})
    df = pd.DataFrame(rows)
    grid = bootstrap_cohort_grid(df, n_bootstrap=10)

    assert cohort_flip_stats(grid)["flip_sign"]["point"] == 1, (
        "default adversarial persona is no longer bad_persona — every reported "
        "flip count depends on this default"
    )
    other = cohort_flip_stats(grid, adversarial_persona="decoy_persona")
    assert other["flip_sign"]["point"] == 0


# ---------------------------------------------------------------------------
# Designed x measured principle matrix
# ---------------------------------------------------------------------------


def _synth_matrix_long(
    rng: np.random.Generator,
    *,
    diagonal_effect: float = 0.0,
    column_offsets: dict[str, float] | None = None,
    n_per_principle: int = 12,
    models: list[str] | None = None,
    noise: float = 0.1,
) -> pd.DataFrame:
    """Synthetic multi-label scores with a known diagonal and column structure."""
    models = models or ["m1", "m2", "m3"]
    offsets = column_offsets or {}
    rows = []
    for designed in PRINCIPLES:
        for k in range(n_per_principle):
            scenario_id = f"{designed}-{k:03d}"
            for model in models:
                for scored in PRINCIPLES:
                    score = 0.5 + offsets.get(scored, 0.0)
                    if scored == designed:
                        score += diagonal_effect
                    rows.append({
                        "scenario_id": scenario_id,
                        "source_model": model,
                        "designed_principle": designed,
                        "scored_principle": scored,
                        "score": score + rng.normal(0.0, noise),
                    })
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_designed_measured_recovers_a_known_diagonal_effect():
    """A planted diagonal effect is recovered, and its CI excludes zero."""
    rng = np.random.default_rng(11)
    long = _synth_matrix_long(rng, diagonal_effect=-0.5, noise=0.2)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=300, seed=BOOTSTRAP_SEED)

    assert matrix.point.shape == (8, 8)
    assert set(np.unique(matrix.n_per_cell)) == {36}, "12 scenarios x 3 models"
    assert set(np.unique(matrix.n_scenarios)) == {12}

    contrasts = discriminant_contrasts(matrix)
    pooled = contrasts[contrasts.designed_principle == "pooled"].iloc[0]
    assert pooled.contrast == pytest.approx(-0.5, abs=0.06)
    assert pooled.excludes_zero
    assert pooled.ci_upper < 0


@pytest.mark.unit
def test_flat_matrix_gives_a_contrast_that_includes_zero():
    """No planted effect must not manufacture one -- the null has to be reachable.

    If this ever fails, the analysis cannot report the "principles do not
    discriminate" outcome the pre-committed interpretation requires.
    """
    rng = np.random.default_rng(12)
    long = _synth_matrix_long(rng, diagonal_effect=0.0, noise=0.3)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=300, seed=BOOTSTRAP_SEED)
    pooled = discriminant_contrasts(matrix).iloc[-1]

    assert pooled.contrast == pytest.approx(0.0, abs=0.06)
    assert not pooled.excludes_zero


@pytest.mark.unit
def test_column_centring_removes_a_harsh_rubric_artifact():
    """A harsh column with no real diagonal effect fools the raw row contrast.

    This is the objection the raw contrast cannot answer: if one principle's
    rubric is simply strict, its whole column is depressed, and that column's own
    diagonal then looks discriminating for a rubric reason. Column-centring is
    the correction, and the test plants exactly that artifact.
    """
    rng = np.random.default_rng(13)
    harsh = PRINCIPLES[0]
    long = _synth_matrix_long(rng, diagonal_effect=0.0,
                              column_offsets={harsh: -0.8}, noise=0.05)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=200, seed=BOOTSTRAP_SEED)

    raw = discriminant_contrasts(matrix, centered=False).set_index("designed_principle")
    centered = discriminant_contrasts(matrix, centered=True).set_index("designed_principle")

    assert raw.loc[harsh, "contrast"] < -0.6, "raw contrast should be fooled"
    assert centered.loc[harsh, "contrast"] == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_pooled_contrast_is_invariant_to_additive_column_effects():
    """Pooling already cancels any additive column effect, exactly.

    A column offset d raises its own row's contrast by d and lowers each of the
    other seven by d/7, which sums to zero. So the pooled headline needs no
    column correction and the centred variant only changes the per-row numbers.
    Worth pinning: it is the reason the headline can be reported raw.
    """
    rng = np.random.default_rng(14)
    offsets = {PRINCIPLES[0]: -0.8, PRINCIPLES[3]: +0.4}
    long = _synth_matrix_long(rng, diagonal_effect=-0.3,
                              column_offsets=offsets, noise=0.05)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=100, seed=BOOTSTRAP_SEED)

    raw = discriminant_contrasts(matrix, centered=False).iloc[-1]
    centered = discriminant_contrasts(matrix, centered=True).iloc[-1]
    assert raw.contrast == pytest.approx(centered.contrast, abs=1e-9)


@pytest.mark.unit
def test_shared_scenario_draw_tightens_the_within_row_contrast():
    """The shared per-row draw must preserve the within-scenario pairing.

    Every cell in a row is built from the same scenarios, so a scenario that is
    simply harsh moves all eight cells together and cancels out of the contrast.
    Resampling each cell independently would discard that and inflate the CI.
    Here the scenario effect is large relative to the noise, so the paired CI
    must come out clearly narrower.
    """
    rng = np.random.default_rng(15)
    rows = []
    for designed in PRINCIPLES:
        for k in range(12):
            scenario_id = f"{designed}-{k:03d}"
            hardness = rng.normal(0.0, 0.8)  # shared across all 8 columns
            for scored in PRINCIPLES:
                score = 0.5 + hardness + (-0.4 if scored == designed else 0.0)
                rows.append({
                    "scenario_id": scenario_id, "source_model": "m1",
                    "designed_principle": designed, "scored_principle": scored,
                    "score": score + rng.normal(0.0, 0.05),
                })
    long = pd.DataFrame(rows)

    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=400, seed=BOOTSTRAP_SEED)
    paired = discriminant_contrasts(matrix).iloc[-1]
    paired_width = paired.ci_upper - paired.ci_lower

    # Unpaired counterfactual: resample each cell's scenarios independently.
    rng2 = np.random.default_rng(BOOTSTRAP_SEED)
    cells = {}
    for designed in PRINCIPLES:
        sub = long[long.designed_principle == designed]
        for scored in PRINCIPLES:
            cells[(designed, scored)] = (
                sub[sub.scored_principle == scored]
                .groupby("scenario_id")["score"].mean().to_numpy()
            )
    reps = []
    for _ in range(400):
        mat = np.array([
            [cells[(d, s)][rng2.integers(0, 12, 12)].mean() for s in PRINCIPLES]
            for d in PRINCIPLES
        ])
        diag = np.diagonal(mat)
        off = (mat.sum(axis=1) - diag) / 7.0
        reps.append((diag - off).mean())
    unpaired_width = float(np.percentile(reps, 97.5) - np.percentile(reps, 2.5))

    assert paired_width < unpaired_width, (
        f"paired CI ({paired_width:.4f}) should be narrower than unpaired "
        f"({unpaired_width:.4f}); the shared scenario draw is not being applied"
    )


@pytest.mark.unit
def test_designed_measured_is_seed_reproducible_and_nan_tolerant():
    rng = np.random.default_rng(16)
    long = _synth_matrix_long(rng, diagonal_effect=-0.4)
    a = discriminant_contrasts(
        bootstrap_designed_measured_matrix(long, n_bootstrap=50, seed=7))
    b = discriminant_contrasts(
        bootstrap_designed_measured_matrix(long, n_bootstrap=50, seed=7))
    pd.testing.assert_frame_equal(a, b)

    # A judge failure leaves a hole; the cell means around it must still compute.
    holed = long.drop(long.index[:25])
    matrix = bootstrap_designed_measured_matrix(holed, n_bootstrap=50, seed=7)
    assert matrix.n_per_cell.min() < 36
    assert np.isfinite(matrix.point).all()


@pytest.mark.unit
def test_diagonal_ranks_are_ordinal_and_direction_correct():
    """Rank 1 is the lowest cell, in the row and in the column."""
    rng = np.random.default_rng(17)
    long = _synth_matrix_long(rng, diagonal_effect=-0.6, noise=0.05)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=100, seed=BOOTSTRAP_SEED)
    ranks = diagonal_ranks(matrix)

    assert (ranks.rank_in_row == 1).all()
    assert (ranks.rank_in_column == 1).all()
    assert (ranks.share_lowest_in_row > 0.9).all()

    # Reverse the effect: the diagonal becomes the highest cell in its row.
    long_high = _synth_matrix_long(np.random.default_rng(18),
                                   diagonal_effect=+0.6, noise=0.05)
    high = diagonal_ranks(
        bootstrap_designed_measured_matrix(long_high, n_bootstrap=100,
                                           seed=BOOTSTRAP_SEED))
    assert (high.rank_in_row == 8).all()


@pytest.mark.unit
def test_mirror_ranks_track_the_reversed_direction():
    """The from-top columns must be the same claim with the inequality flipped.

    They exist because the real run came out reversed. If they were merely the
    complement of the lowest-rank columns they would add nothing; the check is
    that a planted *positive* diagonal, which the committed direction scores as
    the weakest possible result, is scored by the mirror as the strongest.
    """
    matrix = bootstrap_designed_measured_matrix(
        _synth_matrix_long(np.random.default_rng(18), diagonal_effect=+0.6,
                           noise=0.05),
        n_bootstrap=100, seed=BOOTSTRAP_SEED)
    ranks = diagonal_ranks(matrix)

    assert (ranks.rank_in_row_from_top == 1).all()
    assert (ranks.rank_in_column_from_top == 1).all()
    assert (ranks.share_highest_in_row > 0.9).all()
    assert (ranks.share_top_two_in_row >= ranks.share_highest_in_row).all()
    # The committed direction sees nothing here -- both must be reported.
    assert (ranks.share_lowest_in_row < 0.1).all()

    # An unestimable row is not ranked in either direction. NaN comparisons read
    # False, so a naive mirror would call an absent diagonal the highest cell.
    holed = matrix.point.copy()
    holed[2, 2] = np.nan
    reps = matrix.replicates.copy()
    reps[:, 2, 2] = np.nan
    gapped = diagonal_ranks(DesignedMeasuredMatrix(
        principles=matrix.principles, models=matrix.models, point=holed,
        replicates=reps, n_per_cell=matrix.n_per_cell,
        n_scenarios=matrix.n_scenarios))
    row = gapped.iloc[2]
    assert not row.estimable
    # Not a rank of any kind: the column is float once a row goes unranked, so
    # the guarantee is "missing", not the literal None that was appended.
    assert pd.isna(row.rank_in_row_from_top)
    assert pd.isna(row.rank_in_column_from_top)
    assert np.isnan(row.share_highest_in_row)


@pytest.mark.unit
def test_cell_difference_is_paired_within_replicate():
    """The FHR-vs-PLTW style comparison must be a paired difference."""
    rng = np.random.default_rng(19)
    long = _synth_matrix_long(rng, diagonal_effect=-0.5, noise=0.2)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=300, seed=BOOTSTRAP_SEED)

    fhr, pltw = "foster-healthy-relationships", "prioritize-long-term-wellbeing"
    point, lo, hi = matrix.cell_difference(fhr, fhr, pltw)
    assert point == pytest.approx(-0.5, abs=0.1)
    assert hi < 0, "the planted effect should be detected"

    paired_width = hi - lo
    i = matrix.principles.index(fhr)
    a, b = matrix.replicates[:, i, i], matrix.replicates[:, i, matrix.principles.index(pltw)]
    naive_width = float(
        np.sqrt((np.percentile(a, 97.5) - np.percentile(a, 2.5)) ** 2
                + (np.percentile(b, 97.5) - np.percentile(b, 2.5)) ** 2)
    )
    assert paired_width < naive_width


@pytest.mark.unit
def test_empty_cell_does_not_poison_other_rows_or_the_pooled_headline():
    """One unscored cell must cost that row, not the whole matrix.

    Regression for the review: column-centring used a plain mean, so a single
    all-NaN cell turned every centred contrast NaN, and the pooled contrast used
    a plain mean, so the paper's headline number rendered as an em-dash and as
    "does not exclude zero" on data where seven of eight rows were fine.
    """
    rng = np.random.default_rng(31)
    long = _synth_matrix_long(rng, diagonal_effect=-0.5, noise=0.15)
    # Delete one (designed, scored) cell entirely -- all 36 of its observations.
    hole = (long.designed_principle == PRINCIPLES[0]) & \
           (long.scored_principle == PRINCIPLES[5])
    long = long[~hole]
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=200, seed=BOOTSTRAP_SEED)
    assert matrix.n_per_cell[0, 5] == 0
    assert np.isnan(matrix.point[0, 5])

    for centered in (False, True):
        c = discriminant_contrasts(matrix, centered=centered).set_index("designed_principle")
        pooled = c.loc["pooled"]
        assert pooled.estimable, f"pooled unestimable with centered={centered}"
        assert np.isfinite(pooled.contrast) and np.isfinite(pooled.ci_lower)
        assert pooled.contrast == pytest.approx(-0.5, abs=0.12)
        # Rows that share the holed column must still be estimable.
        others = c.drop(index=["pooled"])
        assert others.estimable.all(), f"a hole in one cell disabled {(~others.estimable).sum()} rows"

    # The raw == centred identity is EXACT only on a complete matrix: a hole
    # makes its row average six off-diagonal cells instead of seven and its
    # column's mean cover fewer rows, so the offsets no longer cancel term for
    # term. It must stay close, but the analysis must not assert exactness here.
    raw = discriminant_contrasts(matrix, centered=False).iloc[-1]
    cen = discriminant_contrasts(matrix, centered=True).iloc[-1]
    assert raw.contrast != pytest.approx(cen.contrast, abs=1e-12)
    assert raw.contrast == pytest.approx(cen.contrast, abs=1e-2)


@pytest.mark.unit
def test_unestimable_row_is_flagged_not_scored_as_a_null():
    """A row with no data must read as 'no data', never as a null result."""
    rng = np.random.default_rng(32)
    long = _synth_matrix_long(rng, diagonal_effect=-0.5, noise=0.15)
    long = long[long.designed_principle != PRINCIPLES[3]]  # drop a whole row
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=100, seed=BOOTSTRAP_SEED)

    c = discriminant_contrasts(matrix).set_index("designed_principle")
    dead = c.loc[PRINCIPLES[3]]
    assert not dead.estimable
    assert not dead.excludes_zero  # False here means "cannot say", per the flag
    assert np.isnan(dead.contrast)
    assert c.loc["pooled"].n_rows_used == 7, "pooled must report the rows it used"

    ranks = diagonal_ranks(matrix).set_index("designed_principle")
    dead_rank = ranks.loc[PRINCIPLES[3]]
    assert not dead_rank.estimable
    assert pd.isna(dead_rank.rank_in_row), (
        "an unscored row must not be ranked -- NaN < NaN is False, which would "
        "report it as rank 1 of 8, the strongest ordinal evidence in the table"
    )
    assert np.isnan(dead_rank.share_lowest_in_row)
    # Surviving rows are unaffected and still ranked against real competitors.
    alive = ranks.drop(index=[PRINCIPLES[3]])
    assert alive.estimable.all()
    assert (alive.n_cells_ranked_in_column == 7).all()


@pytest.mark.unit
def test_duplicate_judged_calls_are_rejected():
    """Duplicates would be silently deduped by fancy-index assignment."""
    rng = np.random.default_rng(33)
    long = _synth_matrix_long(rng, diagonal_effect=-0.3)
    doubled = pd.concat([long, long], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        bootstrap_designed_measured_matrix(doubled, n_bootstrap=10)


@pytest.mark.unit
def test_pairwise_interaction_recovers_a_planted_effect():
    """A planted diagonal effect of d shows up as an interaction of 2d.

    Each inner difference contributes d with opposite sign -- `a - b` gains it
    and `c - d` loses it -- so the difference of differences doubles it.
    """
    rng = np.random.default_rng(41)
    long = _synth_matrix_long(rng, diagonal_effect=0.4, noise=0.2)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=2000,
                                                seed=BOOTSTRAP_SEED)
    pw = pairwise_interactions(matrix)

    assert len(pw) == 28, "8 principles give 28 unordered pairs, not 56"
    assert pw.estimable.all()
    assert pw.interaction.min() == pytest.approx(0.8, abs=0.15)
    assert pw.interaction.max() == pytest.approx(0.8, abs=0.15)
    assert pw.excludes_zero.all()
    # 2 / (B + 1) = 0.001 clears Holm's first threshold of 0.05 / 28 = 0.00179,
    # so the family is resolvable and a real effect can be detected.
    assert (pw.p_holm < 0.05).all()


@pytest.mark.unit
def test_pairwise_interaction_is_immune_to_rubric_leniency():
    """Column offsets alone must produce no interaction.

    This is the property the diagonal contrast lacks and the reason this
    statistic replaced it: a rubric that is uniformly harsher than another
    depresses its whole column, which the within-row contrast reads as signal
    and the difference-in-differences cancels exactly.
    """
    offsets = {p: v for p, v in zip(PRINCIPLES, [-0.6, -0.4, -0.2, 0.0,
                                                 0.2, 0.4, 0.6, 0.8])}
    rng = np.random.default_rng(42)
    long = _synth_matrix_long(rng, diagonal_effect=0.0,
                              column_offsets=offsets, noise=0.2)
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=2000,
                                                seed=BOOTSTRAP_SEED)
    pw = pairwise_interactions(matrix)

    assert pw.interaction.abs().max() < 0.15, (
        "leniency differences of up to 1.4 scale points moved the interaction; "
        "the difference-in-differences is not cancelling column effects"
    )
    # Not `excludes_zero.any() is False`: 28 uncorrected 95% intervals under a
    # true null are *expected* to throw ~1.4 false positives, so demanding zero
    # would be asserting that the CI has no type-I error rate at all. The
    # family-wise claim is the one Holm makes, and that is what is checked.
    assert pw.excludes_zero.sum() <= 4, "far above the ~1.4 expected by chance"
    assert not (pw.p_holm < 0.05).any(), (
        "Holm must control the family-wise error rate under a true null"
    )
    # The same data through the pre-committed contrast, for contrast: column
    # offsets DO move it, which is why it needed a centred companion.
    raw = discriminant_contrasts(matrix).set_index("designed_principle")
    assert raw.loc["pooled"].contrast != pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_pairwise_interaction_is_symmetric_and_flags_missing_cells():
    """Order of the pair cannot matter, and an absent row is not a null."""
    rng = np.random.default_rng(43)
    long = _synth_matrix_long(rng, diagonal_effect=0.3, noise=0.15)

    # Symmetry: rebuild with the principle order reversed in the labels and
    # confirm the same pair gets the same number.
    matrix = bootstrap_designed_measured_matrix(long, n_bootstrap=200,
                                                seed=BOOTSTRAP_SEED)
    pw = pairwise_interactions(matrix)
    x, y = PRINCIPLES[1], PRINCIPLES[5]
    row = pw[(pw.principle_a == x) & (pw.principle_b == y)].iloc[0]
    i, j = matrix.principles.index(x), matrix.principles.index(y)
    flipped = ((matrix.point[j, j] - matrix.point[j, i])
               - (matrix.point[i, j] - matrix.point[i, i]))
    assert row.interaction == pytest.approx(flipped, abs=1e-12)

    # A dropped row makes every pair containing it unestimable, not zero.
    holed = long[long.designed_principle != PRINCIPLES[3]]
    hpw = pairwise_interactions(
        bootstrap_designed_measured_matrix(holed, n_bootstrap=100,
                                           seed=BOOTSTRAP_SEED))
    dead = hpw[(hpw.principle_a == PRINCIPLES[3])
               | (hpw.principle_b == PRINCIPLES[3])]
    assert len(dead) == 7
    assert not dead.estimable.any()
    assert not dead.excludes_zero.any()  # False here means "cannot say"
    assert dead.p_value.isna().all()
    assert dead.p_holm.isna().all(), (
        "an unestimable pair must not consume a Holm step; ranking it would "
        "make every real test stricter for a test that was never run"
    )
    assert hpw[hpw.estimable].p_holm.notna().all()


@pytest.mark.unit
def test_bootstrap_p_floor_is_two_over_b_plus_one():
    """The floor that forces the pairwise replicate count is real, not folklore.

    `scripts/compute_discriminant_pairwise.py` raises B from 1,000 to 10,000
    because 2 / 1001 = 0.0020 exceeds Holm's first threshold for 28 tests
    (0.05 / 28 = 0.00179), making the family unresolvable regardless of the
    data. If this convention ever changes, that reasoning must be revisited.
    """
    for b in (1000, 10_000):
        all_positive = np.full(b, 1.0)
        assert _bootstrap_two_sided_p(all_positive) == pytest.approx(2 / (b + 1))
    assert 2 / 1001 > 0.05 / 28, "the documented conflict at B=1,000"
    assert 2 / 10_001 < 0.05 / 28, "and its resolution at B=10,000"

    # Straddling zero symmetrically is the least significant possible outcome.
    straddle = np.concatenate([np.full(500, -1.0), np.full(500, 1.0)])
    assert _bootstrap_two_sided_p(straddle) == pytest.approx(1.0)
    assert np.isnan(_bootstrap_two_sided_p(np.full(10, np.nan)))


@pytest.mark.unit
def test_holm_adjust_matches_the_textbook_and_skips_nan():
    """Step-down, monotone, and NaN excluded from the family size."""
    p = [0.01, 0.02, 0.03, 0.04]
    adj = holm_adjust(p)
    np.testing.assert_allclose(adj, [0.04, 0.06, 0.06, 0.06])
    assert np.all(np.diff(adj) >= 0), "must be monotone non-decreasing"

    # A NaN shrinks the family from 4 to 3 rather than being ranked.
    with_nan = holm_adjust([0.01, 0.02, np.nan, 0.04])
    assert np.isnan(with_nan[2])
    np.testing.assert_allclose(with_nan[[0, 1, 3]], [0.03, 0.04, 0.04])
    assert holm_adjust([0.9, 0.9]) .max() <= 1.0, "capped at 1"
    assert np.isnan(holm_adjust([np.nan, np.nan])).all()
