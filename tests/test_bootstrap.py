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
