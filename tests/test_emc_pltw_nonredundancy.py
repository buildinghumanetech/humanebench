"""Tests for emc/pltw non-redundancy analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from humanebench.bootstrap import PRINCIPLES  # noqa: E402


# ---------------------------------------------------------------------------
# EIV correction tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_eiv_recovers_near_one_for_redundant_columns():
    """Planted redundant columns: shared latent + independent noise.
    Raw r < 1 due to noise; corrected r ≈ 1."""
    from compute_emc_pltw_nonredundancy import eiv_corrected_pearson

    rng = np.random.default_rng(42)
    n_models = 15
    latent = rng.normal(0, 1, size=n_models)

    # Single-draw noise (simulates one scenario sample, not averaged over many)
    noise_x = rng.normal(0, 0.5, size=n_models)
    noise_y = rng.normal(0, 0.5, size=n_models)

    x = latent + noise_x
    y = latent + noise_y

    # SE^2 represents the known measurement error variance
    se2_x = np.full(n_models, 0.25)  # 0.5^2
    se2_y = np.full(n_models, 0.25)

    from scipy.stats import pearsonr
    raw_r, _ = pearsonr(x, y)
    result = eiv_corrected_pearson(x, y, se2_x, se2_y)

    assert raw_r < 0.999, f"raw r should be attenuated, got {raw_r}"
    assert result["valid"]
    assert abs(result["r_corrected"] - 1.0) < 0.15, f"corrected r should be near 1, got {result['r_corrected']}"


@pytest.mark.unit
def test_eiv_corrected_below_one_for_distinct_columns():
    """Planted distinct columns with known latent r < 1."""
    from compute_emc_pltw_nonredundancy import eiv_corrected_pearson

    rng = np.random.default_rng(99)
    n_models = 15
    n_boot = 200

    shared = rng.normal(0, 1, size=n_models)
    indep = rng.normal(0, 1, size=n_models)
    latent_x = shared + indep * 0.5
    latent_y = shared - indep * 0.5

    noise_x = rng.normal(0, 0.2, size=(n_boot, n_models))
    noise_y = rng.normal(0, 0.2, size=(n_boot, n_models))

    x = latent_x + noise_x.mean(axis=0)
    y = latent_y + noise_y.mean(axis=0)
    se2_x = np.var(noise_x, axis=0, ddof=1)
    se2_y = np.var(noise_y, axis=0, ddof=1)

    result = eiv_corrected_pearson(x, y, se2_x, se2_y)
    assert result["valid"]
    assert result["r_corrected"] < 0.95, f"corrected r should be < 0.95, got {result['r_corrected']}"


@pytest.mark.unit
def test_eiv_invalid_when_noise_exceeds_signal():
    """When noise variance >= signal variance, the correction is invalid."""
    from compute_emc_pltw_nonredundancy import eiv_corrected_pearson

    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    y = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    se2_x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    se2_y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result = eiv_corrected_pearson(x, y, se2_x, se2_y)
    assert not result["valid"]
    assert np.isnan(result["r_corrected"])


# ---------------------------------------------------------------------------
# Pair-stats and focal-pair join tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pair_stats_determinism():
    """Same seed produces identical pair_stats."""
    from compute_emc_pltw_nonredundancy import pair_stats
    from humanebench.bootstrap import bootstrap_model_principle_grid

    rng = np.random.default_rng(77)
    rows = []
    models = ["m1", "m2", "m3"]
    personas = ["baseline"]
    for principle in PRINCIPLES:
        for k in range(15):
            for model in models:
                rows.append({
                    "persona": "baseline", "model": model,
                    "principle": principle,
                    "sample_id": f"{principle}-{k:03d}",
                    "score": rng.normal() * 0.5,
                })
    df = pd.DataFrame(rows)

    g1 = bootstrap_model_principle_grid(df, models, personas, n_bootstrap=30, seed=12345)
    g2 = bootstrap_model_principle_grid(df, models, personas, n_bootstrap=30, seed=12345)

    ps1 = pair_stats(g1, 0, 0, 1)
    ps2 = pair_stats(g2, 0, 0, 1)
    assert ps1["pearson_r"] == ps2["pearson_r"]
    assert ps1["r_corrected"] == ps2["r_corrected"] or (np.isnan(ps1["r_corrected"]) and np.isnan(ps2["r_corrected"]))


@pytest.mark.unit
def test_focal_pair_join_on_synthetic_pairwise():
    """Unordered pair normalization works for the discriminant join."""
    from compute_emc_pltw_nonredundancy import all_pair_correlations
    from humanebench.bootstrap import bootstrap_model_principle_grid

    rng = np.random.default_rng(88)
    rows = []
    models = ["m1", "m2", "m3"]
    personas = ["baseline"]
    for principle in PRINCIPLES:
        for k in range(10):
            for model in models:
                rows.append({
                    "persona": "baseline", "model": model,
                    "principle": principle,
                    "sample_id": f"{principle}-{k:03d}",
                    "score": rng.normal(),
                })
    df = pd.DataFrame(rows)
    grid = bootstrap_model_principle_grid(df, models, personas, n_bootstrap=20, seed=42)

    # Create a tiny pairwise CSV with reversed pair order
    pairwise_path = Path("$TMPDIR") / "test_pairwise.csv"
    pairwise_df = pd.DataFrame([{
        "principle_a": "prioritize-long-term-wellbeing",
        "principle_b": "enable-meaningful-choices",
        "significant": True,
        "p_holm": 0.01,
        "equivalent": False,
    }])
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        pairwise_df.to_csv(f, index=False)
        pairwise_path = Path(f.name)

    try:
        result = all_pair_correlations(grid, pairwise_path)
        focal = result[result["is_focal_pair"]]
        assert len(focal) == 1, f"expected 1 focal row, got {len(focal)}"
        assert focal.iloc[0]["scenario_level_significant"] is True
    finally:
        pairwise_path.unlink(missing_ok=True)
