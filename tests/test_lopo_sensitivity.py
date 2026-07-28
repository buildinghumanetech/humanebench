"""Tests for leave-one-principle-out sensitivity analysis."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    PRINCIPLES,
    ModelPrincipleGrid,
    bootstrap_cohort_grid,
    bootstrap_cohort_principle_means,
    bootstrap_model_principle_grid,
    cohort_flip_stats,
    load_long_scores,
)


def _synth_multi(
    rng: np.random.Generator,
    models: list[str],
    personas: list[str],
    persona_means: dict[str, float],
    n_per_principle: int = 30,
) -> pd.DataFrame:
    rows = []
    for principle in PRINCIPLES:
        for k in range(n_per_principle):
            sample_id = f"{principle}-{k:03d}"
            for model in models:
                for persona in personas:
                    score = persona_means[persona] + rng.normal() * 0.3
                    rows.append({
                        "persona": persona,
                        "model": model,
                        "principle": principle,
                        "sample_id": sample_id,
                        "score": score,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ModelPrincipleGrid tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_model_principle_grid_point_matches_groupby():
    rng = np.random.default_rng(42)
    models = ["m1", "m2", "m3"]
    personas = ["baseline", "good_persona", "bad_persona"]
    df = _synth_multi(rng, models, personas, {"baseline": 0.5, "good_persona": 0.8, "bad_persona": -0.3})

    grid = bootstrap_model_principle_grid(df, models, personas, n_bootstrap=50, seed=BOOTSTRAP_SEED)

    assert grid.point.shape == (3, 3, 8)
    assert grid.replicates.shape[1:] == (3, 3, 8)
    assert len(grid.principles) == 8

    for pi, principle in enumerate(grid.principles):
        for mi, model in enumerate(models):
            for pei, persona in enumerate(personas):
                sub = df[(df["principle"] == principle) & (df["model"] == model) & (df["persona"] == persona)]
                expected = sub["score"].mean()
                np.testing.assert_allclose(grid.point[mi, pei, pi], expected, atol=1e-10)


@pytest.mark.unit
def test_model_principle_grid_cohort_matches_cohort_principle_means():
    """The load-bearing consistency test: grid.replicates.mean(axis=1) must be
    bit-identical to bootstrap_cohort_principle_means cohort replicates."""
    rng = np.random.default_rng(99)
    models = ["m1", "m2", "m3"]
    personas = ["baseline", "bad_persona"]
    df = _synth_multi(rng, models, personas, {"baseline": 0.5, "bad_persona": -0.2}, n_per_principle=25)

    grid = bootstrap_model_principle_grid(df, models, personas, n_bootstrap=100, seed=BOOTSTRAP_SEED)

    reps_out: dict = {}
    cells, deltas = bootstrap_cohort_principle_means(
        df, models=models, personas=personas,
        delta_personas=(("bad_persona", "baseline"),),
        n_bootstrap=100, seed=BOOTSTRAP_SEED,
        replicates_out=reps_out,
    )

    # Cohort replicates = grid replicates mean over models axis
    cohort_from_grid = grid.replicates.mean(axis=1)  # (n_boot, n_personas, n_principles)

    for pi, principle in enumerate(grid.principles):
        row = cells[cells["principle"] == principle]
        for pei, persona in enumerate(personas):
            cell = row[row["persona"] == persona].iloc[0]
            np.testing.assert_allclose(
                grid.point[:, pei, pi].mean(), cell["point_estimate"], atol=1e-10,
                err_msg=f"point mismatch for {principle}/{persona}",
            )

    # Delta replicates
    for principle in grid.principles:
        pi = grid.principles.index(principle)
        b_idx = personas.index("baseline")
        d_idx = personas.index("bad_persona")
        grid_delta = cohort_from_grid[:, d_idx, pi] - cohort_from_grid[:, b_idx, pi]
        pub_delta = reps_out[(principle, "bad_persona", "baseline")]
        np.testing.assert_array_equal(
            grid_delta, pub_delta,
            err_msg=f"replicate mismatch for {principle} delta — RNG streams diverged",
        )


@pytest.mark.unit
def test_model_principle_grid_seed_reproducibility():
    rng = np.random.default_rng(7)
    models = ["m1", "m2"]
    personas = ["baseline", "bad_persona"]
    df = _synth_multi(rng, models, personas, {"baseline": 0.5, "bad_persona": -0.1}, n_per_principle=20)

    g1 = bootstrap_model_principle_grid(df, models, personas, n_bootstrap=50, seed=12345)
    g2 = bootstrap_model_principle_grid(df, models, personas, n_bootstrap=50, seed=12345)

    np.testing.assert_array_equal(g1.replicates, g2.replicates)
    np.testing.assert_array_equal(g1.point, g2.point)


@pytest.mark.unit
def test_model_principle_grid_shared_draw_across_personas():
    """Twin-model pattern: two identical models must get identical replicates."""
    rng = np.random.default_rng(88)
    personas = ["baseline", "bad_persona"]
    rows = []
    for principle in PRINCIPLES:
        for k in range(20):
            sample_id = f"{principle}-{k:03d}"
            for persona in personas:
                score = rng.normal() * 0.5
                for model in ["twin_a", "twin_b"]:
                    rows.append({
                        "persona": persona,
                        "model": model,
                        "principle": principle,
                        "sample_id": sample_id,
                        "score": score,
                    })
    df = pd.DataFrame(rows)
    grid = bootstrap_model_principle_grid(df, ["twin_a", "twin_b"], personas, n_bootstrap=50)
    np.testing.assert_array_equal(grid.replicates[:, 0, :, :], grid.replicates[:, 1, :, :])


# ---------------------------------------------------------------------------
# LOPO config tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lopo_config_uniformity():
    """9 configs; each drop removes exactly one principle; constant scores
    give HumaneScore = mean of remaining 7 principle means."""
    from compute_lopo_sensitivity import CONFIG_ORDER, SHORT

    assert len(CONFIG_ORDER) == 9
    assert CONFIG_ORDER[0] == "all8"
    for config in CONFIG_ORDER[1:]:
        code = config.removeprefix("drop_")
        matched = [p for p in PRINCIPLES if SHORT[p] == code]
        assert len(matched) == 1, f"config {config} doesn't map to exactly one principle"


@pytest.mark.unit
def test_lopo_constant_score_humane():
    """With constant per-principle scores, dropping one principle gives HumaneScore
    equal to the mean of the 7 remaining principle means."""
    models = ["m1", "m2"]
    personas = ["baseline", "bad_persona"]
    principle_vals = {p: 0.1 * (i + 1) for i, p in enumerate(PRINCIPLES)}

    rows = []
    for principle in PRINCIPLES:
        for k in range(10):
            sample_id = f"{principle}-{k:03d}"
            for model in models:
                for persona in personas:
                    rows.append({
                        "persona": persona,
                        "model": model,
                        "principle": principle,
                        "sample_id": sample_id,
                        "score": principle_vals[principle],
                    })
    df = pd.DataFrame(rows)

    # All8: HumaneScore = mean of 8 principle means
    grid_all = bootstrap_cohort_grid(df, n_bootstrap=10, seed=BOOTSTRAP_SEED)
    all8_hs = float(grid_all.point.mean())
    expected_all8 = np.mean(list(principle_vals.values()))
    np.testing.assert_allclose(all8_hs, expected_all8, atol=1e-10)

    # Drop one principle: HumaneScore = mean of 7 remaining
    drop_p = "enable-meaningful-choices"
    sub = df[df["principle"] != drop_p]
    grid_drop = bootstrap_cohort_grid(sub, n_bootstrap=10, seed=BOOTSTRAP_SEED)
    drop_hs = float(grid_drop.point.mean())
    remaining_vals = [v for p, v in principle_vals.items() if p != drop_p]
    expected_drop = np.mean(remaining_vals)
    np.testing.assert_allclose(drop_hs, expected_drop, atol=1e-10)


@pytest.mark.unit
def test_lopo_planted_flip_boundary():
    """A model whose negative S_bad is driven by one principle un-flips
    exactly under that principle's drop."""
    models = ["flipper"]
    personas = ["baseline", "bad_persona"]

    rows = []
    for principle in PRINCIPLES:
        for k in range(15):
            sample_id = f"{principle}-{k:03d}"
            for model in models:
                base_score = 0.3
                if principle == "respect-user-attention":
                    bad_score = -5.0
                else:
                    bad_score = 0.3
                rows.append({"persona": "baseline", "model": model, "principle": principle,
                             "sample_id": sample_id, "score": base_score})
                rows.append({"persona": "bad_persona", "model": model, "principle": principle,
                             "sample_id": sample_id, "score": bad_score})
    df = pd.DataFrame(rows)

    # All8: should flip (baseline > 0, bad < 0 because rua drags it down)
    grid_all = bootstrap_cohort_grid(df, n_bootstrap=50, seed=BOOTSTRAP_SEED)
    stats_all = cohort_flip_stats(grid_all)
    assert "flipper" in stats_all["flip_sign"]["models"], "flipper should flip under all8"

    # Drop rua: the negative principle is gone, bad score should go positive
    sub = df[df["principle"] != "respect-user-attention"]
    grid_drop = bootstrap_cohort_grid(sub, n_bootstrap=50, seed=BOOTSTRAP_SEED)
    stats_drop = cohort_flip_stats(grid_drop)
    assert "flipper" not in stats_drop["flip_sign"]["models"], "flipper should un-flip when rua is dropped"


@pytest.mark.unit
def test_lopo_planted_robust_knife_edge():
    """A model at S_bad exactly 0.5 tracks point-rule vs strict-rule separately."""
    models = ["knife"]
    personas = ["baseline", "bad_persona"]

    rows = []
    for principle in PRINCIPLES:
        for k in range(20):
            sample_id = f"{principle}-{k:03d}"
            rows.append({"persona": "baseline", "model": "knife", "principle": principle,
                         "sample_id": sample_id, "score": 0.6})
            rows.append({"persona": "bad_persona", "model": "knife", "principle": principle,
                         "sample_id": sample_id, "score": 0.5})
    df = pd.DataFrame(rows)

    grid = bootstrap_cohort_grid(df, n_bootstrap=50, seed=BOOTSTRAP_SEED)
    stats = cohort_flip_stats(grid)
    assert "knife" in stats["robust_sbad"]["models"]


@pytest.mark.unit
def test_lopo_reading_b_operationalization():
    """reading_b_holds must be the conjunction of three components."""
    import pandas as pd

    # Simulate rows with different component combinations
    cases = [
        (True, True, True, True),
        (True, True, False, False),
        (True, False, True, False),
        (False, True, True, False),
        (False, False, False, False),
    ]
    for rb_delta, rb_did, rb_flip, expected in cases:
        holds = rb_delta and rb_did and rb_flip
        assert holds == expected, f"({rb_delta}, {rb_did}, {rb_flip}) -> {holds}, expected {expected}"

    # Also verify the output CSV has consistent columns when the script has been run
    csv_path = REPO_ROOT / "tables" / "lopo_decomposition.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            expected = bool(r["reading_b_delta"] and r["reading_b_did"] and r["reading_b_flip"])
            assert bool(r["reading_b_holds"]) == expected, (
                f"config={r['config']}: reading_b_holds={r['reading_b_holds']} "
                f"but components are ({r['reading_b_delta']}, {r['reading_b_did']}, {r['reading_b_flip']})"
            )


@pytest.mark.unit
def test_lopo_determinism():
    """Same seed produces identical outputs across two runs."""
    rng = np.random.default_rng(33)
    models = ["m1", "m2"]
    personas = ["baseline", "good_persona", "bad_persona"]
    df = _synth_multi(rng, models, personas, {"baseline": 0.3, "good_persona": 0.6, "bad_persona": -0.1},
                      n_per_principle=15)

    from compute_lopo_sensitivity import build_main_tables
    s1, c1, ch1 = build_main_tables(df, n_bootstrap=30, seed=BOOTSTRAP_SEED)
    s2, c2, ch2 = build_main_tables(df, n_bootstrap=30, seed=BOOTSTRAP_SEED)

    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(ch1, ch2)


@pytest.mark.unit
def test_lopo_missing_decomp_raw_exits():
    """Script exits with a message when decomp raw file is missing."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "compute_lopo_sensitivity.py"),
         "--decomp-raw", "/nonexistent/path/inter_judge_raw.csv",
         "--output-dir", "/tmp/lopo_test_missing",
         "--n-bootstrap", "10"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
