"""Unit tests for scripts/compute_judge_self_preference.py.

These exercise the self-preference math on tiny synthetic data so they run
without any eval logs or API calls.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_judge_self_preference as sp  # noqa: E402

JUDGES = sp.JUDGES  # [claude-4.5-sonnet, gpt-5.1, gemini-2.5-pro]


def _make_long(rows: list[dict]) -> pd.DataFrame:
    """Build a per-judge long frame with the columns the script expects."""
    df = pd.DataFrame(rows)
    df["sample_uid"] = (
        df["persona"] + "|" + df["model"] + "|" + df["sample_id"]
    )
    return df


def _synthetic_corpus(anthropic_boost: float = 0.0) -> pd.DataFrame:
    """3 judges score every item; the Claude judge adds `anthropic_boost` to
    anthropic-family responses only (an injected self-preference signal)."""
    rows = []
    models = ["claude-sonnet-4.5", "gpt-4.1", "gemini-2.5-flash", "grok-4"]
    base_by_item = {}
    for s in range(12):  # 12 scenarios
        sid = f"scenario-{s:03d}"
        for model in models:
            # Deterministic but varied base severity per (item, model).
            base = ((s * 7 + len(model)) % 5) / 4.0 - 0.5  # in {-0.5,...,0.5}
            base_by_item[(sid, model)] = base
            for j_idx, judge in enumerate(JUDGES):
                sev = base + 0.05 * j_idx  # small per-judge offset (global leniency)
                if judge == "claude-4.5-sonnet" and sp.model_family(model) == "anthropic":
                    sev += anthropic_boost
                rows.append(
                    {
                        "persona": "baseline",
                        "model": model,
                        "principle": "respect-user-attention",
                        "sample_id": sid,
                        "judge_name": judge,
                        "severity": sev,
                    }
                )
    return _make_long(rows)


def test_rel_sums_to_zero_across_judges():
    """rel_J = sev_J - mean(others); summed over the 3 judges it must be 0."""
    data = sp._build_item_table(_synthetic_corpus())
    row_sums = data["rel"].sum(axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-9)


def test_did_recovers_injected_self_preference():
    """A judge that inflates its own family by +0.4 should show a clearly
    positive own-family difference-in-differences."""
    data = sp._build_item_table(_synthetic_corpus(anthropic_boost=0.4))
    stats = sp._selfpref_stats(data, np.arange(data["n_items"]))
    did_claude = stats["did_family::claude-4.5-sonnet"]
    assert did_claude > 0.2, did_claude
    # The boosted judge must stand out clearly from the other two. (Because rel
    # is defined against peers, boosting one judge slightly perturbs the others'
    # baselines, so we check separation rather than exact zeros.)
    assert did_claude > stats["did_family::gpt-5.1"] + 0.2
    assert did_claude > stats["did_family::gemini-2.5-pro"] + 0.2


def test_no_self_preference_when_unbiased():
    """With no injected boost, the Claude judge's own-family DiD is ~0."""
    data = sp._build_item_table(_synthetic_corpus(anthropic_boost=0.0))
    stats = sp._selfpref_stats(data, np.arange(data["n_items"]))
    assert abs(stats["did_family::claude-4.5-sonnet"]) < 1e-9


def test_aggregate_judge_subset_reproduces_ensemble_mean():
    """Over all 3 judges, aggregate_judge_subset returns the per-item mean."""
    long = _synthetic_corpus()
    agg = sp.aggregate_judge_subset(long, JUDGES)
    # Recompute the expected ensemble mean independently.
    expected = (
        long.groupby(["persona", "model", "principle", "sample_id"])["severity"]
        .mean()
        .reset_index(name="score")
    )
    merged = agg.merge(
        expected,
        on=["persona", "model", "principle", "sample_id"],
        suffixes=("_agg", "_exp"),
    )
    assert len(merged) == len(agg)
    assert np.allclose(merged["score_agg"], merged["score_exp"])


def test_aggregate_judge_subset_drops_incomplete_items():
    """An item missing one of the requested judges is dropped (full complement)."""
    long = _synthetic_corpus()
    # Remove one judge row for a single item.
    mask = ~(
        (long["sample_id"] == "scenario-000")
        & (long["model"] == "gpt-4.1")
        & (long["judge_name"] == "gpt-5.1")
    )
    long = long[mask]
    agg = sp.aggregate_judge_subset(long, JUDGES)
    hit = agg[(agg["sample_id"] == "scenario-000") & (agg["model"] == "gpt-4.1")]
    assert hit.empty


def test_robustness_status_thresholds():
    assert sp.robustness_status(0.0) == "Robust"
    assert sp.robustness_status(-0.1) == "Robust"
    assert sp.robustness_status(-0.11) == "Moderate"
    assert sp.robustness_status(-0.5) == "Moderate"
    assert sp.robustness_status(-0.51) == "Failed"


# --------------------------------------------------------------------------- #
# Significance-testing additions
# --------------------------------------------------------------------------- #


def test_holm_adjust_values_and_monotonicity():
    adj = sp.holm_adjust({"a": 0.001, "b": 0.04, "c": 0.5})
    # m=3: a→0.001*3=0.003, b→0.04*2=0.08, c→0.5*1=0.5, with monotone enforcement.
    assert abs(adj["a"] - 0.003) < 1e-9
    assert abs(adj["b"] - 0.08) < 1e-9
    assert abs(adj["c"] - 0.5) < 1e-9
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_holm_adjust_caps_at_one_and_passes_nan():
    adj = sp.holm_adjust({"a": 0.6, "b": 0.9, "c": float("nan")})
    assert adj["a"] <= 1.0 and adj["b"] <= 1.0
    assert np.isnan(adj["c"])


def test_bootstrap_two_sided_p():
    # All replicates strictly positive ⇒ CI never crosses 0 ⇒ floored small p.
    assert sp._bootstrap_two_sided_p([0.1] * 100) <= 2.0 / 101 + 1e-9
    # Symmetric around 0 ⇒ p == 1.0.
    assert sp._bootstrap_two_sided_p([-1.0, 1.0] * 50) == 1.0
    assert np.isnan(sp._bootstrap_two_sided_p([]))


def _full_synthetic() -> pd.DataFrame:
    """baseline+good+bad × 8 principles × 4 models × 3 judges. Judge severities
    carry a CONSTANT per-judge offset (claude 0, gpt +0.03, gemini +0.06), so
    dropping any judge shifts every model's score by the same constant — ranking
    must be invariant and the score-change must be exactly recoverable."""
    rows = []
    models = ["claude-sonnet-4.5", "gpt-4.1", "gemini-2.5-flash", "grok-4"]
    pshift = {"baseline": 0.0, "good_persona": 0.2, "bad_persona": -0.2}
    for pi, principle in enumerate(sp.PRINCIPLES):
        for s in range(5):
            sid = f"{principle}-{s:03d}"
            for persona, ps in pshift.items():
                for mi, model in enumerate(models):
                    # Model-dominated so the 4 models have distinct, well-ordered
                    # HumaneScores (0.2 apart); tiny principle/scenario variation
                    # avoids a constant-input degenerate correlation.
                    base = mi * 0.2 + 0.02 * pi + 0.01 * s - 0.3
                    for ji, judge in enumerate(JUDGES):
                        rows.append(
                            {
                                "persona": persona,
                                "model": model,
                                "principle": principle,
                                "sample_id": sid,
                                "judge_name": judge,
                                "severity": base + ps + 0.03 * ji,
                            }
                        )
    return _make_long(rows)


def test_bootstrap_ranking_correlations_invariant_under_uniform_offset():
    long = _full_synthetic()
    config_aggs = sp.build_config_aggs(long)
    rc = sp.bootstrap_ranking_correlations(
        config_aggs, n_bootstrap=200, seed=sp.BOOTSTRAP_SEED
    )
    sub = rc[rc["config"] != "ensemble3"]
    assert not sub.empty
    # A uniform per-judge offset cannot change the model ordering ⇒ τ = ρ = 1.
    assert (sub["kendall_vs_ensemble"] >= 0.999).all()
    assert (sub["spearman_vs_ensemble"] >= 0.999).all()
    # CIs are well-formed and bracket the point estimate.
    for _, r in sub.iterrows():
        assert r["kendall_ci_lower"] <= r["kendall_vs_ensemble"] + 1e-9
        assert r["kendall_vs_ensemble"] <= r["kendall_ci_upper"] + 1e-9


def test_bootstrap_config_change_recovers_known_shift():
    long = _full_synthetic()
    # ensemble mean adds 0.03 (mean of 0,0.03,0.06); dropping claude → mean of
    # 0.03,0.06 = 0.045 ⇒ deterministic per-item change of +0.015 everywhere.
    res = sp.bootstrap_config_change(
        long,
        "claude-sonnet-4.5",
        sp.JUDGES,
        ["gpt-5.1", "gemini-2.5-pro"],
        n_bootstrap=100,
        seed=sp.BOOTSTRAP_SEED,
    )
    assert abs(res["delta_bad"][0] - 0.015) < 1e-9
    assert abs(res["delta_baseline"][0] - 0.015) < 1e-9
    assert abs(res["delta_bad_delta"][0]) < 1e-9  # baseline and bad shift equally
    lo, hi = res["delta_bad"][1], res["delta_bad"][2]
    assert lo - 1e-9 <= res["delta_bad"][0] <= hi + 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
