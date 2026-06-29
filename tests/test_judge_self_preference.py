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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
