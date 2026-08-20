"""Unit tests for compare_panel_vs_single_judge.py (synthetic data only)."""
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compare_panel_vs_single_judge import (
    Cell,
    FHR_SLUG,
    between_run_reliability,
    contrast_fhr_deescalation,
    contrast_trivial_reproduction,
    holm,
    inverse_prob_weighted_mean,
    load_cells,
    per_principle_overstatement,
    per_stratum_tables,
    severity_dist,
    within_run_reliability,
    wilson_ci,
)

pytestmark = pytest.mark.unit

ALL_FLAGS_FALSE = {
    "is_worst": False, "is_negative": False, "is_positive_extreme": False,
    "is_positive": False, "is_qa_test": False, "is_trivial": False,
    "is_repeated_reply": False,
}


def make_cell(slug=FHR_SLUG, base_id="t1", repeat_kind=None, stratum="worst",
              flags=None, orig=-1.0, panel=None):
    f = dict(ALL_FLAGS_FALSE)
    if flags:
        f.update(flags)
    is_nan = panel is None
    return Cell(
        hb_id=f"{base_id}__{slug}", sample_id=base_id, base_id=base_id, slug=slug,
        repeat_kind=repeat_kind, stratum=stratum, flags=f,
        orig_severity=orig, orig_relevant=True,
        panel_scores=None if is_nan else list(panel), is_nan=is_nan,
    )


# --------------------------------------------------------------------------- #
# Ordinal median / dispersion
# --------------------------------------------------------------------------- #
def test_panel_median_is_ordinal_middle():
    assert make_cell(panel=[-1.0, -0.5, -0.5]).panel_median == -0.5
    assert make_cell(panel=[-1.0, -0.5, 0.5]).panel_median == -0.5
    assert make_cell(panel=[0.5, 1.0, 1.0]).panel_median == 1.0
    # median never interpolates off the scale
    assert make_cell(panel=[-1.0, 1.0, 0.5]).panel_median == 0.5


def test_unanimous_and_spread():
    c = make_cell(panel=[-0.5, -0.5, -0.5])
    assert c.unanimous is True and c.spread == 0.0
    c = make_cell(panel=[-1.0, -0.5, 1.0])
    assert c.unanimous is False and c.spread == 2.0


def test_nan_cell_has_no_stats():
    c = make_cell(panel=None)
    assert c.is_nan and c.panel_median is None and c.panel_mean is None


def test_severity_dist_zero_fills():
    d = severity_dist([-0.5, -0.5, 1.0])
    assert d == {-1.0: 0, -0.5: 2, 0.5: 0, 1.0: 1}


# --------------------------------------------------------------------------- #
# Holm / Wilson
# --------------------------------------------------------------------------- #
def test_holm_step_down():
    adj = holm({"a": 0.01, "b": 0.04})
    assert adj["a"] == pytest.approx(0.02)  # 2 * 0.01
    assert adj["b"] == pytest.approx(0.04)  # max(0.02, 1 * 0.04)


def test_holm_monotone_nondecreasing():
    adj = holm({"a": 0.30, "b": 0.02})
    # smaller raw p gets the bigger multiplier but result stays monotone
    assert adj["b"] <= adj["a"]


def test_holm_nan_does_not_poison_the_other_contrast():
    # an empty/uncomputable contrast (NaN p) must NOT force the valid one to 1.0
    adj = holm({"a": float("nan"), "b": 0.03})
    assert math.isnan(adj["a"])
    # only one finite p remains -> corrected over m=1, so it is unchanged
    assert adj["b"] == pytest.approx(0.03)


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(1, 1)
    assert 0.0 <= lo <= hi <= 1.0 and hi == pytest.approx(1.0)
    assert math.isnan(wilson_ci(0, 0)[0])


# --------------------------------------------------------------------------- #
# Confirmatory contrast (a): FHR de-escalation vs RTM control
# --------------------------------------------------------------------------- #
def test_contrast_a_asymmetric_correction_is_significant():
    cells = []
    # 9/10 FHR -1.0 cells de-escalated (panel median above the floor)
    for i in range(9):
        cells.append(make_cell(base_id=f"n{i}", slug=FHR_SLUG, stratum="worst",
                                flags={"is_worst": True}, orig=-1.0,
                                panel=[-0.5, -0.5, -1.0]))
    cells.append(make_cell(base_id="n9", slug=FHR_SLUG, stratum="worst",
                           flags={"is_worst": True}, orig=-1.0,
                           panel=[-1.0, -1.0, -1.0]))
    # 1/10 positive-extreme +1.0 control cells regressed (stable control)
    for i in range(9):
        cells.append(make_cell(base_id=f"p{i}", slug="respect-user-attention",
                               stratum="positive_extreme",
                               flags={"is_positive_extreme": True}, orig=1.0,
                               panel=[1.0, 1.0, 1.0]))
    cells.append(make_cell(base_id="p9", slug="respect-user-attention",
                           stratum="positive_extreme",
                           flags={"is_positive_extreme": True}, orig=1.0,
                           panel=[0.5, 0.5, 1.0]))
    r = contrast_fhr_deescalation(cells)
    assert r["fhr_n"] == 10 and r["fhr_deescalated"] == 9
    assert r["ctrl_n"] == 10 and r["ctrl_regressed"] == 1
    assert r["pvalue"] < 0.05  # asymmetry detected


def test_contrast_a_symmetric_softening_not_significant():
    # both tails soften equally -> RTM, should NOT be significant
    cells = []
    for i in range(10):
        deesc = i < 5
        cells.append(make_cell(base_id=f"n{i}", slug=FHR_SLUG, stratum="worst",
                               flags={"is_worst": True}, orig=-1.0,
                               panel=[-0.5, -0.5, -0.5] if deesc else [-1.0, -1.0, -1.0]))
    for i in range(10):
        reg = i < 5
        cells.append(make_cell(base_id=f"p{i}", slug="respect-user-attention",
                               stratum="positive_extreme",
                               flags={"is_positive_extreme": True}, orig=1.0,
                               panel=[0.5, 0.5, 0.5] if reg else [1.0, 1.0, 1.0]))
    r = contrast_fhr_deescalation(cells)
    assert r["fhr_deescalated"] == 5 and r["ctrl_regressed"] == 5
    assert r["pvalue"] > 0.05


def test_contrast_a_excludes_repeats_and_nan_and_nonfhr():
    cells = [
        make_cell(base_id="a", slug=FHR_SLUG, flags={"is_worst": True}, orig=-1.0,
                  panel=[-0.5, -0.5, -0.5]),
        # repeat copy must not double-count
        make_cell(base_id="a", slug=FHR_SLUG, repeat_kind="rep2",
                  flags={"is_worst": True}, orig=-1.0, panel=[-0.5, -0.5, -0.5]),
        # NaN excluded
        make_cell(base_id="b", slug=FHR_SLUG, flags={"is_worst": True}, orig=-1.0,
                  panel=None),
        # non-FHR excluded
        make_cell(base_id="c", slug="respect-user-attention",
                  flags={"is_worst": True}, orig=-1.0, panel=[-0.5, -0.5, -0.5]),
    ]
    r = contrast_fhr_deescalation(cells)
    assert r["fhr_n"] == 1


# --------------------------------------------------------------------------- #
# Confirmatory contrast (b): trivial -0.5 reproduction
# --------------------------------------------------------------------------- #
def test_contrast_b_high_reproduction_significant():
    cells = []
    for i in range(18):
        cells.append(make_cell(base_id=f"t{i}", slug="respect-user-attention",
                               stratum="trivial", flags={"is_trivial": True},
                               orig=-0.5, panel=[-0.5, -0.5, -0.5]))
    for i in range(2):
        cells.append(make_cell(base_id=f"u{i}", slug="respect-user-attention",
                               stratum="trivial", flags={"is_trivial": True},
                               orig=-0.5, panel=[0.5, 0.5, 0.5]))
    r = contrast_trivial_reproduction(cells)
    assert r["n"] == 20 and r["reproduced"] == 18
    assert r["pvalue"] < 0.05


def test_contrast_b_reproduction_counts_neg_one_as_reproduced():
    # a panel median of -1.0 still "reproduces" the negative flag (<= -0.5)
    cells = [make_cell(base_id="t", slug="respect-user-attention",
                       stratum="trivial", flags={"is_trivial": True},
                       orig=-0.5, panel=[-1.0, -1.0, -0.5])]
    r = contrast_trivial_reproduction(cells)
    assert r["reproduced"] == 1
    assert r["reproduced_exact_-0.5"] == 0  # median is -1.0, not exactly -0.5


# --------------------------------------------------------------------------- #
# Per-stratum tables: flags are the denominators; strata overlap
# --------------------------------------------------------------------------- #
def test_per_stratum_uses_flags_and_overlaps():
    # one cell is both worst AND qa_test -> counted in both strata
    cells = [
        make_cell(base_id="a", flags={"is_worst": True, "is_qa_test": True},
                  orig=-1.0, panel=[-0.5, -0.5, -0.5]),
        make_cell(base_id="b", flags={"is_worst": True}, orig=-1.0,
                  panel=[-1.0, -1.0, -1.0]),
    ]
    strata = per_stratum_tables(cells)
    assert strata["is_worst"]["n_cells"] == 2
    assert strata["is_qa_test"]["n_cells"] == 1  # overlap, not double subtracted


def test_per_stratum_nan_attrition_counted_not_scored():
    cells = [
        make_cell(base_id="a", flags={"is_worst": True}, orig=-1.0, panel=None),
        make_cell(base_id="b", flags={"is_worst": True}, orig=-1.0,
                  panel=[-0.5, -0.5, -0.5]),
    ]
    s = per_stratum_tables(cells)["is_worst"]
    assert s["n_cells"] == 2 and s["n_nan"] == 1
    assert s["nan_rate"] == pytest.approx(0.5)
    # only the scored cell appears in the distribution
    assert sum(s["panel_median_dist"].values()) == 1


def test_per_stratum_shift_direction():
    # orig -1.0, panel median -0.5 -> de-escalated (less severe)
    cells = [make_cell(base_id="a", flags={"is_worst": True}, orig=-1.0,
                       panel=[-0.5, -0.5, -0.5])]
    s = per_stratum_tables(cells)["is_worst"]
    assert s["shift"].get("de-escalated") == 1


def test_overstatement_per_principle():
    cells = [
        make_cell(base_id="a", slug=FHR_SLUG, orig=-1.0, panel=[-0.5, -0.5, -0.5]),
        make_cell(base_id="b", slug=FHR_SLUG, orig=-1.0, panel=[-1.0, -1.0, -1.0]),
        make_cell(base_id="c", slug=FHR_SLUG, orig=-0.5, panel=[-0.5, -0.5, -0.5]),
    ]
    o = per_principle_overstatement(cells)[FHR_SLUG]
    assert o["n_orig_-1.0"] == 2 and o["de-escalated"] == 1


# --------------------------------------------------------------------------- #
# Reliability: base <-> rep2 pairing
# --------------------------------------------------------------------------- #
def test_within_run_reliability_pairs_by_base_and_slug():
    cells = [
        make_cell(base_id="t1", slug=FHR_SLUG, panel=[-0.5, -0.5, -0.5]),
        make_cell(base_id="t1", slug=FHR_SLUG, repeat_kind="rep2",
                  panel=[-0.5, -0.5, -1.0]),  # median still -0.5 -> exact match
        make_cell(base_id="t2", slug=FHR_SLUG, panel=[0.5, 0.5, 0.5]),
        make_cell(base_id="t2", slug=FHR_SLUG, repeat_kind="rep2",
                  panel=[-0.5, -0.5, -0.5]),  # median -0.5 vs 0.5 -> mismatch
    ]
    r = within_run_reliability(cells)
    assert r["n_pairs"] == 2 and r["exact_median_match"] == 1
    assert r["mean_abs_median_diff"] == pytest.approx(0.5)


def test_between_run_filters_rep3_and_ignores_base_cells():
    base = [make_cell(base_id="t1", slug=FHR_SLUG, panel=[-0.5, -0.5, -0.5])]
    # a mis-supplied log full of BASE cells must not pair against itself
    wrong_log = [make_cell(base_id="t1", slug=FHR_SLUG, panel=[-0.5, -0.5, -0.5])]
    assert between_run_reliability(base, wrong_log)["n_pairs"] == 0
    # a correct rep3 log pairs and measures drift
    rep3 = [make_cell(base_id="t1", slug=FHR_SLUG, repeat_kind="rep3",
                      panel=[0.5, 0.5, 0.5])]
    r = between_run_reliability(base, rep3)
    assert r["n_pairs"] == 1 and r["exact_median_match"] == 0
    assert r["mean_abs_median_diff"] == pytest.approx(1.0)


def test_per_stratum_orig_off_scale_reconciles():
    cells = [
        make_cell(base_id="a", flags={"is_worst": True}, orig=-1.0,
                  panel=[-0.5, -0.5, -0.5]),
        make_cell(base_id="b", flags={"is_worst": True}, orig=0.0,  # off-scale
                  panel=[-0.5, -0.5, -0.5]),
        make_cell(base_id="c", flags={"is_worst": True}, orig=None,  # no orig
                  panel=[-0.5, -0.5, -0.5]),
    ]
    s = per_stratum_tables(cells)["is_worst"]
    assert s["orig_off_scale"] == 1 and s["orig_missing"] == 1
    # on-scale dist + off-scale + missing reconcile with scored-cell count (3)
    assert sum(s["orig_dist"].values()) + s["orig_off_scale"] + s["orig_missing"] == 3


def test_ipw_mean_weights_by_fraction():
    manifest = {"strata": {
        "worst": {"sampling_fraction": 1.0},
        "negative": {"sampling_fraction": 0.1},
    }}
    cells = [
        make_cell(base_id="a", stratum="worst", panel=[-1.0, -1.0, -1.0]),
        make_cell(base_id="b", stratum="negative", panel=[0.5, 0.5, 0.5]),
    ]
    # negative cell weighted 10x -> pooled mean pulled toward +0.5
    r = inverse_prob_weighted_mean(cells, manifest)
    assert r["mean"] == pytest.approx((1 * -1.0 + 10 * 0.5) / 11)
    assert r["n_used"] == 2 and r["n_dropped"] == 0


def test_ipw_counts_dropped_cells_not_in_manifest():
    manifest = {"strata": {"worst": {"sampling_fraction": 1.0}}}
    cells = [
        make_cell(base_id="a", stratum="worst", panel=[-1.0, -1.0, -1.0]),
        make_cell(base_id="b", stratum="mystery", panel=[0.5, 0.5, 0.5]),
        make_cell(base_id="c", stratum=None, panel=[0.5, 0.5, 0.5]),
    ]
    r = inverse_prob_weighted_mean(cells, manifest)
    # only the 'worst' cell is used; the other two are dropped and counted
    assert r["mean"] == pytest.approx(-1.0)
    assert r["n_used"] == 1 and r["n_dropped"] == 2
    assert r["dropped_strata"] == {"mystery": 1, None: 1}


# --------------------------------------------------------------------------- #
# load_cells: parse the real .eval shape (nested metadata, NaN, repeats, flags)
# --------------------------------------------------------------------------- #
def _write_eval(path, samples):
    with zipfile.ZipFile(path, "w") as z:
        for s in samples:
            z.writestr(f"samples/{s['id']}_epoch_1.json", json.dumps(s))


def _sample(hb_id, slug, individual, orig_sev, flags=None, value=None,
            repeat_of=None, stratum="worst"):
    cur = {"subset_stratum": stratum}
    if flags is not None:
        cur["flags"] = flags
    if repeat_of is not None:
        cur["repeat_of"] = repeat_of
    score_md = {"ensemble_models": ["m1", "m2", "m3"]}
    if individual is not None:
        score_md["individual_scores"] = individual
        v = value if value is not None else sum(individual) / len(individual)
    else:
        v = math.nan  # strict-ensemble failure
    return {
        "id": hb_id,
        "target": slug,
        "metadata": {
            "used_pregenerated_output": True,
            "metadata": {
                "ai_output": "resp",
                "curation": cur,
                "orig_judgment": None if orig_sev is None else {
                    "severity": orig_sev, "relevant": True, "reasoning": "r"},
            },
        },
        "scores": {"overseer": {"value": v, "answer": slug, "metadata": score_md}},
    }


def test_load_cells_parses_nested_metadata(tmp_path):
    flags = dict(ALL_FLAGS_FALSE, is_worst=True)
    p = tmp_path / "b.eval"
    _write_eval(p, [
        _sample("s_1__foster-healthy-relationships", FHR_SLUG,
                [-0.5, -0.5, -1.0], -1.0, flags=flags),
    ])
    cells = load_cells(p)
    assert len(cells) == 1
    c = cells[0]
    assert c.slug == FHR_SLUG and c.base_id == "s_1"
    assert c.orig_severity == -1.0
    assert c.panel_scores == [-0.5, -0.5, -1.0]
    assert c.panel_median == -0.5
    assert c.flags["is_worst"] is True
    assert c.is_nan is False


def test_load_cells_detects_nan(tmp_path):
    p = tmp_path / "b.eval"
    _write_eval(p, [
        _sample("s_1__respect-user-attention", "respect-user-attention",
                None, -0.5),  # no individual_scores -> NaN
    ])
    c = load_cells(p)[0]
    assert c.is_nan is True and c.panel_scores is None


def test_load_cells_repeat_via_repeat_of(tmp_path):
    p = tmp_path / "b.eval"
    _write_eval(p, [
        _sample("s_1__rep2__foster-healthy-relationships", FHR_SLUG,
                [-0.5, -0.5, -0.5], -1.0, repeat_of="s_1"),
    ])
    c = load_cells(p)[0]
    assert c.repeat_kind == "rep2"
    assert c.base_id == "s_1"  # from repeat_of, not the __rep2 sample_id


def test_load_cells_skips_id_without_separator(tmp_path):
    # a malformed id lacking '__' must be skipped, not crash the whole run
    p = tmp_path / "b.eval"
    _write_eval(p, [
        _sample("s_1__foster-healthy-relationships", FHR_SLUG,
                [-0.5, -0.5, -0.5], -1.0),
        _sample("malformedid", "malformedid", [-0.5, -0.5, -0.5], -0.5),
    ])
    cells = load_cells(p)
    assert len(cells) == 1
    assert cells[0].base_id == "s_1"


def test_load_cells_slug_with_double_underscore_safe(tmp_path):
    # rsplit on the LAST __ recovers the slug even when sample_id has __
    p = tmp_path / "b.eval"
    _write_eval(p, [
        _sample("s_1__rep2__be-transparent-and-honest", "be-transparent-and-honest",
                [0.5, 0.5, 0.5], 0.5, repeat_of="s_1"),
    ])
    c = load_cells(p)[0]
    assert c.slug == "be-transparent-and-honest" and c.base_id == "s_1"
