"""Unit tests for compare_joint_vs_decomposed.py (synthetic data only)."""
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compare_joint_vs_decomposed import (
    coflag_flags,
    first_eigen_share,
    load_joint_arms,
    mcnemar_exact,
    paired_shift,
    position_bias,
    severity_matrix,
    spearman_offdiag,
    wilson_ci,
)

pytestmark = pytest.mark.unit

SLUGS = [f"p{i}" for i in range(8)]


def row(sevs):
    return dict(zip(SLUGS, sevs))


# --------------------------------------------------------------------------- #
# Co-flag rate
# --------------------------------------------------------------------------- #
def test_coflag_counts_negatives_at_threshold():
    scores = {
        "t1": row([-0.5] * 4 + [0.5] * 4),   # exactly 4 negative -> flagged
        "t2": row([-1.0] * 3 + [0.5] * 5),   # 3 negative -> not flagged
        "t3": row([-1.0] * 8),               # all negative -> flagged
    }
    flags = coflag_flags(scores, SLUGS, threshold=4)
    assert flags == {"t1": True, "t2": False, "t3": True}


def test_coflag_nan_decidability():
    # 3 negatives + 2 NaN with threshold 4: could be 3, 4, or 5 -> undecidable
    scores = {"t1": row([-0.5] * 3 + [math.nan] * 2 + [0.5] * 3)}
    assert coflag_flags(scores, SLUGS, threshold=4)["t1"] is None
    # 4 negatives + NaN: already >= threshold -> True regardless of the NaN
    scores = {"t2": row([-0.5] * 4 + [math.nan] + [0.5] * 3)}
    assert coflag_flags(scores, SLUGS, threshold=4)["t2"] is True
    # 1 negative + 1 NaN, threshold 4: even if NaN were negative -> max 2 -> False
    scores = {"t3": row([-0.5] + [math.nan] + [0.5] * 6)}
    assert coflag_flags(scores, SLUGS, threshold=4)["t3"] is False


def test_mcnemar_counts_discordant_and_excludes_undecidable():
    x = {"t1": True, "t2": True, "t3": False, "t4": None, "t5": True}
    y = {"t1": True, "t2": False, "t3": True, "t4": True, "t5": None}
    m = mcnemar_exact(x, y)
    assert m["n_pairs"] == 3          # t4, t5 excluded
    assert m["x_only"] == 1           # t2
    assert m["y_only"] == 1           # t3
    assert m["pvalue"] == pytest.approx(1.0)


def test_mcnemar_detects_one_sided_discordance():
    x = {f"t{i}": True for i in range(12)}
    y = {f"t{i}": False for i in range(12)}
    m = mcnemar_exact(x, y)
    assert m["x_only"] == 12 and m["y_only"] == 0
    assert m["pvalue"] < 0.001


def test_wilson_ci_sane():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi


# --------------------------------------------------------------------------- #
# Inter-principle correlation
# --------------------------------------------------------------------------- #
def test_offdiag_rho_high_when_principles_move_together():
    # every principle = same underlying signal -> rho ~ 1 everywhere
    rng = np.random.default_rng(0)
    base = rng.choice([-1.0, -0.5, 0.5, 1.0], size=50)
    scores = {f"t{i}": row([base[i]] * 8) for i in range(50)}
    mat = severity_matrix(scores, SLUGS, sorted(scores))
    mean_rho, rho, n_pairs = spearman_offdiag(mat)
    assert n_pairs == 28
    assert mean_rho == pytest.approx(1.0)


def test_offdiag_rho_near_zero_when_independent():
    rng = np.random.default_rng(1)
    scores = {
        f"t{i}": row(rng.choice([-1.0, -0.5, 0.5, 1.0], size=8))
        for i in range(400)
    }
    mat = severity_matrix(scores, SLUGS, sorted(scores))
    mean_rho, _, _ = spearman_offdiag(mat)
    assert abs(mean_rho) < 0.1


def test_offdiag_skips_constant_columns_and_nan_rows():
    # p0 constant (zero variance) -> its 7 pairs skipped, not counted as 0
    rng = np.random.default_rng(2)
    scores = {}
    for i in range(30):
        sevs = list(rng.choice([-1.0, 0.5], size=8))
        sevs[0] = -0.5
        if i == 0:
            sevs[1] = math.nan  # one NaN cell -> pairwise-complete handling
        scores[f"t{i}"] = row(sevs)
    mat = severity_matrix(scores, SLUGS, sorted(scores))
    _, rho, n_pairs = spearman_offdiag(mat)
    assert n_pairs == 21  # 28 - 7 pairs involving the constant column
    assert math.isnan(rho[0, 1])


def test_first_eigen_share_bounds():
    # perfectly correlated -> share ~ 1; identity-like -> share ~ 1/8
    ones = np.ones((8, 8))
    assert first_eigen_share(ones) == pytest.approx(1.0)
    eye = np.eye(8)
    assert first_eigen_share(eye) == pytest.approx(1 / 8)


# --------------------------------------------------------------------------- #
# Paired shift and position bias
# --------------------------------------------------------------------------- #
def test_paired_shift_direction_and_nan_exclusion():
    c = {"t1": row([0.5] * 8), "t2": row([-0.5] * 8)}
    b = {"t1": row([-0.5] * 8), "t2": row([-0.5] * 7 + [math.nan])}
    shifts = paired_shift(c, b, SLUGS)
    assert shifts[SLUGS[0]]["n"] == 2
    assert shifts[SLUGS[0]]["mean_delta"] == pytest.approx((1.0 + 0.0) / 2)
    assert shifts[SLUGS[0]]["joint_milder"] == 1
    assert shifts[SLUGS[7]]["n"] == 1  # NaN pair excluded


def test_position_bias_flat_when_no_bias():
    # same severities regardless of order -> centered means ~0 at every position
    rng = np.random.default_rng(3)
    scores, orders = {}, {}
    for i in range(200):
        scores[f"t{i}"] = row(rng.choice([-0.5, 0.5], size=8))
        order = list(SLUGS)
        rng.shuffle(order)
        orders[f"t{i}"] = order
    pb = position_bias(scores, orders, SLUGS)
    for pos, (mean, n) in pb["mean_centered_by_position"].items():
        assert abs(mean) < 0.15
    assert abs(pb["spearman_position_vs_centered"]["rho"]) < 0.05


# --------------------------------------------------------------------------- #
# load_joint_arms hardening
# --------------------------------------------------------------------------- #
def _joint_sample(sid, per_judge, orig_sev=0.5):
    """A minimal joint-log sample. per_judge: list of {slug: severity} dicts."""
    slugs = sorted({k for j in per_judge for k in j})
    return {
        "id": sid,
        "scores": {"joint_overseer": {
            "value": {s: 0.0 for s in slugs},  # loader takes slug set from here
            "metadata": {
                "individual_scores": per_judge,
                "principle_order": slugs,
            },
        }},
        "metadata": {"metadata": {"orig_judgments": {
            s: {"severity": orig_sev} for s in slugs
        }}},
    }


def _write_joint_eval(path, samples):
    with zipfile.ZipFile(path, "w") as z:
        for s in samples:
            z.writestr(f"samples/{s['id']}_epoch_1.json", json.dumps(s))


def test_loader_uses_dynamic_judge_count(tmp_path):
    # a 2-judge panel must NOT be all-NaN'd by a hardcoded 3-judge rule
    p = tmp_path / "c.eval"
    _write_joint_eval(p, [
        _joint_sample("t1", [{"p0": -0.5, "p1": 0.5}, {"p0": -0.5, "p1": 0.5}]),
    ])
    c, a, orders, slugs = load_joint_arms(p)
    assert c["t1"]["p0"] == -0.5 and c["t1"]["p1"] == 0.5


def test_loader_strict_per_slug_any_judge_nan(tmp_path):
    # one judge NaN on one slug -> only that slug NaN, others survive
    p = tmp_path / "c.eval"
    _write_joint_eval(p, [
        _joint_sample("t1", [
            {"p0": -0.5, "p1": 0.5},
            {"p0": math.nan, "p1": 0.5},
            {"p0": -0.5, "p1": 0.5},
        ]),
    ])
    c, *_ = load_joint_arms(p)
    assert math.isnan(c["t1"]["p0"])
    assert c["t1"]["p1"] == 0.5


def test_loader_slug_union_not_first_sample(tmp_path):
    # first sample missing p1 -> slug set still includes it (union), with NaN
    p = tmp_path / "c.eval"
    s1 = _joint_sample("t1", [{"p0": -0.5}, {"p0": -0.5}, {"p0": -0.5}])
    s2 = _joint_sample("t2", [{"p0": 0.5, "p1": 0.5}] * 3)
    _write_joint_eval(p, [s1, s2])
    c, a, orders, slugs = load_joint_arms(p)
    assert slugs == ["p0", "p1"]
    assert math.isnan(c["t1"]["p1"])  # unscored in t1
    assert c["t2"]["p1"] == 0.5


def test_loader_excludes_repeat_ids_and_coerces_str(tmp_path):
    p = tmp_path / "c.eval"
    _write_joint_eval(p, [
        _joint_sample("t1", [{"p0": 0.5}] * 3),
        _joint_sample("t1__rep2", [{"p0": 0.5}] * 3),
        _joint_sample(123, [{"p0": 0.5}] * 3),  # int id -> "123"
    ])
    c, *_ = load_joint_arms(p)
    assert set(c) == {"t1", "123"}


def test_position_bias_detects_late_position_penalty():
    # inject: severity drops 0.5 for principles in positions 5-8
    scores, orders = {}, {}
    rng = np.random.default_rng(4)
    for i in range(200):
        order = list(SLUGS)
        rng.shuffle(order)
        orders[f"t{i}"] = order
        sevs = {}
        for pos, slug in enumerate(order, start=1):
            sevs[slug] = 0.5 if pos <= 4 else -0.5
        scores[f"t{i}"] = sevs
    pb = position_bias(scores, orders, SLUGS)
    early = pb["mean_centered_by_position"][1][0]
    late = pb["mean_centered_by_position"][8][0]
    assert early > 0.3 and late < -0.3
    assert pb["spearman_position_vs_centered"]["rho"] < -0.5
