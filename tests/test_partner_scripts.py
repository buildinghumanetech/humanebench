"""Unit tests for the partner curation/conversion/subset scripts (synthetic data only)."""
import argparse
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from curate_production_pairs import curate, is_synthetic, language_tag
from convert_partner_results import (
    convert_row,
    convert_row_joint,
    has_judgeable_response,
    split_sample_id,
)
from select_comparison_subset import select

pytestmark = pytest.mark.unit

SLUGS = [
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
]


def make_row(sample_id, user="What's 2+2?", assistant=None, severities=None):
    # Unique default response per row so the repeated-response stratum only
    # captures rows that deliberately share a response.
    assistant = assistant if assistant is not None else f"It's 4. (reply to {sample_id})"
    severities = severities or {}
    return {
        "sample_id": sample_id,
        "ts": "2026-01-01",
        "user": "u_1",
        "conv": f"c_{sample_id}",
        "turn_index": 1,
        "user_message": user,
        "assistant_response": assistant,
        "principles": {
            slug: {
                "severity": severities.get(slug, 0.5),
                "score": 0.75,
                "relevant": True,
                "reasoning": "fine",
            }
            for slug in SLUGS
        },
        "relevant_principles": SLUGS[:3],
        "overall_severity": 0.5,
        "mean_severity": 0.5,
    }


class TestCuration:
    def test_synthetic_detection(self):
        assert is_synthetic("Hello! This is a test message from automated testing.")
        assert not is_synthetic("can you remind me to test my code tomorrow")

    def test_dup_clusters_and_stats(self):
        rows = [
            make_row("a", user="hi", assistant="Hey!"),
            make_row("b", user="hi", assistant="Hey!"),
            make_row("c", user="hello", assistant="Hi there!"),
        ]
        stats = curate(rows)
        assert stats["dup_clusters"] == 1
        assert stats["dup_rows"] == 2
        clusters = {r["sample_id"]: r["curation"]["dup_cluster"] for r in rows}
        assert clusters["a"] == clusters["b"] is not None
        assert clusters["c"] is None

    def test_trivial_join(self):
        rows = [make_row("a"), make_row("b")]
        stats = curate(rows, trivial_by_id={"a": True})
        assert rows[0]["curation"]["trivial"] is True
        assert rows[1]["curation"]["trivial"] is None
        assert stats["trivial"] == 1
        assert stats["trivial_unknown"] == 1

    def test_language_tag(self):
        assert language_tag("hello there") == "en"
        assert language_tag("こんにちは、元気ですか") == "other"
        # extended-Latin scripts are Latin, not "other"
        assert language_tag("Xin chào, bạn khỏe không? Tiếng Việt đẹp") == "en"
        assert language_tag("¿Qué tal? ¡Muy bien, señor!") == "en"
        assert language_tag("12345 !!") == "en"

    def test_non_string_messages_tagged_not_crashed(self):
        rows = [make_row("a"), make_row("b")]
        rows[0]["user_message"] = None
        rows[1]["assistant_response"] = None
        stats = curate(rows)
        assert stats["bad_user_message"] == 1
        assert stats["bad_assistant_response"] == 1
        assert rows[0]["curation"]["synthetic_test"] is False


class TestConversion:
    def test_fanout_all_8(self):
        samples = convert_row(make_row("s_1"), "all")
        assert len(samples) == 8
        assert {s["target"] for s in samples} == set(SLUGS)
        for s in samples:
            assert s["metadata"]["ai_output"] == "It's 4. (reply to s_1)"
            assert s["id"] == f"s_1__{s['target']}"

    def test_fanout_relevant_only(self):
        samples = convert_row(make_row("s_1"), "relevant")
        assert len(samples) == 3
        assert {s["target"] for s in samples} == set(SLUGS[:3])

    def test_unknown_slug_rejected(self):
        row = make_row("s_1")
        row["principles"]["not-a-principle"] = row["principles"][SLUGS[0]]
        with pytest.raises(ValueError, match="s_1"):
            convert_row(row, "all")

    def test_unknown_relevant_slug_rejected(self):
        row = make_row("s_1")
        row["relevant_principles"] = [SLUGS[0], "be-transparent-honest"]
        with pytest.raises(ValueError, match="relevant_principles"):
            convert_row(row, "relevant")

    def test_empty_response_not_judgeable(self):
        assert has_judgeable_response(make_row("s_1"))
        for bad in ["", None, 42]:
            row = make_row("s_1")
            row["assistant_response"] = bad
            assert not has_judgeable_response(row)

    def test_joint_mode_one_sample_per_turn(self):
        row = make_row("s_1")
        sample = convert_row_joint(row)
        assert sample["id"] == "s_1"  # no principle suffix
        assert sample["target"] == ""
        assert sample["metadata"]["ai_output"] == row["assistant_response"]
        assert set(sample["metadata"]["orig_judgments"]) == set(SLUGS)

    def test_joint_mode_unknown_slug_rejected(self):
        # joint conversion must validate slugs like the per-principle path does
        row = make_row("s_1")
        row["principles"]["not-a-principle"] = row["principles"][SLUGS[0]]
        with pytest.raises(ValueError, match="s_1"):
            convert_row_joint(row)

    def test_id_roundtrip_with_double_underscore(self):
        for sid in ["s_1", "s_1__rep2", "weird__id__x"]:
            hb_id = f"{sid}__{SLUGS[0]}"
            assert split_sample_id(hb_id) == (sid, SLUGS[0])


def subset_args(**overrides):
    defaults = dict(
        seed=42,
        negative_sample=2,
        positive_sample=2,
        trivial_sample=2,
        positive_extreme_sample=2,
        repeat_slice=3,
        repeat_response_min=10,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestSubsetSelection:
    def make_pool(self):
        rows = []
        for i in range(5):  # worst stratum
            rows.append(make_row(f"w{i}", severities={SLUGS[0]: -1.0}))
        for i in range(10):  # negative stratum
            rows.append(make_row(f"n{i}", severities={SLUGS[1]: -0.5}))
        for i in range(10):  # positive (default severities are all +0.5)
            rows.append(make_row(f"p{i}"))
        for i in range(4):  # positive-extreme stratum (a +1.0, no -1.0)
            rows.append(make_row(f"x{i}", severities={SLUGS[2]: 1.0}))
        for i in range(12):  # repeated identical response
            rows.append(make_row(f"r{i}", user=f"u{i}", assistant="Hey! How's it going?"))
        rows.append(
            make_row("t0", user="Hello! This is a test message from automated testing.")
        )
        curate(rows, trivial_by_id={"p0": True, "p1": True, "p2": True})
        return rows

    def test_strata(self):
        out, stats, extras = select(self.make_pool(), subset_args())
        assert stats["worst"] == 5  # all -1.0 rows included
        assert stats["qa_test"] == 1
        assert stats["repeated"] == 12
        assert stats["positive_extreme"] == 2
        assert stats["negative"] == 2
        assert stats["trivial"] == 2
        assert stats["positive"] == 2
        assert stats["repeat"] == 3
        reps = [r for r in out if r["curation"]["subset_stratum"] == "repeat"]
        assert all(r["sample_id"].endswith("__rep2") for r in reps)
        assert all(r["curation"]["repeat_of"] + "__rep2" == r["sample_id"] for r in reps)

    def test_membership_flags_independent_of_stratum_label(self):
        # A QA row that ALSO qualifies for the worst stratum gets labeled
        # "worst" (priority), but its is_qa_test flag must still be true.
        rows = [
            make_row(
                "qa_worst",
                user="Hello! This is a test message from automated testing.",
                severities={SLUGS[0]: -1.0},
            ),
            make_row("plain"),
        ]
        curate(rows)
        out, stats, extras = select(rows, subset_args(repeat_slice=0))
        qa = next(r for r in out if r["sample_id"] == "qa_worst")
        assert qa["curation"]["subset_stratum"] == "worst"
        assert qa["curation"]["flags"]["is_qa_test"] is True
        assert qa["curation"]["flags"]["is_worst"] is True
        # severity flags partition: a worst row is not also negative
        assert qa["curation"]["flags"]["is_negative"] is False
        assert extras["manifest"]["selected_flags"]["is_qa_test"] == 1

    def test_mixed_extreme_row_is_negative_class(self):
        # A +1.0 alongside a -0.5 cell: negative-class, NOT a positive-tail
        # control (the negative tail is what the comparison targets).
        rows = [make_row("mixed", severities={SLUGS[0]: 1.0, SLUGS[1]: -0.5})]
        curate(rows)
        out, stats, extras = select(rows, subset_args(repeat_slice=0))
        mixed = out[0]
        assert mixed["curation"]["flags"]["is_positive_extreme"] is False
        assert mixed["curation"]["flags"]["is_negative"] is True
        assert mixed["curation"]["subset_stratum"] == "negative"
        assert extras["manifest"]["strata"]["positive_extreme"]["eligible"] == 0

    def test_manifest_records_pools_and_fractions(self):
        out, stats, extras = select(self.make_pool(), subset_args())
        m = extras["manifest"]
        assert m["strata"]["worst"]["sampling_fraction"] == 1.0
        neg = m["strata"]["negative"]
        assert neg["labeled"] == 2 and neg["eligible"] == 10 and neg["residual_pool"] == 10
        assert abs(neg["sampling_fraction"] - 0.2) < 1e-9
        assert m["population"] == len(self.make_pool())
        assert m["population_flags"]["is_worst"] == 5
        # unified schema: every stratum entry has the same keys
        for entry in m["strata"].values():
            assert set(entry) == {"eligible", "residual_pool", "labeled", "sampling_fraction"}

    def test_flag_totals_are_zero_safe(self):
        rows = [make_row("a"), make_row("b")]  # nothing trivial/worst/etc.
        curate(rows)
        out, stats, extras = select(rows, subset_args(repeat_slice=0))
        pf = extras["manifest"]["population_flags"]
        assert pf["is_trivial"] == 0
        assert pf["is_worst"] == 0
        assert pf["is_qa_test"] == 0

    def test_stratum_rng_streams_independent(self):
        # Resizing one stratum must not change another stratum's draws.
        pool = self.make_pool()
        out_a, _, _ = select(copy.deepcopy(pool), subset_args(positive_extreme_sample=0))
        out_b, _, _ = select(copy.deepcopy(pool), subset_args(positive_extreme_sample=2))

        def labeled(out, stratum):
            return sorted(
                r["sample_id"] for r in out
                if r["curation"]["subset_stratum"] == stratum
            )

        for stratum in ("negative", "trivial", "positive"):
            assert labeled(out_a, stratum) == labeled(out_b, stratum)

    def test_between_run_repeats_mirror_repeat_slice(self):
        out, stats, extras = select(self.make_pool(), subset_args())
        between = extras["between_run"]
        assert len(between) == stats["repeat"] == 3
        assert all(r["sample_id"].endswith("__rep3") for r in between)
        rep2_bases = {
            r["curation"]["repeat_of"]
            for r in out
            if r["curation"]["subset_stratum"] == "repeat"
        }
        rep3_bases = {r["curation"]["repeat_of"] for r in between}
        assert rep2_bases == rep3_bases

    def test_deterministic_for_seed(self):
        pool = self.make_pool()
        out1, _, _ = select(copy.deepcopy(pool), subset_args())
        out2, _, _ = select(copy.deepcopy(pool), subset_args())
        assert [r["sample_id"] for r in out1] == [r["sample_id"] for r in out2]

    def test_no_double_selection(self):
        out, _, _ = select(self.make_pool(), subset_args())
        non_repeat = [r["sample_id"] for r in out if not r["sample_id"].endswith("__rep2")]
        assert len(non_repeat) == len(set(non_repeat))

    def test_uncurated_input_rejected(self):
        rows = [make_row("a"), make_row("b")]  # no curate() pass
        with pytest.raises(SystemExit, match="curation"):
            select(rows, subset_args())

    def test_rows_without_judgments_excluded(self):
        pool = self.make_pool()
        bare = make_row("nojudge")
        bare["principles"] = {}
        pool.append(bare)
        curate([bare])
        out, stats, _ = select(pool, subset_args())
        assert stats["excluded_no_judgments"] == 1
        assert all(r["sample_id"] != "nojudge" for r in out)
