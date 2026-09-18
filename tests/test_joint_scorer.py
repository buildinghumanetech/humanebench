"""Unit tests for the joint (all-principles-in-one-call) scorer's pure parts."""
import json
import math

import pytest

from humanebench.humane_patterns import HUMANE_PATTERNS
from humanebench.joint_scorer import (
    PRINCIPLE_ORDER_SALT,
    _coerce_invalid,
    build_joint_prompt,
    joint_pattern_score,
    parse_joint_response,
    principle_order_for,
    strict_ensemble_mean,
)
from humanebench.scorer import InvalidOverseerResponse

pytestmark = pytest.mark.unit

SLUGS = list(HUMANE_PATTERNS.keys())


def joint_payload(severity=0.5, invalid=False):
    payload = {
        slug: {"severity": severity, "reasoning": "fine"} for slug in SLUGS
    }
    payload["invalid"] = invalid
    return payload


class TestPrompt:
    def test_contains_all_slugs_and_conversation(self):
        prompt = build_joint_prompt("What's 2+2?", "It's 4.")
        for slug in SLUGS:
            assert slug in prompt
        assert "What's 2+2?" in prompt
        assert "It's 4." in prompt
        # the global rules travel with the joint prompt too
        assert "fails to meaningfully meet the principle" in prompt


class TestPrincipleOrder:
    def test_is_a_full_permutation(self):
        order = principle_order_for("s_1")
        assert sorted(p.id for p in order) == sorted(SLUGS)

    def test_deterministic_per_sample_id(self):
        # same id -> same order across calls (reproducible across runs)
        a = [p.id for p in principle_order_for("s_abc")]
        b = [p.id for p in principle_order_for("s_abc")]
        assert a == b

    def test_randomizes_across_ids(self):
        # different turns get different orders (not all identical)
        orders = {
            tuple(p.id for p in principle_order_for(f"s_{i}"))
            for i in range(20)
        }
        assert len(orders) > 1

    def test_salt_changes_order(self):
        a = [p.id for p in principle_order_for("s_1", salt="A")]
        b = [p.id for p in principle_order_for("s_1", salt="B")]
        assert a != b or len(SLUGS) == 1

    def test_prompt_follows_given_order(self):
        order = list(reversed(principle_order_for("s_1")))
        prompt = build_joint_prompt("hi", "hello", order)
        positions = [prompt.index(f"(id: {p.id})") for p in order]
        assert positions == sorted(positions)  # prompt lists them in `order`

    def test_default_salt_is_used(self):
        a = [p.id for p in principle_order_for("s_1")]
        b = [p.id for p in principle_order_for("s_1", salt=PRINCIPLE_ORDER_SALT)]
        assert a == b


class TestInvalidCoercion:
    def test_real_booleans(self):
        assert _coerce_invalid(True) is True
        assert _coerce_invalid(False) is False

    def test_string_false_is_false(self):
        # the bug: raw bool("false") is True; a JSON string "false" must be False
        assert _coerce_invalid("false") is False
        assert _coerce_invalid("no") is False
        assert _coerce_invalid("0") is False

    def test_string_true_is_true(self):
        assert _coerce_invalid("true") is True
        assert _coerce_invalid("True") is True
        assert _coerce_invalid("yes") is True

    def test_numbers_and_unknowns(self):
        assert _coerce_invalid(1) is True
        assert _coerce_invalid(0) is False
        assert _coerce_invalid("maybe") is False  # safe default: not invalid


class TestParsing:
    def test_valid_payload(self):
        evals, invalid = parse_joint_response(json.dumps(joint_payload()))
        assert set(evals) == set(SLUGS)
        assert evals[SLUGS[0]].severity == 0.5
        assert invalid is False

    def test_invalid_flag_carried(self):
        _, invalid = parse_joint_response(json.dumps(joint_payload(invalid=True)))
        assert invalid is True

    def test_string_false_invalid_flag_not_triggered(self):
        payload = joint_payload()
        payload["invalid"] = "false"  # JSON string, must NOT flip to invalid
        _, invalid = parse_joint_response(json.dumps(payload))
        assert invalid is False

    def test_missing_principle_omitted_not_rejected(self):
        # tolerant: a missing principle is omitted, the other 7 survive
        payload = joint_payload()
        del payload[SLUGS[3]]
        evals, _ = parse_joint_response(json.dumps(payload))
        assert SLUGS[3] not in evals
        assert len(evals) == len(SLUGS) - 1

    def test_out_of_scale_severity_omitted_not_rejected(self):
        # tolerant: one off-scale severity voids only that slug, not the turn
        payload = joint_payload()
        payload[SLUGS[0]]["severity"] = 0.0
        evals, _ = parse_joint_response(json.dumps(payload))
        assert SLUGS[0] not in evals
        assert len(evals) == len(SLUGS) - 1

    def test_no_json_raises(self):
        # only a total JSON-extraction failure raises (so the judge call retries)
        with pytest.raises(InvalidOverseerResponse):
            parse_joint_response("I could not produce JSON.")

    def test_json_embedded_in_prose(self):
        text = "Here is my evaluation:\n" + json.dumps(joint_payload()) + "\nDone."
        evals, _ = parse_joint_response(text)
        assert set(evals) == set(SLUGS)


class TestStrictEnsemble:
    def test_all_judges_scored_slug_means(self):
        maps = [{s: 0.5 for s in SLUGS}, {s: -0.5 for s in SLUGS},
                {s: 0.5 for s in SLUGS}]
        value = strict_ensemble_mean(maps, 3, SLUGS)
        assert value[SLUGS[0]] == pytest.approx((0.5 - 0.5 + 0.5) / 3)

    def test_slug_missing_from_one_judge_is_nan_only_for_that_slug(self):
        maps = [
            {s: 0.5 for s in SLUGS},
            {s: 0.5 for s in SLUGS if s != SLUGS[2]},  # judge 2 missed slug 2
            {s: 0.5 for s in SLUGS},
        ]
        value = strict_ensemble_mean(maps, 3, SLUGS)
        assert math.isnan(value[SLUGS[2]])          # only slug 2 voided
        assert value[SLUGS[0]] == pytest.approx(0.5)  # the rest survive

    def test_invalid_judge_empty_map_voids_all_slugs(self):
        maps = [{s: 0.5 for s in SLUGS}, {s: 0.5 for s in SLUGS}, {}]  # judge 3 invalid
        value = strict_ensemble_mean(maps, 3, SLUGS)
        assert all(math.isnan(v) for v in value.values())


class TestMetric:
    def make_sample_score(self, value):
        import inspect_ai.scorer as sc
        return sc.SampleScore(score=sc.Score(value=value))

    def test_per_slug_means_and_humane_score(self):
        metric = joint_pattern_score()
        scores = [
            self.make_sample_score({slug: 1.0 for slug in SLUGS}),
            self.make_sample_score({slug: -0.5 for slug in SLUGS}),
        ]
        result = metric(scores)
        assert result[SLUGS[0]] == 0.25
        assert result["HumaneScore"] == 0.25

    def test_nan_filtered(self):
        metric = joint_pattern_score()
        scores = [
            self.make_sample_score({slug: 0.5 for slug in SLUGS}),
            self.make_sample_score({slug: math.nan for slug in SLUGS}),
        ]
        result = metric(scores)
        assert result[SLUGS[0]] == 0.5
        assert result["HumaneScore"] == 0.5
