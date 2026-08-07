"""Unit tests for the joint (all-principles-in-one-call) scorer's pure parts."""
import json
import math

import pytest

from humanebench.humane_patterns import HUMANE_PATTERNS
from humanebench.joint_scorer import (
    PRINCIPLE_ORDER_SALT,
    build_joint_prompt,
    joint_pattern_score,
    parse_joint_response,
    principle_order_for,
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


class TestParsing:
    def test_valid_payload(self):
        evals, invalid = parse_joint_response(json.dumps(joint_payload()))
        assert set(evals) == set(SLUGS)
        assert evals[SLUGS[0]].severity == 0.5
        assert invalid is False

    def test_invalid_flag_carried(self):
        _, invalid = parse_joint_response(json.dumps(joint_payload(invalid=True)))
        assert invalid is True

    def test_missing_principle_rejected(self):
        payload = joint_payload()
        del payload[SLUGS[3]]
        with pytest.raises(InvalidOverseerResponse, match=SLUGS[3]):
            parse_joint_response(json.dumps(payload))

    def test_out_of_scale_severity_rejected(self):
        payload = joint_payload()
        payload[SLUGS[0]]["severity"] = 0.0
        with pytest.raises(Exception):
            parse_joint_response(json.dumps(payload))

    def test_json_embedded_in_prose(self):
        text = "Here is my evaluation:\n" + json.dumps(joint_payload()) + "\nDone."
        evals, _ = parse_joint_response(text)
        assert set(evals) == set(SLUGS)


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
