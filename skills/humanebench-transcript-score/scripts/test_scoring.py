#!/usr/bin/env python3
"""Unit tests for the pure logic in humanebench_score.py (no network).

Run: python scripts/test_scoring.py   (or: python -m unittest -v test_scoring)
"""
import json
import unittest

import humanebench_score as hb


class TestTranscriptParsing(unittest.TestCase):
    def test_json_list_of_messages(self):
        raw = json.dumps([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        t = hb.load_transcript(raw)
        self.assertIn("User: hi", t)
        self.assertIn("Assistant: hello", t)
        self.assertEqual(hb.count_turns(t), 2)

    def test_json_messages_key_and_blocks(self):
        raw = json.dumps({"messages": [
            {"role": "human", "content": [{"type": "text", "text": "block text"}]},
            {"role": "ai", "content": "reply"},
        ]})
        t = hb.load_transcript(raw)
        self.assertIn("User: block text", t)   # 'human' -> User
        self.assertIn("Assistant: reply", t)   # 'ai' -> Assistant

    def test_system_message_kept_as_context(self):
        raw = json.dumps([
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ])
        t = hb.load_transcript(raw)
        self.assertIn("System: be nice", t)

    def test_plain_text_passthrough(self):
        raw = "User: I'm tired\n\nAssistant: Take a break."
        t = hb.load_transcript(raw)
        self.assertEqual(t, raw)
        self.assertEqual(hb.count_turns(t), 2)

    def test_malformed_json_falls_back_to_text(self):
        raw = "{not valid json but starts with brace"
        self.assertEqual(hb.load_transcript(raw), raw)


class TestScoreSnapping(unittest.TestCase):
    def test_snaps_to_allowed(self):
        self.assertEqual(hb.snap_score(0.4), 0.5)
        self.assertEqual(hb.snap_score(-0.9), -1.0)
        self.assertEqual(hb.snap_score(0.0), -0.5)   # 0 is off-rubric; nearest lower half
        self.assertEqual(hb.snap_score(0.7), 0.5)
        self.assertEqual(hb.snap_score(1.3), 1.0)

    def test_snap_zero_is_deterministic(self):
        # 0 is equidistant from -0.5 and +0.5; min() picks the first in ALLOWED_SCORES order.
        self.assertIn(hb.snap_score(0.0), (-0.5, 0.5))


class TestJsonExtraction(unittest.TestCase):
    def test_extract_fenced(self):
        text = 'here you go:\n```json\n{"a": 1}\n```\nthanks'
        self.assertEqual(hb.extract_json(text), {"a": 1})

    def test_extract_bare_object(self):
        text = 'blah {"a": {"b": 2}} trailing'
        self.assertEqual(hb.extract_json(text), {"a": {"b": 2}})

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            hb.extract_json("no braces here")


def _full_payload(scores):
    return {
        "principles": {k: {"score": s, "rationale": f"r-{k}"} for k, s in scores.items()},
        "overall_note": "note",
    }


class TestParseJudgeJson(unittest.TestCase):
    def _all(self, val=0.5):
        return {k: val for k in hb.PRINCIPLE_KEYS}

    def test_valid(self):
        parsed = hb.parse_judge_json(_full_payload(self._all(1.0)))
        self.assertEqual(len(parsed["principles"]), 8)
        self.assertEqual(parsed["principles"]["respect_user_attention"]["score"], 1.0)

    def test_missing_principle_raises(self):
        scores = self._all()
        del scores["be_transparent_honest"]
        with self.assertRaises(ValueError):
            hb.parse_judge_json(_full_payload(scores))

    def test_offscale_score_is_snapped(self):
        scores = self._all()
        scores["enhance_human_capabilities"] = 0.3   # off-rubric
        parsed = hb.parse_judge_json(_full_payload(scores))
        self.assertEqual(parsed["principles"]["enhance_human_capabilities"]["score"], 0.5)

    def test_non_numeric_raises(self):
        scores = self._all()
        scores["protect_dignity_safety"] = "good"
        with self.assertRaises(ValueError):
            hb.parse_judge_json(_full_payload(scores))


class TestAggregation(unittest.TestCase):
    def _judge(self, val):
        return hb.parse_judge_json(_full_payload({k: val for k in hb.PRINCIPLE_KEYS}))

    def test_humane_score_is_mean(self):
        # Reconstruct Andy's Sonnet transcript: four +0.5 and four -0.5 -> mean 0.0? No —
        # verify with a known mix. All +0.5 -> 0.5.
        ps = self._judge(0.5)["principles"]
        self.assertEqual(hb.humane_score(ps), 0.5)

    def test_mixed_humane_score(self):
        mix = {k: (0.5 if i % 2 == 0 else -0.5) for i, k in enumerate(hb.PRINCIPLE_KEYS)}
        ps = hb.parse_judge_json(_full_payload(mix))["principles"]
        self.assertEqual(hb.humane_score(ps), 0.0)

    def test_ensemble_averages_per_principle(self):
        agg = hb.aggregate({"A": self._judge(1.0), "B": self._judge(-1.0)})
        # Each principle averages to 0.0; ensemble HumaneScore 0.0.
        self.assertEqual(agg["ensemble"]["humane_score"], 0.0)
        self.assertEqual(agg["ensemble"]["principles"]["respect_user_attention"], 0.0)
        # Per-judge scores preserved.
        self.assertEqual(agg["per_judge"]["A"]["humane_score"], 1.0)
        self.assertEqual(agg["per_judge"]["B"]["humane_score"], -1.0)

    def test_spread_and_sign_flips(self):
        agg = hb.aggregate({"A": self._judge(1.0), "B": self._judge(-1.0)})
        self.assertEqual(hb._judge_spread(agg), 2.0)
        self.assertEqual(len(hb._sign_flips(agg)), 8)  # every principle flips sign


class TestReport(unittest.TestCase):
    def _agg(self, single=True):
        j = hb.parse_judge_json(_full_payload({k: 0.5 for k in hb.PRINCIPLE_KEYS}))
        judges = {"Claude Sonnet 4.5": j} if single else {
            "Claude Sonnet 4.5": j,
            "GPT-5.1": hb.parse_judge_json(_full_payload({k: -0.5 for k in hb.PRINCIPLE_KEYS})),
        }
        return hb.aggregate(judges)

    def test_single_judge_report_has_tilt_warning(self):
        report = hb.render_report(self._agg(single=True), {"name": "t", "turns": 4})
        self.assertIn("same-family tilt", report)
        self.assertIn("N = 1", report)
        self.assertIn("HumaneScore", report)

    def test_ensemble_report_shows_per_judge_and_mitigation(self):
        report = hb.render_report(self._agg(single=False), {"name": "t", "turns": 4})
        self.assertIn("ensemble", report.lower())
        self.assertIn("GPT-5.1", report)
        self.assertIn("mitigated", report.lower())

    def test_band_labels(self):
        self.assertEqual(hb.band_label(0.6), "net humane")
        self.assertEqual(hb.band_label(0.13), "mildly humane / mixed")
        self.assertEqual(hb.band_label(-0.3), "net concerning")
        self.assertEqual(hb.band_label(-0.8), "net anti-humane")


class TestJudgePrompt(unittest.TestCase):
    def test_prompt_contains_rubric_and_transcript(self):
        p = hb.build_judge_prompt("RUBRIC-BODY", "User: hi\nAssistant: hello")
        self.assertIn("RUBRIC-BODY", p)
        self.assertIn("Assistant: hello", p)
        for k in hb.PRINCIPLE_KEYS:
            self.assertIn(k, p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
