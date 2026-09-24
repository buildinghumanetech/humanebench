"""Unit tests for the custom-prompt v4 evaluation: prompt-file handling, the v4
scorer's parsing and non-score handling, and the comparison math."""

import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Target

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import compare_prompt as cp  # noqa: E402
import custom_prompt_task as cpt  # noqa: E402
from humanebench import scorer_v4 as v4  # noqa: E402
from tests.conftest import create_mock_judge_model  # noqa: E402


# --- helpers ------------------------------------------------------------------

def judgement(overrides: dict[str, dict] | None = None) -> str:
    """A valid v4 judge response: everything not_applicable unless overridden."""
    overrides = overrides or {}
    principles = []
    for code in v4.V4_PRINCIPLES:
        principles.append({"name": code, **overrides.get(code, {"outcome": "not_applicable"})})
    return json.dumps({"principles": principles, "covered": [], "coverage": {}, "notes": ""})


def scored(score: float, confidence: str = "high") -> dict:
    p = {"outcome": "score", "score": score, "confidence": confidence}
    if score < 0:
        p |= {"tier": "row", "evidence": "quote", "rationale": "why", "behavior": "b",
              "suggestion": "s", "unless": ""}
    return p


INSUFFICIENT = {"outcome": "insufficient_context", "question": "q?", "resolves": "a -> b"}


class State:
    input_text = "user prompt"

    class Output:
        completion = "assistant response"

    output = Output()


# --- prompt file handling -----------------------------------------------------

class TestPromptFile:
    def test_reads_text_and_hashes_it(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("Be kind.\n", encoding="utf-8")
        text, sha = cpt.load_system_prompt(str(f))
        assert text == "Be kind.\n"
        assert sha == hashlib.sha256(b"Be kind.\n").hexdigest()

    def test_missing_file(self, tmp_path):
        with pytest.raises(cpt.PromptFileError, match="not found"):
            cpt.load_system_prompt(str(tmp_path / "nope.md"))

    @pytest.mark.parametrize("content", ["", "   \n\t\n"])
    def test_empty_file(self, tmp_path, content):
        f = tmp_path / "prompt.md"
        f.write_text(content)
        with pytest.raises(cpt.PromptFileError, match="empty"):
            cpt.load_system_prompt(str(f))

    @pytest.mark.parametrize("arg", [None, "", "  "])
    def test_no_argument(self, arg):
        with pytest.raises(cpt.PromptFileError, match="system_prompt_file="):
            cpt.load_system_prompt(arg)

    def test_directory(self, tmp_path):
        with pytest.raises(cpt.PromptFileError, match="not a file"):
            cpt.load_system_prompt(str(tmp_path))

    def test_binary_file(self, tmp_path):
        f = tmp_path / "prompt.bin"
        f.write_bytes(b"\xff\xfe\x00\x80")
        with pytest.raises(cpt.PromptFileError, match="UTF-8"):
            cpt.load_system_prompt(str(f))

    def test_relative_path_resolves_against_invocation_dir_not_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "p.md").write_text("hello")
        # Inspect chdirs into src/ while building tasks; PWD still names where
        # the user ran the command.
        monkeypatch.chdir(ROOT / "src")
        monkeypatch.setenv("PWD", str(tmp_path))
        text, _ = cpt.load_system_prompt("prompts/p.md")
        assert text == "hello"

    def test_task_fails_clearly_without_prompt(self):
        with pytest.raises(cpt.PromptFileError):
            cpt.custom_prompt_eval()

    def test_task_records_hash_not_text(self, tmp_path):
        secret = "SECRET-PROMPT-TEXT {principle} {{braces}}"
        f = tmp_path / "p.md"
        f.write_text(secret)
        task = cpt.custom_prompt_eval(system_prompt_file=str(f), per_principle=2)
        assert task.metadata["system_prompt_sha256"] == hashlib.sha256(secret.encode()).hexdigest()
        assert task.metadata["rubric_version"] == "v4"
        assert secret not in json.dumps(task.metadata)
        assert len(task.dataset) == 16

    def test_baseline_sends_no_system_prompt_but_records_pairing_hash(self, tmp_path):
        f = tmp_path / "p.md"
        f.write_text("x")
        task = cpt.baseline_v4_eval(system_prompt_file=str(f), per_principle=1)
        assert task.metadata["condition"] == "baseline"
        assert task.metadata["system_prompt_sha256"] == hashlib.sha256(b"x").hexdigest()
        assert cpt.baseline_v4_eval().metadata["system_prompt_sha256"] is None

    async def test_system_message_is_verbatim(self):
        """Inspect's system_message() would str.format this and eat the braces."""
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.solver import TaskState

        content = "Use {principle} and {{double}} literally.\nsrc/prompt.md"
        state = TaskState(model="m", sample_id=1, epoch=1, input="hi",
                          messages=[ChatMessageUser(content="hi")],
                          metadata={"principle": "REPLACED"})
        out = await cpt.verbatim_system_message(content)(state, None)
        assert out.messages[0].role == "system"
        assert out.messages[0].content == content


class TestSampling:
    def _samples(self, n_per=5):
        return [
            Sample(id=f"{p}-{i:03d}", input="x", target=p)
            for p in sorted(v4.PRINCIPLE_CODES) for i in range(n_per)
        ]

    def test_n_per_principle(self):
        out = cpt.stratify(self._samples(), per_principle=3, seed=1)
        assert len(out) == 24
        for p in v4.PRINCIPLE_CODES:
            assert sum(1 for s in out if s.target == p) == 3

    def test_deterministic_for_seed(self):
        a = [s.id for s in cpt.stratify(self._samples(), 3, seed=7)]
        b = [s.id for s in cpt.stratify(self._samples(), 3, seed=7)]
        assert a == b

    def test_interleaved_so_limit_stays_balanced(self):
        out = cpt.stratify(self._samples(), per_principle=None)
        first8 = {s.target for s in out[:8]}
        assert first8 == set(v4.PRINCIPLE_CODES)

    def test_per_principle_larger_than_group(self):
        assert len(cpt.stratify(self._samples(2), per_principle=10)) == 16

    def test_real_dataset_drops_excluded_items(self):
        from humanebench.excluded import load_excluded_ids
        ids = {s.id for s in cpt.load_samples()}
        assert len(ids) == 800 - len(load_excluded_ids())
        assert not ids & load_excluded_ids()


# --- v4 parsing and non-score states -----------------------------------------

class TestParseJudgement:
    def test_valid(self):
        j = v4.parse_judgement(judgement({"dignity_safety": scored(-0.5), "equity_inclusion": INSUFFICIENT}))
        assert j["dignity_safety"]["score"] == -0.5
        assert j["equity_inclusion"]["outcome"] == "insufficient_context"

    def test_fenced_json(self):
        assert v4.parse_judgement("```json\n" + judgement() + "\n```")

    @pytest.mark.parametrize("bad, match", [
        ({"outcome": "score", "score": 0, "confidence": "high"}, "illegal score"),
        ({"outcome": "score", "score": 0.5}, "confidence"),
        ({"outcome": "score", "score": 0.5, "confidence": 0.9}, "confidence"),
        ({"outcome": "score", "score": -1.0, "confidence": "high"}, "without tier"),
        ({"outcome": "not_applicable", "score": 0.5}, "carries a score"),
        ({"outcome": "insufficient_context", "resolves": "x"}, "without question"),
        ({"outcome": "maybe"}, "unknown outcome"),
        ({"score": 0.5, "confidence": "high"}, "v3-shaped"),
    ])
    def test_rejects(self, bad, match):
        with pytest.raises(v4.InvalidJudgement, match=match):
            v4.parse_judgement(judgement({"respect_attention": bad}))

    def test_rejects_seven_principles(self):
        data = json.loads(judgement())
        data["principles"].pop()
        with pytest.raises(v4.InvalidJudgement, match="expected 8"):
            v4.parse_judgement(json.dumps(data))

    def test_judge_prompt_read_from_disk_with_both_slots(self):
        text = v4.load_judge_prompt()
        assert text == (ROOT / "rubrics" / "judge_prompt_v4.md").read_text()
        filled = v4.assemble_prompt(text, "USER-{{.MessageContent}}", "RESPONSE")
        assert "USER-{{.MessageContent}}" in filled and "RESPONSE" in filled
        assert "{{.UserPrompt}}" not in filled


class TestNonScoreStates:
    @pytest.mark.parametrize("outcome", ["not_applicable", "insufficient_context", "covered"])
    def test_non_score_outcomes_do_not_count(self, outcome):
        status, value = v4.target_status({"outcome": outcome})
        assert status == outcome and value is None

    def test_low_confidence_is_dropped(self):
        assert v4.target_status(scored(1.0, "low")) == ("low_confidence", None)

    def test_combine_means_only_counting_judges(self):
        status, value = v4.combine_judges([("score", 1.0), ("not_applicable", None), ("score", -0.5)])
        assert status == "scored" and value == pytest.approx(0.25)

    def test_combine_nothing_counts_is_nan_not_zero(self):
        status, value = v4.combine_judges([("not_applicable", None), ("low_confidence", None)])
        assert status == "not_scored" and math.isnan(value)


class TestScorerV4:
    async def _run(self, responses_per_judge, target="protect-dignity-and-safety"):
        judges = [create_mock_judge_model(r) for r in responses_per_judge]
        with patch("inspect_ai.model.get_model", side_effect=lambda m: m):
            scorer = v4.overseer_v4(model=judges, score_attempts=2)
            return await scorer(State(), Target(target))

    async def test_scores_target_principle_only(self):
        r = judgement({"dignity_safety": scored(1.0), "respect_attention": scored(-1.0)})
        score = await self._run([[r], [r], [r]])
        assert score.value == 1.0
        assert score.answer == "protect-dignity-and-safety"
        assert score.metadata["status"] == "scored"
        assert score.metadata["rubric_version"] == "v4"

    async def test_mixed_scope_means_counting_judges(self):
        score = await self._run([
            [judgement({"dignity_safety": scored(1.0)})],
            [judgement({"dignity_safety": INSUFFICIENT})],
            [judgement({"dignity_safety": scored(-0.5)})],
        ])
        assert score.value == pytest.approx(0.25)
        assert score.metadata["judge_statuses"] == ["score", "insufficient_context", "score"]

    async def test_all_not_applicable_is_nan(self):
        score = await self._run([[judgement()]] * 3)
        assert math.isnan(score.value)
        assert score.metadata["status"] == "not_scored"

    async def test_retries_then_succeeds(self):
        good = judgement({"dignity_safety": scored(0.5)})
        score = await self._run([["not json", good], [good], [good]])
        assert score.value == 0.5

    async def test_any_judge_failing_is_nan(self):
        good = judgement({"dignity_safety": scored(0.5)})
        score = await self._run([[good], ["bad", "bad"], [good]])
        assert math.isnan(score.value)
        assert score.metadata["status"] == "judge_failed"

    def test_metric_excludes_nan_and_absent_principles(self):
        from inspect_ai.scorer import SampleScore, Score
        metric = v4.humane_score_v4()
        scores = [
            SampleScore(score=Score(value=1.0, answer="a")),
            SampleScore(score=Score(value=0.0, answer="a")),
            SampleScore(score=Score(value=math.nan, answer="a")),
            SampleScore(score=Score(value=-0.5, answer="b")),
            SampleScore(score=Score(value=math.nan, answer="c")),
        ]
        out = metric(scores)
        assert out["a"] == 0.5 and out["b"] == -0.5
        assert "c" not in out
        assert out["HumaneScore"] == 0.0


# --- comparison math ----------------------------------------------------------

P1, P2, P3 = cp.PRINCIPLES[0], cp.PRINCIPLES[1], cp.PRINCIPLES[2]


def r(principle, value, status="scored", votes=None):
    return cp.SampleResult("id", principle, status, value, votes or [])


class TestComparison:
    def test_means_exclude_non_score_states(self):
        s = cp.summarize([
            r(P1, 1.0), r(P1, 0.0),
            r(P1, None, "not_scored", ["not_applicable"] * 3),
            r(P1, None, "judge_failed"),
            r(P1, None, "errored"),
        ])
        p = s.principles[P1]
        assert p.mean == 0.5
        assert p.scored == 2 and p.not_scored == 1 and p.failed == 2

    def test_principle_with_nothing_in_scope_has_no_mean(self):
        s = cp.summarize([r(P1, 1.0), r(P2, None, "not_scored")])
        assert s.principles[P2].mean is None
        assert s.principles[P3].mean is None
        # HumaneScore is the mean of principles that have one, not over eight.
        assert s.humane_score == 1.0

    def test_deltas_and_worse(self):
        base = [r(P1, 0.5), r(P2, 0.5), r(P3, None, "not_scored")]
        cust = [r(P1, 1.0), r(P2, -0.5), r(P3, 1.0)]
        c = cp.compare(base, cust)
        assert c.principle_delta(P1) == pytest.approx(0.5)
        assert c.principle_delta(P2) == pytest.approx(-1.0)
        assert c.principle_delta(P3) is None  # no baseline mean: no delta, not +1.0
        assert c.worse == [P2]
        assert c.baseline.humane_score == 0.5
        assert c.custom.humane_score == pytest.approx(0.5)
        assert c.overall_delta == pytest.approx(0.0)

    def test_context_blocked_rate_ignores_not_applicable(self):
        s = cp.summarize([
            r(P1, 0.5, votes=["score", "insufficient_context", "not_applicable"]),
            r(P1, 0.5, votes=["score", "score", "not_applicable"]),
        ])
        assert s.principles[P1].context_blocked_rate == pytest.approx(1 / 4)

    def test_report(self):
        c = cp.compare([r(P1, 0.5), r(P2, 0.5)], [r(P1, 1.0), r(P2, 0.0)])
        out = cp.format_report(c, model="m", prompt_sha="abc123")
        assert out.splitlines()[0] == "HumaneBench rubric v4, not comparable to the v1 leaderboard."
        assert "abc123" in out
        assert f"Got worse: {P2} (-0.50, noisy)" in out
        assert "too noisy to interpret" in out
        assert "Overall: no change" in out
        # A principle with no in-scope sample renders as "-", never 0.00.
        row = next(line for line in out.splitlines() if line.startswith(P3))
        assert "0.00" not in row and "-" in row

    def test_report_does_not_flag_rows_with_enough_samples(self):
        base = [r(P1, 0.5)] * cp.NOISY_N
        cust = [r(P1, 0.0)] * cp.NOISY_N
        out = cp.format_report(cp.compare(base, cust))
        row = next(line for line in out.splitlines() if line.startswith(P1))
        assert "noisy" not in row
        assert f"Got worse: {P1} (-0.50)" in out

    def test_report_flags_context_blocked_runs_as_directional(self):
        votes = ["insufficient_context", "insufficient_context", "score"]
        c = cp.compare([r(P1, 0.5, votes=votes)], [r(P1, 0.5)])
        assert "Directional only" in cp.format_report(c)
