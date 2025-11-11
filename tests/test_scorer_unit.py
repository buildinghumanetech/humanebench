"""Unit tests for scorer.py - focusing on failure modes with mocked judge models."""

import math
import pytest
import json
from unittest.mock import AsyncMock, patch
from inspect_ai.solver import TaskState, Generate
from inspect_ai.scorer import Target
from inspect_ai.model import ModelOutput

from humanebench.scorer import overseer, _extract_json, _try_score
from humanebench.humane_patterns import HumanePattern
from tests.conftest import create_mock_model_output, create_mock_judge_model


class TestJSONExtraction:
    """Test the _extract_json function's progressive recovery strategies."""

    def test_valid_json(self):
        """Test extraction of valid JSON."""
        response = '{"severity": 1.0, "reasoning": "Good", "invalid": false}'
        result = _extract_json(response)
        assert result["severity"] == 1.0
        assert result["reasoning"] == "Good"
        assert result["invalid"] is False

    def test_json_with_extra_text_before(self):
        """Test JSON extraction when there's explanatory text before the JSON."""
        response = 'Sure! Here is my evaluation: {"severity": 0.5, "reasoning": "Okay", "invalid": false}'
        result = _extract_json(response)
        assert result["severity"] == 0.5

    def test_json_with_extra_text_after(self):
        """Test JSON extraction when there's text after the JSON."""
        response = '{"severity": -0.5, "reasoning": "Bad", "invalid": false} Hope this helps!'
        result = _extract_json(response)
        assert result["severity"] == -0.5

    def test_json_with_trailing_comma(self):
        """Test JSON with trailing commas (should be sanitized)."""
        response = '{"severity": 1.0, "reasoning": "Good", "invalid": false,}'
        result = _extract_json(response)
        assert result["severity"] == 1.0

    def test_json_with_comments(self):
        """Test JSON with // comments (should be removed)."""
        response = '''
        {
            "severity": 0.5, // This is good
            "reasoning": "Decent response", // Explanation
            "invalid": false
        }
        '''
        result = _extract_json(response)
        assert result["severity"] == 0.5

    def test_invalid_json_raises(self):
        """Test that completely invalid JSON raises InvalidOverseerResponse."""
        from humanebench.scorer import InvalidOverseerResponse

        with pytest.raises(InvalidOverseerResponse):
            _extract_json("This is not JSON at all")

    def test_malformed_json_raises(self):
        """Test that malformed JSON (missing brace) raises InvalidOverseerResponse."""
        from humanebench.scorer import InvalidOverseerResponse

        with pytest.raises(InvalidOverseerResponse):
            _extract_json('{"severity": 1.0, "reasoning": "Good"')  # Missing closing brace


class TestInvalidSeverityValues:
    """Test handling of invalid severity values (the Nov 8-10 Pydantic bug!)."""

    @pytest.mark.asyncio
    async def test_integer_zero_severity(self, sample_humane_pattern):
        """Test that severity=0 (integer) is caught and retried."""
        # Mock a judge that returns integer 0 instead of valid float
        mock_model = create_mock_judge_model([
            '{"severity": 0, "reasoning": "Neutral", "invalid": false}',  # Attempt 1: invalid
            '{"severity": 0, "reasoning": "Neutral", "invalid": false}',  # Attempt 2: invalid
            '{"severity": 0, "reasoning": "Neutral", "invalid": false}',  # Attempt 3: invalid
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should fail after 3 attempts
        assert result is None
        assert completion is not None

    @pytest.mark.asyncio
    async def test_out_of_range_severity_positive(self, sample_humane_pattern):
        """Test that severity=0.7 (not in valid set) is caught and retried."""
        mock_model = create_mock_judge_model([
            '{"severity": 0.7, "reasoning": "Good but not great", "invalid": false}',
            '{"severity": 0.7, "reasoning": "Good but not great", "invalid": false}',
            '{"severity": 0.7, "reasoning": "Good but not great", "invalid": false}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_out_of_range_severity_negative(self, sample_humane_pattern):
        """Test that severity=-0.3 (not in valid set) is caught and retried."""
        mock_model = create_mock_judge_model([
            '{"severity": -0.3, "reasoning": "Somewhat bad", "invalid": false}',
            '{"severity": -0.3, "reasoning": "Somewhat bad", "invalid": false}',
            '{"severity": -0.3, "reasoning": "Somewhat bad", "invalid": false}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_way_out_of_range_severity(self, sample_humane_pattern):
        """Test that severity=2.0 (way out of range) is caught and retried."""
        mock_model = create_mock_judge_model([
            '{"severity": 2.0, "reasoning": "Amazing!", "invalid": false}',
            '{"severity": 2.0, "reasoning": "Amazing!", "invalid": false}',
            '{"severity": 2.0, "reasoning": "Amazing!", "invalid": false}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_severity_then_valid(self, sample_humane_pattern):
        """Test that invalid severity retries and succeeds on second attempt."""
        mock_model = create_mock_judge_model([
            '{"severity": 0.7, "reasoning": "Invalid first", "invalid": false}',  # Attempt 1: invalid
            '{"severity": 1.0, "reasoning": "Valid second", "invalid": false}',    # Attempt 2: valid!
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should succeed on second attempt
        assert result is not None
        assert result.severity == 1.0
        assert result.reasoning == "Valid second"


class TestRetryExhaustion:
    """Test that retry logic exhausts properly after max attempts."""

    @pytest.mark.asyncio
    async def test_json_parse_failure_exhausts_retries(self, sample_humane_pattern):
        """Test that consistently invalid JSON exhausts all retries."""
        mock_model = create_mock_judge_model([
            "Not JSON at all",  # Attempt 1
            "Still not JSON",   # Attempt 2
            "Nope, no JSON",    # Attempt 3
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should return None after exhausting retries
        assert result is None
        assert completion == "Nope, no JSON"  # Last attempt's completion

    @pytest.mark.asyncio
    async def test_retry_count_is_respected(self, sample_humane_pattern):
        """Test that retry count parameter is respected."""
        responses = ["Bad JSON"] * 5  # More than we'll try
        mock_model = create_mock_judge_model(responses)

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=2,  # Only try 2 times
        )

        # Should only call generate 2 times
        assert mock_model.generate.call_count == 2
        assert result is None


class TestMissingRequiredFields:
    """Test handling of JSON with missing required fields."""

    @pytest.mark.asyncio
    async def test_missing_severity_field(self, sample_humane_pattern):
        """Test JSON missing the 'severity' field."""
        mock_model = create_mock_judge_model([
            '{"reasoning": "Good", "invalid": false}',  # Missing severity
            '{"reasoning": "Good", "invalid": false}',
            '{"reasoning": "Good", "invalid": false}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        assert result is None  # Should fail gracefully

    @pytest.mark.asyncio
    async def test_missing_reasoning_field(self, sample_humane_pattern):
        """Test JSON missing the 'reasoning' field."""
        mock_model = create_mock_judge_model([
            '{"severity": 1.0, "invalid": false}',  # Missing reasoning
            '{"severity": 1.0, "invalid": false}',
            '{"severity": 1.0, "invalid": false}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_missing_invalid_field(self, sample_humane_pattern):
        """Test JSON missing the 'invalid' field."""
        mock_model = create_mock_judge_model([
            '{"severity": 1.0, "reasoning": "Good"}',  # Missing invalid
            '{"severity": 1.0, "reasoning": "Good"}',
            '{"severity": 1.0, "reasoning": "Good"}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        assert result is None


class TestPydanticValidationErrors:
    """Test handling of Pydantic validation errors."""

    @pytest.mark.asyncio
    async def test_severity_as_string(self, sample_humane_pattern):
        """Test severity as string instead of float."""
        mock_model = create_mock_judge_model([
            '{"severity": "high", "reasoning": "Good", "invalid": false}',
            '{"severity": "high", "reasoning": "Good", "invalid": false}',
            '{"severity": "high", "reasoning": "Good", "invalid": false}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_as_integer(self, sample_humane_pattern):
        """Test 'invalid' field as integer - Pydantic coerces to boolean."""
        mock_model = create_mock_judge_model([
            '{"severity": 1.0, "reasoning": "Good", "invalid": 1}',
        ])

        result, completion = await _try_score(
            model=mock_model,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Pydantic coerces integer 1 to boolean True, which is fine
        assert result is not None
        assert result.invalid is True
        assert result.severity == 1.0


class TestMalformedJudgeOutputsIntegration:
    """Test that the full overseer() scorer handles malformed judge outputs gracefully."""

    @pytest.mark.asyncio
    async def test_malformed_json_single_judge_returns_zero_score(self, sample_humane_pattern):
        """Test that malformed JSON from judge results in Score with value=0."""
        # Create mock task state
        class MockTaskState:
            input_text = "Test prompt"

            class Output:
                completion = "Test response"

            output = Output()

        # Mock judge returning bad JSON on all attempts
        mock_judge = create_mock_judge_model([
            "This is not JSON",
            "Still not JSON",
            "Nope",
        ])

        # Patch get_model to return our mock directly
        with patch('inspect_ai.model.get_model', return_value=mock_judge):
            # Create scorer with the mock
            scorer = overseer(
                humane_patterns=[sample_humane_pattern],
                model=mock_judge,
                score_attempts=3,
            )

            # Call scorer
            score = await scorer(MockTaskState(), Target(sample_humane_pattern.id))

        # Assertions
        assert math.isnan(score.value), "Failed judge should result in score of NaN"
        assert "Failed to score" in score.explanation
        assert score.metadata is not None
        assert "failed_model" in score.metadata

    @pytest.mark.asyncio
    async def test_malformed_json_ensemble_all_judges_fail(self, sample_humane_pattern):
        """Test that when all judges in ensemble fail, score is 0."""
        class MockTaskState:
            input_text = "Test prompt"

            class Output:
                completion = "Test response"

            output = Output()

        # Create multiple judges that all fail
        mock_judge_1 = create_mock_judge_model(["Not JSON"] * 3)
        mock_judge_2 = create_mock_judge_model(["Also not JSON"] * 3)

        # Patch get_model to return the appropriate mock for each call
        with patch('inspect_ai.model.get_model', side_effect=[mock_judge_1, mock_judge_2]):
            scorer = overseer(
                humane_patterns=[sample_humane_pattern],
                model=[mock_judge_1, mock_judge_2],
                score_attempts=3,
            )

            score = await scorer(MockTaskState(), Target(sample_humane_pattern.id))

        # All judges failed, should return NaN
        assert math.isnan(score.value)
        assert "Failed to score" in score.explanation
        assert "failed_model" in score.metadata

    @pytest.mark.asyncio
    async def test_malformed_json_ensemble_one_judge_fails(self, sample_humane_pattern):
        """Test that if one judge in ensemble fails, entire ensemble fails."""
        class MockTaskState:
            input_text = "Test prompt"

            class Output:
                completion = "Test response"

            output = Output()

        # First judge succeeds, second fails
        mock_judge_1 = create_mock_judge_model([
            '{"severity": 1.0, "reasoning": "Good", "invalid": false}'
        ])
        mock_judge_2 = create_mock_judge_model(["Not JSON"] * 3)

        with patch('inspect_ai.model.get_model', side_effect=[mock_judge_1, mock_judge_2]):
            scorer = overseer(
                humane_patterns=[sample_humane_pattern],
                model=[mock_judge_1, mock_judge_2],
                score_attempts=3,
            )

            score = await scorer(MockTaskState(), Target(sample_humane_pattern.id))

        # Per current logic (line 160), if ANY judge fails, entire ensemble fails
        assert math.isnan(score.value)
        assert "Failed to score" in score.explanation
        assert "failed_model" in score.metadata

    @pytest.mark.asyncio
    async def test_missing_severity_field_returns_zero_score(self, sample_humane_pattern):
        """Test that missing severity field in judge response results in Score with value=NaN."""
        class MockTaskState:
            input_text = "Test prompt"

            class Output:
                completion = "Test response"

            output = Output()

        # Judge returns JSON without severity field
        mock_judge = create_mock_judge_model([
            '{"reasoning": "Good", "invalid": false}',
            '{"reasoning": "Good", "invalid": false}',
            '{"reasoning": "Good", "invalid": false}',
        ])

        with patch('inspect_ai.model.get_model', return_value=mock_judge):
            scorer = overseer(
                humane_patterns=[sample_humane_pattern],
                model=mock_judge,
                score_attempts=3,
            )

            score = await scorer(MockTaskState(), Target(sample_humane_pattern.id))

        assert math.isnan(score.value)
        assert "Failed to score" in score.explanation

    @pytest.mark.asyncio
    async def test_invalid_severity_exhausts_retries_returns_zero(self, sample_humane_pattern):
        """Test that invalid severity after retry exhaustion results in Score with value=NaN."""
        class MockTaskState:
            input_text = "Test prompt"

            class Output:
                completion = "Test response"

            output = Output()

        # Judge returns severity=0 (invalid) on all attempts
        mock_judge = create_mock_judge_model([
            '{"severity": 0, "reasoning": "Neutral", "invalid": false}',
            '{"severity": 0, "reasoning": "Neutral", "invalid": false}',
            '{"severity": 0, "reasoning": "Neutral", "invalid": false}',
        ])

        with patch('inspect_ai.model.get_model', return_value=mock_judge):
            scorer = overseer(
                humane_patterns=[sample_humane_pattern],
                model=mock_judge,
                score_attempts=3,
            )

            score = await scorer(MockTaskState(), Target(sample_humane_pattern.id))

        # Should call judge exactly 3 times
        assert mock_judge.generate.call_count == 3
        # Should return score of NaN after exhausting retries
        assert math.isnan(score.value)
        assert "Failed to score" in score.explanation


class TestUnavailableJudgeLLM:
    """Test handling of exceptions from judge model.generate()."""

    @pytest.mark.asyncio
    async def test_judge_network_timeout_is_caught(self, sample_humane_pattern):
        """Test that TimeoutError from judge is handled gracefully."""
        import asyncio

        mock_judge = AsyncMock()
        mock_judge.generate.side_effect = asyncio.TimeoutError("Connection timeout")

        result, completion = await _try_score(
            model=mock_judge,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should not crash, should return None
        assert result is None
        assert mock_judge.generate.call_count == 3  # Should retry

    @pytest.mark.asyncio
    async def test_judge_api_error_is_caught(self, sample_humane_pattern):
        """Test that generic API exceptions are handled gracefully."""
        mock_judge = AsyncMock()
        mock_judge.generate.side_effect = Exception("API Error 503: Service Unavailable")

        result, completion = await _try_score(
            model=mock_judge,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should not crash, should return None
        assert result is None
        assert mock_judge.generate.call_count == 3  # Should retry

    @pytest.mark.asyncio
    async def test_judge_connection_error_is_caught(self, sample_humane_pattern):
        """Test that connection errors are handled gracefully."""
        mock_judge = AsyncMock()
        mock_judge.generate.side_effect = ConnectionError("Failed to connect to API")

        result, completion = await _try_score(
            model=mock_judge,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should not crash, should return None
        assert result is None
        assert mock_judge.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_judge_rate_limit_error_is_caught(self, sample_humane_pattern):
        """Test that rate limit errors are handled gracefully."""
        mock_judge = AsyncMock()
        mock_judge.generate.side_effect = Exception("Rate limit exceeded")

        result, completion = await _try_score(
            model=mock_judge,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should not crash, should return None
        assert result is None
        assert mock_judge.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_judge_authentication_error_is_caught(self, sample_humane_pattern):
        """Test that authentication errors are handled gracefully."""
        mock_judge = AsyncMock()
        mock_judge.generate.side_effect = Exception("Authentication failed: Invalid API key")

        result, completion = await _try_score(
            model=mock_judge,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should not crash, should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_ensemble_one_judge_unavailable_fails_all(self, sample_humane_pattern):
        """Test that if one judge in ensemble raises exception, entire ensemble fails."""
        class MockTaskState:
            input_text = "Test prompt"

            class Output:
                completion = "Test response"

            output = Output()

        # First judge works fine
        mock_judge_1 = create_mock_judge_model([
            '{"severity": 1.0, "reasoning": "Good", "invalid": false}'
        ])

        # Second judge raises exception
        mock_judge_2 = AsyncMock()
        mock_judge_2.generate.side_effect = Exception("Network error")
        # Need to set __str__ for the error message
        mock_judge_2.__str__ = lambda self: "mock-judge-2"

        with patch('inspect_ai.model.get_model', side_effect=[mock_judge_1, mock_judge_2]):
            scorer = overseer(
                humane_patterns=[sample_humane_pattern],
                model=[mock_judge_1, mock_judge_2],
                score_attempts=3,
            )

            score = await scorer(MockTaskState(), Target(sample_humane_pattern.id))

        # Per current logic, if ANY judge fails, entire ensemble fails
        assert math.isnan(score.value)
        assert "Failed to score" in score.explanation
        assert "failed_model" in score.metadata

    @pytest.mark.asyncio
    async def test_judge_generic_exception_is_caught(self, sample_humane_pattern):
        """Test that unexpected exceptions are handled gracefully."""
        mock_judge = AsyncMock()
        mock_judge.generate.side_effect = RuntimeError("Unexpected error occurred")

        result, completion = await _try_score(
            model=mock_judge,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should not crash, should return None
        assert result is None
        assert mock_judge.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_judge_exception_then_success(self, sample_humane_pattern):
        """Test that judge can recover from exception on retry."""
        mock_judge = AsyncMock()

        # First attempt raises exception, second succeeds
        mock_judge.generate.side_effect = [
            Exception("Temporary error"),
            create_mock_model_output('{"severity": 1.0, "reasoning": "Good", "invalid": false}'),
        ]

        result, completion = await _try_score(
            model=mock_judge,
            score_prompt="Test prompt",
            issue=sample_humane_pattern,
            temperature=0.0,
            system_message="Test system message",
            score_attempts=3,
        )

        # Should succeed on second attempt
        assert result is not None
        assert result.severity == 1.0
        assert mock_judge.generate.call_count == 2  # Only needed 2 attempts
