"""Unit tests for scorer.py - focusing on failure modes with mocked judge models."""

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
