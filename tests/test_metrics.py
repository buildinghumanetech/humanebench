"""Tests for metrics and utility functions in scorer.py and humane_patterns.py."""

import math
import pytest
from inspect_ai.scorer import Score

from humanebench.scorer import humane_pattern_score
from humanebench.humane_patterns import (
    resolve_humane_patterns,
    HumanePattern,
    RESPECT_USER_ATTENTION,
    ENABLE_MEANINGFUL_CHOICES,
)


class MockSampleScore:
    """Mock SampleScore object for testing metrics."""

    def __init__(self, pattern_id: str, value: float):
        self.score = Score(
            value=value,
            answer=pattern_id,
        )


class TestHumanePatternScoreMetric:
    """Test the humane_pattern_score() metric calculation."""

    def test_single_pattern_multiple_samples(self):
        """Test averaging across multiple samples for one pattern."""
        metric = humane_pattern_score(["respect-user-attention"])

        scores = [
            MockSampleScore("respect-user-attention", 1.0),
            MockSampleScore("respect-user-attention", 0.5),
            MockSampleScore("respect-user-attention", -0.5),
        ]

        result = metric(scores)

        # Average: (1.0 + 0.5 + -0.5) / 3 = 0.333...
        assert result["respect-user-attention"] == 0.33
        assert result["HumaneScore"] == 0.33

    def test_multiple_patterns_multiple_samples(self):
        """Test metric with multiple patterns and samples."""
        metric = humane_pattern_score([
            "respect-user-attention",
            "enable-meaningful-choices",
        ])

        scores = [
            MockSampleScore("respect-user-attention", 1.0),
            MockSampleScore("respect-user-attention", 0.5),
            MockSampleScore("enable-meaningful-choices", -0.5),
            MockSampleScore("enable-meaningful-choices", -1.0),
        ]

        result = metric(scores)

        # respect-user-attention: (1.0 + 0.5) / 2 = 0.75
        assert result["respect-user-attention"] == 0.75
        # enable-meaningful-choices: (-0.5 + -1.0) / 2 = -0.75
        assert result["enable-meaningful-choices"] == -0.75
        # HumaneScore: (0.75 + -0.75) / 2 = 0.0
        assert result["HumaneScore"] == 0.0

    def test_filters_nan_values(self):
        """Test that NaN values (failed judges) are filtered out."""
        metric = humane_pattern_score(["respect-user-attention"])

        scores = [
            MockSampleScore("respect-user-attention", 1.0),
            MockSampleScore("respect-user-attention", math.nan),  # Failed judge
            MockSampleScore("respect-user-attention", 0.5),
            MockSampleScore("respect-user-attention", math.nan),  # Failed judge
        ]

        result = metric(scores)

        # Should only average 1.0 and 0.5, ignoring NaN values
        # Average: (1.0 + 0.5) / 2 = 0.75
        assert result["respect-user-attention"] == 0.75
        assert result["HumaneScore"] == 0.75

    def test_all_nan_values_returns_zero(self):
        """Test that if all scores are NaN, pattern score is 0."""
        metric = humane_pattern_score(["respect-user-attention"])

        scores = [
            MockSampleScore("respect-user-attention", math.nan),
            MockSampleScore("respect-user-attention", math.nan),
            MockSampleScore("respect-user-attention", math.nan),
        ]

        result = metric(scores)

        # No valid scores, should return 0
        assert result["respect-user-attention"] == 0
        assert result["HumaneScore"] == 0

    def test_scores_for_unknown_patterns_ignored(self):
        """Test that scores for patterns not in metric config are ignored."""
        metric = humane_pattern_score(["respect-user-attention"])

        scores = [
            MockSampleScore("respect-user-attention", 1.0),
            MockSampleScore("unknown-pattern", 0.5),  # Should be ignored
        ]

        result = metric(scores)

        assert result["respect-user-attention"] == 1.0
        assert result["HumaneScore"] == 1.0
        assert "unknown-pattern" not in result


class TestResolveHumanePatterns:
    """Test the resolve_humane_patterns() function."""

    def test_none_returns_all_patterns(self):
        """Test that None returns all 8 patterns."""
        result = resolve_humane_patterns(None)

        assert len(result) == 8
        assert all(isinstance(p, HumanePattern) for p in result)

        # Check all expected pattern IDs are present
        pattern_ids = {p.id for p in result}
        assert "respect-user-attention" in pattern_ids
        assert "enable-meaningful-choices" in pattern_ids
        assert "enhance-human-capabilities" in pattern_ids
        assert "protect-dignity-and-safety" in pattern_ids
        assert "foster-healthy-relationships" in pattern_ids
        assert "prioritize-long-term-wellbeing" in pattern_ids
        assert "be-transparent-and-honest" in pattern_ids
        assert "design-for-equity-and-inclusion" in pattern_ids

    def test_single_string_id_returns_list(self):
        """Test that single string ID returns list with one pattern."""
        result = resolve_humane_patterns("respect-user-attention")

        assert len(result) == 1
        assert isinstance(result[0], HumanePattern)
        assert result[0].id == "respect-user-attention"
        assert result[0].display_name == "Respect User Attention"

    def test_comma_separated_string_returns_multiple(self):
        """Test that comma-separated string returns multiple patterns."""
        result = resolve_humane_patterns(
            "respect-user-attention,enable-meaningful-choices,be-transparent-and-honest"
        )

        assert len(result) == 3
        assert all(isinstance(p, HumanePattern) for p in result)

        pattern_ids = [p.id for p in result]
        assert "respect-user-attention" in pattern_ids
        assert "enable-meaningful-choices" in pattern_ids
        assert "be-transparent-and-honest" in pattern_ids

    def test_list_of_string_ids_returns_patterns(self):
        """Test that list of string IDs returns matching patterns."""
        result = resolve_humane_patterns([
            "respect-user-attention",
            "enable-meaningful-choices",
        ])

        assert len(result) == 2
        assert all(isinstance(p, HumanePattern) for p in result)
        assert result[0].id == "respect-user-attention"
        assert result[1].id == "enable-meaningful-choices"

    def test_mixed_list_with_objects_and_strings(self):
        """Test list with mix of HumanePattern objects and string IDs."""
        result = resolve_humane_patterns([
            RESPECT_USER_ATTENTION,  # HumanePattern object
            "enable-meaningful-choices",  # String ID
        ])

        assert len(result) == 2
        assert result[0] == RESPECT_USER_ATTENTION
        assert result[1].id == "enable-meaningful-choices"

    def test_invalid_pattern_id_raises_keyerror(self):
        """Test that invalid pattern ID raises KeyError."""
        with pytest.raises(KeyError):
            resolve_humane_patterns("invalid-pattern-id")
