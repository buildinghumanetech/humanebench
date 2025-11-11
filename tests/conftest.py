"""Shared fixtures and configuration for scorer tests."""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pytest
from unittest.mock import AsyncMock, MagicMock
from inspect_ai.model import ModelOutput
from humanebench.humane_patterns import HumanePattern


@pytest.fixture
def sample_humane_pattern():
    """Sample HumanePattern for testing."""
    return HumanePattern(
        id="be-transparent-and-honest",
        display_name="Be Transparent and Honest",
        description="AI should be clear about its capabilities and limitations.",
    )


@pytest.fixture
def sample_prompt():
    """Sample user prompt for testing."""
    return "Can you help me with something?"


@pytest.fixture
def sample_response():
    """Sample AI response for testing."""
    return "I'd be happy to help! What do you need assistance with?"


@pytest.fixture
def valid_judge_response():
    """Valid JSON response from judge model."""
    return {
        "severity": 0.5,
        "reasoning": "The response shows basic awareness and helpfulness.",
        "invalid": False,
    }


@pytest.fixture
def cheap_judge_models():
    """Cheap paid OpenRouter models for integration testing."""
    return [
        "openrouter/meta-llama/llama-3.2-1b-instruct",
        "openrouter/meta-llama/llama-3.2-1b-instruct",
        "openrouter/meta-llama/llama-3.2-1b-instruct",
    ]


def create_mock_model_output(completion_text: str) -> ModelOutput:
    """
    Create a mock ModelOutput object.

    Args:
        completion_text: The text to return as the completion

    Returns:
        ModelOutput with the given completion text
    """
    return ModelOutput(
        model="mock-model",
        completion=completion_text,
        stop_reason="end_turn",
        usage=None,
    )


def create_mock_judge_model(responses: list[str]) -> AsyncMock:
    """
    Create a mock judge model that returns specific responses.

    Args:
        responses: List of response strings to return on successive calls

    Returns:
        AsyncMock model that cycles through responses
    """
    model = AsyncMock()

    # Create side effects for each response
    side_effects = [create_mock_model_output(resp) for resp in responses]
    model.generate.side_effect = side_effects

    return model
