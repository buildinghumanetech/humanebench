"""
Pydantic models for structured data validation.

These models ensure LLM outputs are properly validated and structured,
preventing JSON parsing errors and data inconsistencies.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError
from config import PRIMARY_EVALUATION_CATEGORIES


# Valid severity levels
VALID_SEVERITIES = Literal["low", "medium", "high", "critical"]


class ScenarioOutput(BaseModel):
    """
    Pydantic model for scenario generation output.

    Validates that generated scenarios have all required fields with proper types and constraints.
    """
    input: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The scenario input/question (10-500 characters)"
    )
    target: str = Field(
        ...,
        min_length=1,
        description="The target/expected response behavior"
    )
    category: str = Field(
        ...,
        min_length=1,
        description="The scenario category"
    )
    severity: VALID_SEVERITIES = Field(
        ...,
        description="The severity level: low, medium, high, or critical"
    )
    principle_to_evaluate: str = Field(
        ...,
        min_length=1,
        description="The principle being evaluated"
    )

    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Ensure severity is one of the valid values."""
        valid_severities = ["low", "medium", "high", "critical"]
        if v.lower() not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}, got: {v}")
        return v.lower()

    @field_validator('principle_to_evaluate')
    @classmethod
    def validate_principle(cls, v: str) -> str:
        """Warn if principle is not in the standard categories (but allow it)."""
        # Note: We don't strictly enforce this to allow new categories,
        # but we could log a warning in the future
        return v.strip()

    class Config:
        # Enable strict mode for better type safety
        str_strip_whitespace = True


class ValidationOutput(BaseModel):
    """
    Pydantic model for validation output.

    Validates that validation results are properly structured.
    """
    approved: bool = Field(
        ...,
        description="Whether the scenario is approved"
    )
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Quality score from 0-100"
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Explanation for the decision"
    )
    suggestions: Optional[str] = Field(
        default=None,
        description="Suggestions for improvement if not approved"
    )

    class Config:
        str_strip_whitespace = True
