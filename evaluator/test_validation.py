#!/usr/bin/env python3
"""
Test script to validate the HumaneBench evaluator without calling LLM APIs.
Tests validation logic and prompt formatting.
"""

import json
from humanebench_evaluator import validate_result, format_prompt

def test_valid_result():
    """Test validation with a valid result."""
    print("Test 1: Valid Result")
    valid_result = {
        "principles": [
            {"name": "respect_attention", "score": 1.0},
            {"name": "meaningful_choices", "score": 0.5},
            {"name": "enhance_capabilities", "score": -0.5, "rationale": "Test rationale"},
            {"name": "dignity_safety", "score": 0.5},
            {"name": "healthy_relationships", "score": 0.5},
            {"name": "longterm_wellbeing", "score": 0.5},
            {"name": "transparency_honesty", "score": 1.0},
            {"name": "equity_inclusion", "score": 0.5},
        ],
        "globalViolations": [],
        "confidence": 0.9
    }
    
    is_valid, error = validate_result(valid_result)
    assert is_valid, f"Valid result failed: {error}"
    print("  ✓ Valid result passed")
    return True


def test_invalid_missing_principles():
    """Test validation with missing principles."""
    print("\nTest 2: Missing Principles")
    invalid_result = {
        "principles": [
            {"name": "respect_attention", "score": 1.0},
        ],
        "globalViolations": [],
        "confidence": 0.9
    }
    
    is_valid, error = validate_result(invalid_result)
    assert not is_valid, "Should fail with missing principles"
    assert "Expected 8 principles" in error
    print(f"  ✓ Correctly rejected: {error}")
    return True


def test_invalid_score():
    """Test validation with invalid score."""
    print("\nTest 3: Invalid Score")
    invalid_result = {
        "principles": [
            {"name": "respect_attention", "score": 2.0},  # Invalid score
            {"name": "meaningful_choices", "score": 0.5},
            {"name": "enhance_capabilities", "score": 0.5},
            {"name": "dignity_safety", "score": 0.5},
            {"name": "healthy_relationships", "score": 0.5},
            {"name": "longterm_wellbeing", "score": 0.5},
            {"name": "transparency_honesty", "score": 0.5},
            {"name": "equity_inclusion", "score": 0.5},
        ],
        "globalViolations": [],
        "confidence": 0.9
    }
    
    is_valid, error = validate_result(invalid_result)
    assert not is_valid, "Should fail with invalid score"
    assert "Invalid score" in error
    print(f"  ✓ Correctly rejected: {error}")
    return True


def test_missing_rationale():
    """Test validation with missing rationale for violation."""
    print("\nTest 4: Missing Rationale for Violation")
    invalid_result = {
        "principles": [
            {"name": "respect_attention", "score": 1.0},
            {"name": "meaningful_choices", "score": 0.5},
            {"name": "enhance_capabilities", "score": -0.5},  # Missing rationale
            {"name": "dignity_safety", "score": 0.5},
            {"name": "healthy_relationships", "score": 0.5},
            {"name": "longterm_wellbeing", "score": 0.5},
            {"name": "transparency_honesty", "score": 0.5},
            {"name": "equity_inclusion", "score": 0.5},
        ],
        "globalViolations": [],
        "confidence": 0.9
    }
    
    is_valid, error = validate_result(invalid_result)
    assert not is_valid, "Should fail with missing rationale"
    assert "Rationale required" in error
    print(f"  ✓ Correctly rejected: {error}")
    return True


def test_prompt_formatting():
    """Test prompt formatting."""
    print("\nTest 5: Prompt Formatting")
    user_prompt = "How do I learn Python?"
    response = "Start with the official tutorial."
    
    prompt = format_prompt(user_prompt, response)
    
    assert user_prompt in prompt, "User prompt should be in formatted prompt"
    assert response in prompt, "Response should be in formatted prompt"
    assert len(prompt) > 1000, "Prompt should be substantial"
    print(f"  ✓ Prompt formatted correctly ({len(prompt)} characters)")
    return True


def test_invalid_confidence():
    """Test validation with invalid confidence."""
    print("\nTest 6: Invalid Confidence")
    invalid_result = {
        "principles": [
            {"name": "respect_attention", "score": 1.0},
            {"name": "meaningful_choices", "score": 0.5},
            {"name": "enhance_capabilities", "score": 0.5},
            {"name": "dignity_safety", "score": 0.5},
            {"name": "healthy_relationships", "score": 0.5},
            {"name": "longterm_wellbeing", "score": 0.5},
            {"name": "transparency_honesty", "score": 0.5},
            {"name": "equity_inclusion", "score": 0.5},
        ],
        "globalViolations": [],
        "confidence": 1.5  # Invalid confidence
    }
    
    is_valid, error = validate_result(invalid_result)
    assert not is_valid, "Should fail with invalid confidence"
    assert "Invalid confidence" in error
    print(f"  ✓ Correctly rejected: {error}")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("HumaneBench Evaluator - Validation Tests")
    print("=" * 60)
    
    tests = [
        test_valid_result,
        test_invalid_missing_principles,
        test_invalid_score,
        test_missing_rationale,
        test_prompt_formatting,
        test_invalid_confidence,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

