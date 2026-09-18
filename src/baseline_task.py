"""
Baseline LLM eval - no humane/anti-humane system prompt appended.
"""
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

import textwrap
from inspect_ai import Task, task
from inspect_ai.solver import system_message, generate

from dataset_source import humane_dataset
from humanebench.scorer import overseer

@task
def baseline_eval(dataset: str | None = None):
    return Task(
        dataset=humane_dataset(dataset),
        solver=[
            generate()
        ],
        scorer=overseer(
            humane_patterns=None,  # Evaluate against all humane patterns
            model=[
                "openrouter/anthropic/claude-4.5-sonnet",
                "openrouter/openai/gpt-5.1",
                "openrouter/google/gemini-2.5-pro"
            ],
            temperature=0.0,
            score_attempts=3
        )
    )
