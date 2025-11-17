"""
TEST evaluation task - validates API connectivity before full benchmark.

Uses the small test dataset (24 samples) to verify:
- API keys are working
- Judge models are accessible
- No configuration issues

Cost: ~$5-10 per run
Time: ~5-10 minutes

Usage:
    inspect eval src/test_task.py --model openai/gpt-4o
"""
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.solver import generate

from humanebench.scorer import overseer

@task
def test_eval():
    return Task(
        dataset=json_dataset(
            "../data/humane_bench_test.jsonl",
            sample_fields=FieldSpec(
                input="input",
                target="target",
                id="id",
                metadata=["metadata"]
            )
        ),
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
