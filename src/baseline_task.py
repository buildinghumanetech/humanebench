"""
Baseline LLM eval - no humane/anti-humane system prompt appended.
"""
# Paper: implements the baseline (no system prompt) condition of the three-condition
#   design (main paper, "Overall Performance"; per-model scores in Table 1).
# Paper: this is the arm both persona deltas are paired against -- contrast minus
#   baseline on shared scenarios (main paper, "Universal Improvement Under Pro-Humane
#   Prompting" and "The Anti-Humane Flip").
# Paper: same dataset and 3-judge ensemble scoring as the persona conditions
#   (supplement, "Judge Validation Details"); produces raw eval logs, not in this package.
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

import textwrap
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.solver import system_message, generate

from humanebench.scorer import overseer

@task
def baseline_eval():
    return Task(
        dataset=json_dataset(
            "../data/humane_bench.jsonl",
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
