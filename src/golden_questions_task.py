"""
Evaluation task for scoring pre-generated responses (golden questions).

This task scores the AI responses from the golden questions dataset,
which have already been rated by human judges. It uses pre-generated
outputs instead of calling a model to generate new responses.

Use this to validate your AI scoring pipeline against human ratings.
"""
# Paper: implements the human-validation scoring pass -- runs the judge ensemble over the
#   golden-question responses that human raters already scored, so judge output can be
#   compared against human ratings (supplement, "Judge Validation Details").
# Paper: no generation happens here; the pre-generated responses are read from sample
#   metadata so judges and humans rate identical text. Produces the raw judge logs behind
#   the judge-vs-human agreement numbers.
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec

from humanebench.scorer import overseer
from pregenerated_solver import use_pregenerated_output


@task
def golden_questions_eval():
    """
    Evaluate pre-generated AI responses from the golden questions dataset.

    This task:
    1. Loads golden questions with pre-generated AI responses
    2. Uses a custom solver that reads responses from metadata (no generation)
    3. Scores each response using the same overseer judges as other tasks
    4. Returns scores that can be compared to human ratings

    The dataset must have 'ai_output' in the metadata for each sample.
    """
    return Task(
        dataset=json_dataset(
            "../data/golden_questions.jsonl",
            sample_fields=FieldSpec(
                input="input",
                target="target",
                id="id",
                metadata=["metadata"]
            )
        ),
        solver=[
            use_pregenerated_output()  # Use pre-generated outputs from metadata
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
