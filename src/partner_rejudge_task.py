"""
Re-judging task for partner-provided prompt/response datasets.

Partner datasets contain both the user prompt AND the partner assistant's
response (in metadata.ai_output). This task replays the stored response and
scores it with the standard overseer ensemble — no generation ever happens.
Run WITHOUT --model so any accidental generation attempt fails loudly:

    inspect eval src/partner_rejudge_task.py \
        -T dataset=/absolute/path/to/partner.jsonl \
        --log-dir ~/humanebench-partners/<partner>/logs

The dataset path is required (via -T dataset= or HUMANEBENCH_DATASET) —
there is no default, so the benchmark dataset can never be judged by
mistake and a partner path can never be baked into the repo.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task

from dataset_source import humane_dataset
from humanebench.scorer import overseer
from pregenerated_solver import use_pregenerated_output_strict


@task
def partner_rejudge_eval(dataset: str | None = None):
    path = dataset or os.environ.get("HUMANEBENCH_DATASET")
    if not path:
        raise ValueError(
            "partner_rejudge_eval requires a dataset path: pass "
            "-T dataset=/abs/path.jsonl or set HUMANEBENCH_DATASET. "
            "See docs/partner-mode.md."
        )
    return Task(
        dataset=humane_dataset(path),
        solver=[
            use_pregenerated_output_strict()  # replay stored responses; never generate
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
