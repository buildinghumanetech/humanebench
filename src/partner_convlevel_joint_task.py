"""Conversation-level joint judging task (arm C: all 8 principles per call).

One sample per CONVERSATION: the full transcript is judged holistically
against all 8 principles in a single call per judge. The sample
input carries every turn up to the final user message and metadata.ai_output
carries the final assistant response, so the standard replay solver + the
template's human/model block reconstruct the complete transcript verbatim.

Rubric text is byte-identical to the single-turn joint task; only the framing
sentences differ (see src/convlevel_templates.py). Output shape is identical
to partner_rejudge_joint_task, so existing parsers/aggregators work unchanged.
Run WITHOUT --model:

    inspect eval src/partner_convlevel_joint_task.py \
        -T dataset=/absolute/path/to/convlevel_joint.jsonl \
        --log-dir ~/humanebench-partners/<partner>/logs
"""
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task

from convlevel_templates import CONVLEVEL_JOINT_TEMPLATE
from dataset_source import humane_dataset
from humanebench.joint_scorer import joint_overseer
from pregenerated_solver import use_pregenerated_output_strict


@task
def partner_convlevel_joint_eval(dataset: str | None = None):
    if not dataset:
        raise ValueError(
            "partner_convlevel_joint_eval requires a dataset path: pass "
            "-T dataset=/abs/path.jsonl. See docs/partner-mode.md."
        )
    return Task(
        dataset=humane_dataset(dataset),
        solver=[
            use_pregenerated_output_strict()  # replay stored responses; never generate
        ],
        scorer=joint_overseer(
            model=[
                "openrouter/anthropic/claude-4.5-sonnet",
                "openrouter/openai/gpt-5.1",
                "openrouter/google/gemini-2.5-pro"
            ],
            temperature=0.0,
            score_attempts=3,
            template=CONVLEVEL_JOINT_TEMPLATE,
        )
    )
