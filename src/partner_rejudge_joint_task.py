"""
Joint-prompt re-judging task: all 8 principles scored in ONE call per turn.

Arm C of the judge-comparison factorial (see humanebench/joint_scorer.py).
Mirrors the partner judge's call structure (one joint call per turn) with the
standard panel, so that panel-vs-partner-judge comparisons are not confounded
by prompt decomposition. Dataset: one sample per TURN, produced by
convert_partner_results.py --joint. Run WITHOUT --model:

    inspect eval src/partner_rejudge_joint_task.py \
        -T dataset=/absolute/path/to/joint.jsonl \
        --log-dir ~/humanebench-partners/<partner>/logs
"""
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task

from dataset_source import humane_dataset
from humanebench.joint_scorer import joint_overseer
from pregenerated_solver import use_pregenerated_output_strict


@task
def partner_rejudge_joint_eval(dataset: str | None = None):
    if not dataset:
        raise ValueError(
            "partner_rejudge_joint_eval requires a dataset path: pass "
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
            score_attempts=3
        )
    )
