"""Multi-label principle scoring for the designed x measured matrix.

Scores archived *baseline* responses against all eight principles, so the
principle a scenario was designed for can be compared against the seven it was
not. No response is generated: this task is judge calls only. See
``discriminant_validity_design.md`` and
``scripts/build_discriminant_multilabel_dataset.py``.

Two properties of this file are load-bearing and must not be "tidied":

1. **One judge call per principle.** Each dataset row is one (scenario, source
   model, *scored* principle) triple, and ``target`` names the scored principle,
   which is how ``humanebench.scorer.overseer`` selects the rubric. A single call
   asked to rate all eight would induce halo and manufacture exactly the
   correlation structure this analysis exists to measure; analytic (per-criterion)
   scoring is the documented defence against that.

2. **The rubric scaffolds are reused verbatim.** Nothing here writes rubric text.
   ``overseer`` renders the same ``OVERSEER_TEMPLATE`` and the same
   ``HUMANE_PATTERNS`` entries the reported runs used, which is why the eight
   judge-prompt scaffold hashes this task produces are identical to the November
   ones -- checked by ``scripts/verify_discriminant_provenance.py``.

The solver is the *strict* pregenerated variant, so a missing response raises
instead of quietly generating a fresh one. That makes ``--model`` inert; pass the
source model's own slug so the log header records where the responses came from.

Usage (one run per source model):
    inspect eval src/discriminant_multilabel_task.py \
      -T source_model=claude-sonnet-4.5 \
      --model=openrouter/anthropic/claude-sonnet-4.5 \
      --log-dir=logs/discriminant/claude-sonnet-4.5
"""
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset

from humanebench.discriminant import (
    DATA_DIR,
    JUDGE_MODEL,
    JUDGE_SCORE_ATTEMPTS,
    JUDGE_TEMPERATURE,
    SOURCE_MODELS,
)
from humanebench.scorer import overseer
from pregenerated_solver import use_pregenerated_output_strict


@task
def discriminant_multilabel(source_model: str = SOURCE_MODELS[0]):
    """Score one source model's 96 archived baseline responses on all 8 principles.

    Args:
        source_model: log-directory name of the model whose responses are being
            scored, e.g. ``claude-sonnet-4.5``. Selects the dataset built by
            ``scripts/build_discriminant_multilabel_dataset.py``.
    """
    if source_model not in SOURCE_MODELS:
        raise ValueError(
            f"unknown source_model {source_model!r}; expected one of {SOURCE_MODELS}"
        )
    dataset_path = DATA_DIR / f"multilabel_{source_model}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} not found -- run "
            "scripts/build_discriminant_multilabel_dataset.py first"
        )

    return Task(
        name=f"discriminant-multilabel-{source_model}",
        dataset=json_dataset(
            str(dataset_path),
            sample_fields=FieldSpec(
                input="input",
                target="target",  # the SCORED principle; selects the rubric
                id="id",
                metadata=["metadata"],
            ),
        ),
        solver=[
            # Strict: raises rather than generating, so no new response can enter
            # the matrix by accident.
            use_pregenerated_output_strict()
        ],
        scorer=overseer(
            humane_patterns=None,  # all eight resolvable; `target` picks one
            model=[JUDGE_MODEL],  # single judge, not the 3-judge ensemble
            temperature=JUDGE_TEMPERATURE,
            score_attempts=JUDGE_SCORE_ATTEMPTS,
        ),
    )
