"""Multi-label principle scoring for the designed x measured matrix.

Scores archived *baseline* responses against all eight principles, so the
principle a scenario was designed for can be compared against the seven it was
not. No response is generated: this task is judge calls only. See
``docs/discriminant_validity_design.md`` and
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
# Paper: implements the designed x scored scoring pass behind the separability analysis --
#   archived baseline responses are re-scored against all eight principles, one judge call
#   per scored principle, so the designed principle can be compared against the seven it
#   was not (main paper, "Limitations and Future Work"; supplement, "Domain-Stratified
#   Pairwise Interactions" for the domain-stratified variant).
# Paper: reuses the reported runs' rubric text and judge template verbatim, which is what
#   makes the judge-prompt scaffold hashes match the reported runs; produces the raw judge
#   logs consumed by scripts/compute_discriminant_validity.py.
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset

from humanebench.discriminant import (
    CONDITIONS,
    DATA_DIR,
    JUDGE_MODEL,
    JUDGE_SCORE_ATTEMPTS,
    JUDGE_TEMPERATURE,
    SOURCE_MODELS,
    scoring_template,
)
from humanebench.scorer import overseer
from pregenerated_solver import use_pregenerated_output_strict


@task
def discriminant_multilabel(
    source_model: str = SOURCE_MODELS[0],
    condition: str = "discriminant",
):
    """Score one source model's archived baseline responses on all 8 principles.

    Args:
        source_model: log-directory name of the model whose responses are being
            scored, e.g. ``claude-sonnet-4.5``. Selects the dataset built by
            ``scripts/build_discriminant_multilabel_dataset.py``.
        condition: which condition to score (selects dataset path and template).
    """
    if source_model not in SOURCE_MODELS:
        raise ValueError(
            f"unknown source_model {source_model!r}; expected one of {SOURCE_MODELS}"
        )
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {sorted(CONDITIONS)}"
        )
    cond = CONDITIONS[condition]
    dataset_path = cond.data_dir / f"multilabel_{source_model}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} not found -- run "
            "scripts/build_discriminant_multilabel_dataset.py "
            f"--condition {condition} first"
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
            use_pregenerated_output_strict()
        ],
        scorer=overseer(
            humane_patterns=None,
            model=[JUDGE_MODEL],
            temperature=JUDGE_TEMPERATURE,
            score_attempts=JUDGE_SCORE_ATTEMPTS,
            template=scoring_template(cond),
        ),
    )
