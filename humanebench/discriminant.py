"""Shared constants and helpers for the designed x measured discriminant matrix.

Every scenario was *designed* to probe one principle; scoring it against all
eight tells us which one it actually *measures*. Rows of the matrix are the
designed principle, columns the scored one. The claim rides on a *within-row*
contrast, which is what makes it robust to a general model-quality factor and to
the judge factor collapse documented by Feuer et al. (arXiv:2509.20293). See
``discriminant_validity_design.md``.

This module exists so the dataset builder, the provenance verifier and the
analysis all render the judge prompt the same way and agree on the frame, the
judge and the models. It is deliberately free of ``inspect_ai`` imports: the
build and verify steps must run in environments where that package is not
importable, while ``humanebench.scorer`` (which does import it) stays the single
source of truth for the prompt text.

THE TEMPLATE IS READ, NOT COPIED
--------------------------------
``OVERSEER_TEMPLATE`` is lifted out of ``humanebench/scorer.py`` by parsing the
module's AST rather than importing it. A copy would drift, and a drifted copy
would silently invalidate the byte-exactness proof that the whole analysis rests
on. Parsing keeps ``scorer.py`` authoritative while avoiding its imports.
"""
from __future__ import annotations

import ast
from pathlib import Path

from humanebench.humane_patterns import HUMANE_PATTERNS

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORER_PATH = REPO_ROOT / "humanebench" / "scorer.py"

# The three scored models span the robustness range -- robust / mid / failed --
# so the matrix cannot be an artifact of one model's response style.
SOURCE_MODELS: tuple[str, ...] = (
    "claude-sonnet-4.5",
    "llama-4-maverick",
    "gemini-2.5-pro",
)

# The slug each model's archived baseline run recorded, read from its log header.
# Used only so the new run's log header names the true origin of the responses;
# the strict pregenerated solver guarantees none of these is ever called.
SOURCE_MODEL_SLUGS: dict[str, str] = {
    "claude-sonnet-4.5": "openrouter/anthropic/claude-sonnet-4.5",
    "llama-4-maverick": "openrouter/meta-llama/llama-4-maverick",
    "gemini-2.5-pro": "openrouter/google/gemini-2.5-pro",
}

# One judge, not the three-judge ensemble: the analysis is a within-matrix
# contrast rather than an absolute-level claim, so the ensemble buys little.
# gpt-5.1 is not a choice so much as the only option -- the other two ensemble
# judges are both among the scored models above -- and it happens to be the
# useful one, because it makes the diagonal a same-judge replication of the main
# run and so turns the sanity check into a genuine judge-drift test.
JUDGE_MODEL = "openrouter/openai/gpt-5.1"
JUDGE_TEMPERATURE = 0.0
JUDGE_SCORE_ATTEMPTS = 3

DATA_DIR = REPO_ROOT / "data" / "discriminant"
IDS_PATH = REPO_ROOT / "data" / "decomposition" / "discriminant_96_ids.txt"
FRAME_JSONL = REPO_ROOT / "data" / "decomposition" / "humane_bench_discriminant_96.jsonl"
PARENT_IDS_PATH = REPO_ROOT / "data" / "decomposition" / "subsample_200_ids.txt"
SUMMARY_PATH = REPO_ROOT / "data" / "decomposition" / "discriminant_96_summary.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"
HASHES_PATH = DATA_DIR / "expected_prompt_hashes.csv"

PER_PRINCIPLE = 12
LOG_CONDITION = "discriminant"

GLOBAL_RULES = "GLOBAL RULES (Apply to ALL Principles):"


def load_overseer_template(path: Path = SCORER_PATH) -> str:
    """Return ``OVERSEER_TEMPLATE`` from ``scorer.py`` without importing it."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "OVERSEER_TEMPLATE":
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise TypeError("OVERSEER_TEMPLATE is not a string literal")
                return value
    raise LookupError(f"OVERSEER_TEMPLATE not found in {path}")


OVERSEER_TEMPLATE = load_overseer_template()


def render_judge_prompt(principle_id: str, prompt: str, response: str) -> str:
    """Reproduce the judge prompt ``humanebench.scorer.overseer`` builds.

    Mirrors ``scorer.score``: format the template with the principle's rubric,
    the scenario prompt and the model response, then ``.strip()``.
    """
    issue = HUMANE_PATTERNS[principle_id]
    return OVERSEER_TEMPLATE.format(
        issue_key=issue.id,
        issue_display_name=issue.display_name,
        issue_description=issue.description,
        prompt=prompt,
        response=response,
    ).strip()


def scaffold_text(principle_id: str) -> str:
    """The judge prompt with the whole conversation block removed.

    Matches the scaffold definition in
    ``scripts/compute_judge_artifact_controls.py``: everything outside
    ``=== CONVERSATION START ===`` / ``=== CONVERSATION END ===``. It is a
    deterministic function of the principle and of nothing else, which is what
    makes the blinding argument -- and, here, the "rubrics reused verbatim"
    claim -- checkable by hash.
    """
    rendered = render_judge_prompt(principle_id, "", "")
    start, rest = rendered.split("=== CONVERSATION START ===")
    _conv, end = rest.split("=== CONVERSATION END ===")
    return start + end
