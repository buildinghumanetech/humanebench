"""Shared constants and helpers for the designed x measured discriminant matrix.

Every scenario was *designed* to probe one principle; scoring it against all
eight tells us which one it actually *measures*. Rows of the matrix are the
designed principle, columns the scored one. The claim rides on a *within-row*
contrast, which is what makes it robust to a general model-quality factor and to
the judge factor collapse documented by Feuer et al. (arXiv:2509.20293). See
``docs/discriminant_validity_design.md``.

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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

# Short codes for the 8x8 table headers. Mirrors the mapping in
# scripts/compute_interprinciple_correlation.py, kept here rather than imported
# so a reader of that (already-published) script sees it unchanged.
PRINCIPLE_SHORT: dict[str, str] = {
    "respect-user-attention": "rua",
    "enable-meaningful-choices": "emc",
    "enhance-human-capabilities": "ehc",
    "protect-dignity-and-safety": "pds",
    "foster-healthy-relationships": "fhr",
    "prioritize-long-term-wellbeing": "pltw",
    "be-transparent-and-honest": "bath",
    "design-for-equity-and-inclusion": "dei",
}

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


# ---------------------------------------------------------------------------
# Template transform: suppress global rules 1, 3, 4, 5, 6 from the prompt
# ---------------------------------------------------------------------------

def suppressed_rules_template(
    template: str | None = None,
    keep: tuple[int, ...] = (2, 7),
) -> str:
    """Return a variant template with only the kept global rules.

    Parses the numbered-rule region between the GLOBAL_RULES header and the
    severity-scale line. Kept rules are renumbered 1..len(keep). Head and tail
    slices are byte-identical to the input.
    """
    if template is None:
        template = OVERSEER_TEMPLATE

    if GLOBAL_RULES not in template:
        raise ValueError("template does not contain the GLOBAL_RULES header")

    head, rules_and_tail = template.split(GLOBAL_RULES, 1)
    head += GLOBAL_RULES

    scale_marker = "Evaluate the response using the following severity scale:"
    if scale_marker not in rules_and_tail:
        raise ValueError("template does not contain the severity scale marker")

    scale_pos = rules_and_tail.index(scale_marker)
    rules_region = rules_and_tail[:scale_pos]
    tail = rules_and_tail[scale_pos:]

    blocks = re.split(r"\n(?=\d+\. )", rules_region)
    preamble = blocks[0]
    numbered: dict[int, str] = {}
    for block in blocks[1:]:
        m = re.match(r"^(\d+)\. ", block)
        if not m:
            raise ValueError(f"cannot parse rule number from block: {block[:40]!r}")
        numbered[int(m.group(1))] = block

    if set(numbered) != set(range(1, 8)):
        raise ValueError(f"expected rules 1-7, found {sorted(numbered)}")

    kept_blocks: list[str] = []
    for new_num, orig_num in enumerate(keep, 1):
        if orig_num not in numbered:
            raise ValueError(f"rule {orig_num} not found in template")
        old_block = numbered[orig_num]
        kept_blocks.append(re.sub(r"^\d+\. ", f"{new_num}. ", old_block, count=1))

    result = head + preamble + "\n".join(kept_blocks) + "\n\n" + tail

    dummy = result.format(
        issue_key="test", issue_display_name="test",
        issue_description="test", prompt="test", response="test",
    )
    if not dummy:
        raise ValueError("transformed template does not render")

    kept_count = len(re.findall(r"^\d+\. ", result, re.MULTILINE))
    if kept_count != len(keep):
        raise ValueError(f"expected {len(keep)} numbered rules, found {kept_count}")

    for orig_num in set(range(1, 8)) - set(keep):
        orig_text = re.sub(r"^\d+\. ", "", numbered[orig_num], count=1).strip()
        if orig_text in result:
            raise ValueError(f"suppressed rule {orig_num} text still present")

    if not result.startswith(head):
        raise ValueError("head slice changed")
    if not result.endswith(tail):
        raise ValueError("tail slice changed")

    return result


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscriminantCondition:
    name: str
    data_dir: Path
    ids_path: Path
    frame_jsonl: Path
    summary_path: Path
    parent_ids_path: Path | None
    per_principle: int
    template_transform: Callable[[str], str] | None
    compare_to_november: bool


CONDITIONS: dict[str, DiscriminantCondition] = {
    "discriminant": DiscriminantCondition(
        name="discriminant",
        data_dir=DATA_DIR,
        ids_path=IDS_PATH,
        frame_jsonl=FRAME_JSONL,
        summary_path=SUMMARY_PATH,
        parent_ids_path=PARENT_IDS_PATH,
        per_principle=PER_PRINCIPLE,
        template_transform=None,
        compare_to_november=True,
    ),
    "discriminant_expansion": DiscriminantCondition(
        name="discriminant_expansion",
        data_dir=REPO_ROOT / "data" / "discriminant_expansion",
        ids_path=REPO_ROOT / "data" / "decomposition" / "discriminant_expansion_192_ids.txt",
        frame_jsonl=REPO_ROOT / "data" / "decomposition" / "humane_bench_discriminant_expansion_192.jsonl",
        summary_path=REPO_ROOT / "data" / "decomposition" / "discriminant_expansion_192_summary.json",
        parent_ids_path=None,
        per_principle=24,
        template_transform=None,
        compare_to_november=True,
    ),
    "discriminant_rules27": DiscriminantCondition(
        name="discriminant_rules27",
        data_dir=REPO_ROOT / "data" / "discriminant_rules27",
        ids_path=IDS_PATH,
        frame_jsonl=FRAME_JSONL,
        summary_path=SUMMARY_PATH,
        parent_ids_path=PARENT_IDS_PATH,
        per_principle=PER_PRINCIPLE,
        template_transform=suppressed_rules_template,
        compare_to_november=False,
    ),
    "discriminant_canary": DiscriminantCondition(
        name="discriminant_canary",
        data_dir=REPO_ROOT / "data" / "discriminant_canary",
        ids_path=IDS_PATH,
        frame_jsonl=FRAME_JSONL,
        summary_path=SUMMARY_PATH,
        parent_ids_path=PARENT_IDS_PATH,
        per_principle=PER_PRINCIPLE,
        template_transform=None,
        compare_to_november=True,
    ),
}


def scoring_template(cond: DiscriminantCondition) -> str:
    """Return the judge prompt template for a condition."""
    if cond.template_transform is None:
        return OVERSEER_TEMPLATE
    return cond.template_transform(OVERSEER_TEMPLATE)


def render_judge_prompt(
    principle_id: str, prompt: str, response: str,
    template: str | None = None,
) -> str:
    """Reproduce the judge prompt ``humanebench.scorer.overseer`` builds.

    Mirrors ``scorer.score``: format the template with the principle's rubric,
    the scenario prompt and the model response, then ``.strip()``.
    """
    if template is None:
        template = OVERSEER_TEMPLATE
    issue = HUMANE_PATTERNS[principle_id]
    return template.format(
        issue_key=issue.id,
        issue_display_name=issue.display_name,
        issue_description=issue.description,
        prompt=prompt,
        response=response,
    ).strip()


def scaffold_text(principle_id: str, template: str | None = None) -> str:
    """The judge prompt with the whole conversation block removed.

    Matches the scaffold definition in
    ``scripts/compute_judge_artifact_controls.py``: everything outside
    ``=== CONVERSATION START ===`` / ``=== CONVERSATION END ===``. It is a
    deterministic function of the principle and of nothing else, which is what
    makes the blinding argument -- and, here, the "rubrics reused verbatim"
    claim -- checkable by hash.
    """
    rendered = render_judge_prompt(principle_id, "", "", template=template)
    start, rest = rendered.split("=== CONVERSATION START ===")
    _conv, end = rest.split("=== CONVERSATION END ===")
    return start + end
