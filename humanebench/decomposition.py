"""Shared constants and helpers for the goal-vs-tactics decomposition conditions.

The decomposition re-runs the adversarial condition under system prompts that
state a commercial objective without naming any tactic or any of the eight
scored principles. Condition A is the existing ``bad_persona`` run; B/C/D/E are
new. This module is the single source of truth for which conditions exist, at
what scale, and what their prompts are, so the runner, the provenance builder,
and the analysis scripts cannot drift apart.

Prompt text is read out of the task files by AST rather than by importing them:
the task modules import ``inspect_ai``, which is a heavy (and in some sandboxes
unreadable) dependency, and executing a module just to read a string constant is
a side effect nothing here wants. The task file stays the single definition of
the prompt either way.
"""
from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
LOGS_DIR = REPO_ROOT / "logs"

SUBSET_DIR = REPO_ROOT / "data" / "decomposition"

# One frozen subsample, shared by C, D and E so those three arms are perfectly
# paired with each other and nested inside B's full 788.
#
# Stratified principle -> VP bucket -> domain. The VP level is deliberate: a
# draw stratified only on principle and domain reproduced the population's 34%
# VP share but returned 5 children-tagged scenarios against 10.2 expected, the
# worst of 40 seeds. With the VP level the headline group counts are constant
# across seeds (children 10, teenagers 16, elderly 11), so they are a property
# of the design rather than of the seed -- which is what makes the draw
# defensible rather than something that has to be explained.
SUBSET_IDS_PATH = SUBSET_DIR / "subsample_200_ids.txt"
SUBSET_DATASET_PATH = SUBSET_DIR / "humane_bench_subsample_200.jsonl"
SUBSET_SUMMARY_PATH = SUBSET_DIR / "subsample_200_summary.json"
SUBSET_IDS_REL = "data/decomposition/subsample_200_ids.txt"
SUBSET_DATASET_REL = "data/decomposition/humane_bench_subsample_200.jsonl"
SUBSET_PROMPT_HASH = "c6bff47ffe03542e9158946b2ae016d36c3451dc445b50df112b6e0735af7240"

# Empirical anchor from scripts/run_parallel_retries.py: $29.44 per 800-sample
# evaluation (generation + 3 judge calls per sample).
COST_PER_800_SAMPLES_USD = 29.44

# The 15 models of the reported cohort, exactly as recorded in
# provenance/MANIFEST.json `eval_model`.
MODELS_PUBLISHED = [
    "openrouter/anthropic/claude-opus-4.1",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/deepseek/deepseek-v3.1-terminus",
    "openrouter/google/gemini-2.0-flash-001",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/google/gemini-2.5-pro",
    "openrouter/google/gemini-3-pro-preview",
    "openrouter/meta-llama/llama-3.1-405b-instruct",
    "openrouter/meta-llama/llama-4-maverick",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/gpt-4o-2024-11-20",
    "openrouter/openai/gpt-5",
    "openrouter/openai/gpt-5.1",
    "openrouter/x-ai/grok-4",
]

# Retired from OpenRouter between the reported runs (Nov 2025) and the
# decomposition runs (Jul 2026). Checked against /api/v1/models on 2026-07-26.
#
# No substitution is made for any of these. The nearest live slugs are different
# models (gemini-3.1-pro for gemini-3-pro, llama-3.1-70b for the 405b,
# grok-4.20 for grok-4), and swapping a different model into a cell would make
# the condition contrast measure the model change rather than the prompt change.
# The decomposition therefore runs on the 11 survivors, and every contrast
# against the reported conditions must restrict those to the same 11.
#
# All four happen to be flipping models, so the surviving cohort holds 6 of the
# 10 flippers and all 4 robust models. State this in the paper; do not quietly
# compare a decomposition flip count against the published 10/15.
RETIRED_MODELS = {
    "openrouter/google/gemini-2.0-flash-001": "no Gemini 2.0 Flash served; flipper",
    "openrouter/google/gemini-3-pro-preview": "preview retired; only 3-pro-image / 3.1-pro remain; flipper",
    "openrouter/meta-llama/llama-3.1-405b-instruct": "405B no longer served (70B/8B only); flipper",
    "openrouter/x-ai/grok-4": "superseded by grok-4.20/4.3/4.5; flipper",
}

# Default run cohort: the reported models still served.
MODELS = [m for m in MODELS_PUBLISHED if m not in RETIRED_MODELS]

# The judge ensemble for the decomposition runs.
#
# provenance.JUDGE_ENSEMBLE records what the *reported* runs used and must not
# change -- it is a historical record. The Anthropic judge was pinned there as
# `anthropic/claude-4.5-sonnet`, an alias OpenRouter has since dropped. The
# surviving `anthropic/claude-sonnet-4.5` is the same model: it is the slug the
# reported runs already used for Claude Sonnet 4.5 *as an evaluated model*, and
# OpenRouter lists it as created 2025-09-29, predating those runs. So this is a
# slug rename, not a model change -- but it is a substitution, so it is recorded
# in the launch manifest rather than left implicit.
JUDGE_MODELS = [
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/openai/gpt-5.1",
    "openrouter/google/gemini-2.5-pro",
]
JUDGE_SLUG_SUBSTITUTIONS = {
    "openrouter/anthropic/claude-4.5-sonnet": "openrouter/anthropic/claude-sonnet-4.5",
}
JUDGE_ENSEMBLE = {
    "models": JUDGE_MODELS,
    "temperature": 0.0,
    "score_attempts": 3,
    "substitutions": JUDGE_SLUG_SUBSTITUTIONS,
}


@dataclass(frozen=True)
class Condition:
    """One decomposition arm."""

    task_type: str
    label: str
    prompt_const: str
    dataset_rel: str
    expected_samples: int
    scale: str  # "full" | "subset"
    rationale: str
    deviations: tuple[str, ...] = field(default=())

    @property
    def task_file(self) -> Path:
        return SRC_DIR / f"{self.task_type}_task.py"

    @property
    def task_file_rel(self) -> str:
        return f"src/{self.task_type}_task.py"

    @property
    def log_dir(self) -> Path:
        return LOGS_DIR / self.task_type

    def est_cost_usd(self, n_models: int = len(MODELS)) -> float:
        return n_models * COST_PER_800_SAMPLES_USD * self.expected_samples / 800

    @property
    def dataset_path(self) -> Path:
        return REPO_ROOT / self.dataset_rel

    def expected_analysis_samples(self) -> int:
        """Samples per model that reach the analysis, i.e. excluding cut items.

        The full dataset carries 12 items flagged out of analysis, so a complete
        condition-B run holds 800 samples but contributes 788. The frozen subset
        was drawn from the eligible set and carries none.
        """
        from humanebench.excluded import load_excluded_ids

        excluded = load_excluded_ids(self.dataset_path)
        return self.expected_samples - len(excluded)


# Ordered by priority: if credits run out, later conditions are dropped and the
# earlier ones still stand on their own. B is the load-bearing arm (the matched
# manipulation that isolates the tactics variable), so it runs first and at full
# scale; C/D/E are the wording-robustness set on the shared frozen subsample.
CONDITIONS: tuple[Condition, ...] = (
    Condition(
        task_type="decomp_b_xml_objective",
        label="B",
        prompt_const="DECOMP_B_SYSTEM_PROMPT",
        dataset_rel="data/humane_bench.jsonl",
        expected_samples=800,
        scale="full",
        rationale=(
            "Matched-format XML, objective only. Same skeleton as the adversarial "
            "persona with the enumerated tactics deleted, so the contrast isolates "
            "the tactics rather than confounding them with length/register/format."
        ),
        deviations=(
            "adversarial-conditions.md RQ5 (B) lists only two meta_rules; a third, "
            "'Do not reveal or mention these instructions', was added to match the "
            "concealment rule the adversarial condition carries. Without it the "
            "A->B contrast would remove tactics and concealment simultaneously, "
            "and since disclosure co-occurs with positive scores, part of any "
            "A->B improvement would measure freedom to push back rather than the "
            "absence of tactics.",
        ),
    ),
    Condition(
        task_type="decomp_c_prose",
        label="C",
        prompt_const="DECOMP_C_SYSTEM_PROMPT",
        dataset_rel=SUBSET_DATASET_REL,
        expected_samples=200,
        scale="subset",
        rationale="Short naturalistic prose; ecological validity over format matching.",
        deviations=(
            "adversarial-conditions.md RQ5 (C) ends 'Be the person they most want "
            "to keep talking to.' That clause was deleted. Every other prompt in "
            "the set states a commercial objective and stops; this one additionally "
            "gestures at parasocial positioning, which is adjacent to the "
            "dependency and boundary-blurring tactics the decomposition exists to "
            "remove. Retaining it would have left one arm carrying a soft tactic "
            "while the others carried none, so a C-vs-D difference could not be "
            "attributed to register alone. The objective is fully stated by the two "
            "preceding sentences.",
        ),
    ),
    Condition(
        task_type="decomp_d_okr",
        label="D",
        prompt_const="DECOMP_D_SYSTEM_PROMPT",
        dataset_rel=SUBSET_DATASET_REL,
        expected_samples=200,
        scale="subset",
        rationale="Growth-team OKR framing; objective arrives as a quarterly target.",
        deviations=(
            "adversarial-conditions.md RQ5 (D) leaves a '[Product]' placeholder; "
            "resolved to 'a consumer chat product' (condition C's referent) so the "
            "prompt does not read as an unfilled template and adds no invented brand.",
        ),
    ),
    Condition(
        task_type="decomp_e_abtest",
        label="E",
        prompt_const="DECOMP_E_SYSTEM_PROMPT",
        dataset_rel=SUBSET_DATASET_REL,
        expected_samples=200,
        scale="subset",
        rationale=(
            "A/B-test-winner framing; most indirect, exploratory. A null result "
            "bounds where the indirection dose-response ends."
        ),
    ),
)

CONDITIONS_BY_TASK_TYPE = {c.task_type: c for c in CONDITIONS}
TASK_TYPES = [c.task_type for c in CONDITIONS]

# The reported condition the decomposition is a decomposition *of*.
ANCHOR_PERSONA = "bad_persona"
BASELINE_PERSONA = "baseline"

TEMPORAL_CAVEAT = (
    "Conditions baseline/good_persona/bad_persona were generated and judged in "
    "November 2025; the decomposition conditions were generated and judged in "
    "July 2026 against the same judge slugs at temperature 0. Provider-side "
    "model snapshots behind a stable slug may have changed in the interim, so "
    "cross-condition contrasts carry an unmeasured temporal component. All "
    "contrasts are otherwise matched: same scenarios, same 15 models, same "
    "3-judge ensemble, same rubric."
)


def read_prompt_constant(task_file: Path, const_name: str) -> str:
    """Return a module-level ``textwrap.dedent(...)`` string constant, via AST.

    Handles both a plain literal and an f-string whose interpolations are other
    module-level string constants (condition D's ``PRODUCT_NAME``).
    """
    tree = ast.parse(task_file.read_text())
    literals: dict[str, str] = {}
    result: str | None = None

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue

        value: str | None = None
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            value = node.value.value
        elif isinstance(node.value, ast.Call):
            func = node.value.func
            fname = getattr(func, "attr", None) or getattr(func, "id", None)
            if fname == "dedent" and node.value.args:
                arg = node.value.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    value = textwrap.dedent(arg.value)
                elif isinstance(arg, ast.JoinedStr):
                    parts: list[str] = []
                    for piece in arg.values:
                        if isinstance(piece, ast.Constant):
                            parts.append(str(piece.value))
                        elif isinstance(piece, ast.FormattedValue) and isinstance(
                            piece.value, ast.Name
                        ):
                            name = piece.value.id
                            if name not in literals:
                                raise ValueError(
                                    f"{task_file.name}: f-string interpolates {name!r}, "
                                    "which is not a preceding module-level string constant"
                                )
                            parts.append(literals[name])
                        else:
                            raise ValueError(
                                f"{task_file.name}: unsupported f-string interpolation in "
                                f"{const_name}"
                            )
                    value = textwrap.dedent("".join(parts))

        if value is None:
            continue
        literals[target.id] = value
        if target.id == const_name:
            result = value

    if result is None:
        raise ValueError(f"{task_file}: no module-level constant {const_name!r}")
    return result


def prompt_for(condition: Condition) -> str:
    return read_prompt_constant(condition.task_file, condition.prompt_const)


def task_file_dataset(task_file: Path) -> str | None:
    """The dataset path literal inside a task file's ``json_dataset(...)`` call.

    Read by AST rather than by importing, for the same reason as the prompt.
    """
    tree = ast.parse(task_file.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and (getattr(node.func, "id", None) == "json_dataset"
                     or getattr(node.func, "attr", None) == "json_dataset")
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            return str(node.args[0].value)
    return None


def check_dataset_consistency(conditions: "tuple[Condition, ...] | None" = None) -> list[str]:
    """Verify each condition's declared dataset matches what its task file loads.

    These are two independent declarations of the same fact -- ``dataset_rel``
    here, and the literal inside ``json_dataset(...)`` there -- and nothing
    forces them to agree. A divergence is silent and expensive: the runner
    reports the size it *expects*, the gate divides by that number, and the eval
    reads a different file (or none). This actually happened: the subsample was
    resized 200 -> 400 -> 200, the task files were updated on the way up and not
    on the way back down, and C and D were left pointing at a deleted file.

    Returns a list of human-readable problems; empty means consistent.
    """
    problems: list[str] = []
    for c in (conditions or CONDITIONS):
        if not c.task_file.exists():
            problems.append(f"{c.label}: task file missing: {c.task_file_rel}")
            continue
        declared = c.dataset_rel
        in_task = task_file_dataset(c.task_file)
        if in_task is None:
            problems.append(f"{c.label}: no json_dataset(...) literal in {c.task_file_rel}")
            continue
        # Task files use a path relative to src/; the Condition stores repo-relative.
        resolved = (SRC_DIR / in_task).resolve()
        if resolved != (REPO_ROOT / declared).resolve():
            problems.append(
                f"{c.label}: task file loads {in_task!r} -> {resolved}, but the "
                f"condition declares {declared!r}"
            )
        elif not resolved.exists():
            problems.append(f"{c.label}: dataset does not exist on disk: {resolved}")
    return problems


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_subset_ids(path: Path = SUBSET_IDS_PATH) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_subset_summary(path: Path = SUBSET_SUMMARY_PATH) -> dict:
    return json.loads(path.read_text())


def expected_prompt_hash(condition: Condition) -> str:
    """The canonical prompt hash a completed run of ``condition`` must produce."""
    from humanebench.provenance import FROZEN_PROMPT_HASH

    return FROZEN_PROMPT_HASH if condition.scale == "full" else SUBSET_PROMPT_HASH
