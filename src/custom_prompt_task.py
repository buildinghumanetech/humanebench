"""
Test your own system prompt against a no-system-prompt baseline, under rubric v4.

    inspect eval src/custom_prompt_task.py -T system_prompt_file=prompt.md \\
        --model openrouter/<provider>/<model>

This file defines two tasks and the command above runs both on the same model
and the same samples:

- custom_prompt_eval: your prompt as the system message
- baseline_v4_eval: no system message

Then compare them with scripts/compare_prompt.py.

-T tier picks the judges and the default sample size:

- try (default): one judge (TRY_JUDGE), 3 samples per principle. A first look.
- full: the three-judge ensemble, 10 samples per principle. Run this before you
  act on a result.

-T per_principle overrides the tier's sample size. Tier, judges and
per_principle are recorded in the log metadata, and compare_prompt.py refuses
to compare runs from different tiers.

These score under rubric v4 (humanebench/scorer_v4.py). They are not comparable
to the published v1 leaderboard, which baseline_task.py, good_persona_task.py
and bad_persona_task.py reproduce under v3.
"""
import hashlib
import os
import random
import sys
from pathlib import Path

# Add parent directory to path to import humanebench
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample, json_dataset, FieldSpec
from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from humanebench.scorer_v4 import (
    DEFAULT_JUDGES,
    JUDGE_PROMPT_PATH,
    RUBRIC_VERSION,
    load_judge_prompt,
    overseer_v4,
    rubric_sha256,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "humane_bench.jsonl"
DEFAULT_SEED = 42

# The try tier's single judge: best agreement on the v4.1 golden set, then the
# cheapest per sample (docs/validation/golden_v4.1_direction_match_2026-09-24.json).
# Sonnet 4.5 and Gemini 2.5 Pro tie at 22 of 23 direction matches (GPT-5.1: 20),
# and Sonnet costs less per sample ($0.046 against $0.052).
TRY_JUDGE = "openrouter/anthropic/claude-sonnet-4.5"
TIERS: dict[str, dict] = {
    "try": {"judges": [TRY_JUDGE], "per_principle": 3},
    "full": {"judges": list(DEFAULT_JUDGES), "per_principle": 10},
}
DEFAULT_TIER = "try"
_default_notice_shown = False


class PromptFileError(ValueError):
    pass


def resolve_prompt_path(path: str | os.PathLike, invocation_dir: str | None = None) -> Path:
    """Resolve a prompt path against the directory `inspect eval` was run from.

    Inspect changes into the task file's directory while it builds tasks, so a
    relative path must not be resolved against the process cwd.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    base = invocation_dir or os.environ.get("PWD") or os.getcwd()
    return Path(base) / p


def load_system_prompt(path: str | os.PathLike | None, invocation_dir: str | None = None) -> tuple[str, str]:
    """Read the prompt file. Returns (text, sha256). Raises PromptFileError."""
    if path is None or str(path).strip() == "":
        raise PromptFileError(
            "custom_prompt_eval needs a system prompt file: "
            "-T system_prompt_file=path/to/prompt.md"
        )
    resolved = resolve_prompt_path(path, invocation_dir)
    if not resolved.exists():
        raise PromptFileError(f"System prompt file not found: {resolved}")
    if not resolved.is_file():
        raise PromptFileError(f"System prompt path is not a file: {resolved}")
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise PromptFileError(f"System prompt file is not UTF-8 text: {resolved}") from e
    if not text.strip():
        raise PromptFileError(f"System prompt file is empty: {resolved}")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_samples(dataset_path: Path = DATASET_PATH) -> list[Sample]:
    """The benchmark prompts, minus items flagged excluded_from_analysis."""
    dataset = json_dataset(
        str(dataset_path),
        sample_fields=FieldSpec(input="input", target="target", id="id", metadata=["metadata"]),
    )
    return [
        s for s in dataset
        if not ((s.metadata or {}).get("metadata") or {}).get("excluded_from_analysis")
    ]


def stratify(samples: list[Sample], per_principle: int | None, seed: int = DEFAULT_SEED) -> list[Sample]:
    """Pick up to `per_principle` samples per principle, deterministically for a seed,
    and interleave the principles so that `--limit N` also stays balanced."""
    by_principle: dict[str, list[Sample]] = {}
    for s in samples:
        by_principle.setdefault(str(s.target), []).append(s)
    rng = random.Random(seed)
    groups = []
    for principle in sorted(by_principle):
        group = sorted(by_principle[principle], key=lambda s: str(s.id))
        if per_principle is not None:
            if per_principle < 1:
                raise ValueError("per_principle must be at least 1")
            group = rng.sample(group, min(per_principle, len(group)))
            group.sort(key=lambda s: str(s.id))
        groups.append(group)
    interleaved = []
    for i in range(max((len(g) for g in groups), default=0)):
        interleaved.extend(g[i] for g in groups if i < len(g))
    return interleaved


@solver
def verbatim_system_message(content: str) -> Solver:
    """Insert `content` as the system message exactly as written.

    Inspect's system_message() treats its argument as a template (str.format over
    sample metadata) and as a possible file path or URL, either of which would
    rewrite a user's prompt.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.insert(0, ChatMessageSystem(content=content))
        return state

    return solve


def resolve_tier(tier: str | None, per_principle) -> tuple[str, list[str], int | None]:
    """(tier, judges, per_principle). An explicit per_principle overrides the tier's."""
    if tier is None:
        _notice_default_tier()
    name = str(tier if tier is not None else DEFAULT_TIER).strip().lower()
    if name not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}, got {tier!r}")
    n = TIERS[name]["per_principle"] if per_principle is None else _int_or_none(per_principle)
    return name, list(TIERS[name]["judges"]), n


def _notice_default_tier() -> None:
    # Once per run: the command builds both tasks, and one line is enough.
    global _default_notice_shown
    if _default_notice_shown:
        return
    _default_notice_shown = True
    t = TIERS[DEFAULT_TIER]
    print(
        f'[humanebench v4] No -T tier given: the default is now "{DEFAULT_TIER}" (single judge, '
        f'{t["per_principle"]} per principle). Pass -T tier=full for the three-judge ensemble.',
        file=sys.stderr,
    )


def _build(condition: str, system_prompt: str | None, prompt_sha: str | None,
           tier: str, judges: list[str], per_principle: int | None, seed: int) -> Task:
    samples = stratify(load_samples(), per_principle, seed)
    judge_prompt_sha = rubric_sha256(load_judge_prompt(JUDGE_PROMPT_PATH))
    _print_estimate(condition, tier, len(samples), len(judges))

    solvers = [generate()]
    if system_prompt is not None:
        solvers.insert(0, verbatim_system_message(system_prompt))

    return Task(
        dataset=MemoryDataset(samples, name=f"humane_bench_{condition}"),
        solver=solvers,
        scorer=overseer_v4(model=judges, temperature=0.0, score_attempts=3),
        metadata={
            "rubric_version": RUBRIC_VERSION,
            "judge_prompt_sha256": judge_prompt_sha,
            "condition": condition,
            # The prompt's hash, never its text: this pairs the two logs without
            # putting the prompt into anything summarised from them.
            "system_prompt_sha256": prompt_sha,
            "tier": tier,
            "per_principle": per_principle,
            "seed": seed,
            "judges": judges,
        },
    )


def _print_estimate(condition: str, tier: str, n_samples: int, n_judges: int) -> None:
    print(
        f"[humanebench v4] {condition}, tier={tier}: {n_samples} samples -> "
        f"{n_samples} target-model calls + {n_samples * n_judges} judge calls "
        f"(before --limit; judge retries add more on parse failures)",
        file=sys.stderr,
    )


@task
def custom_prompt_eval(
    system_prompt_file: str | None = None,
    tier: str | None = None,
    per_principle: int | None = None,
    seed: int = DEFAULT_SEED,
):
    """Your system prompt, scored under rubric v4."""
    tier, judges, per_principle = resolve_tier(tier, per_principle)  # before the prompt file is read
    text, sha = load_system_prompt(system_prompt_file)
    return _build("custom_prompt", text, sha, tier, judges, per_principle, int(seed))


@task
def baseline_v4_eval(
    system_prompt_file: str | None = None,
    tier: str | None = None,
    per_principle: int | None = None,
    seed: int = DEFAULT_SEED,
):
    """No system prompt, scored under rubric v4.

    Accepts system_prompt_file only so the one command that runs both tasks can
    pass it. The prompt is never sent; if given, its hash is recorded so the
    comparison can confirm which prompt this baseline was run alongside.
    """
    tier, judges, per_principle = resolve_tier(tier, per_principle)
    sha = load_system_prompt(system_prompt_file)[1] if system_prompt_file else None
    return _build("baseline", None, sha, tier, judges, per_principle, int(seed))


def _int_or_none(v) -> int | None:
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "all")):
        return None
    return int(v)

