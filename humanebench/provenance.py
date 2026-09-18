"""Shared provenance constants and helpers for HumaneBench.

The reported HumaneBench numbers come from Inspect ``.eval`` runs whose logs each
embed the exact ``(id, input, target)`` triples they scored. This module is the
single source of truth for the *content-binding* provenance argument: it pins the
prompt-content freeze commit, the canonical prompt hash, and the helpers that both
``scripts/build_provenance.py`` and ``scripts/verify_provenance.py`` use, so the
two cannot drift. See ``PROVENANCE.md`` for the narrative.

Why content hashes and not timestamps: git does not preserve file mtimes and
local/commit dates are forgeable, so the load-bearing proof is that every run
scored prompts byte-identical to the frozen dataset. If the scored prompts *are*
the frozen set, "the prompts were tuned afterward" is impossible for the reported
numbers. Timestamps and git revisions are corroboration only.
"""
# Paper: implements the artifact-provenance argument -- the frozen prompt-set
# hash, the per-run content check against it, and the run discovery both the
# manifest builder and the verifier share (supplement, "Scenario Construction
# Pipeline", artifact-provenance paragraph).
# Paper: JUDGE_ENSEMBLE below is the historical record of the ensemble the
# reported runs were judged by (main paper, "Judge Validation").
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "humane_bench.jsonl"
DATASET_REPO_REL = "data/humane_bench.jsonl"
LOGS_DIR = REPO_ROOT / "logs"
PERSONAS = ["baseline", "good_persona", "bad_persona"]

# --- Anchors (see PROVENANCE.md) --------------------------------------------
# Earliest commit whose canonical prompt hash equals what every reported run
# scored and what data/humane_bench.jsonl still contains today. (The Nov 7-8
# "final dataset" commits have a *different* prompt hash -- text was still being
# edited until Nov 16 -- which is exactly why we anchor on a hash, not a date.)
FREEZE_COMMIT = "9dc15bd362fc7ac15f6da695e9dd4292e92fcdbc"
FREEZE_COMMIT_DATE = "2025-11-16T23:28:33-05:00"
FROZEN_PROMPT_HASH = "e1af241db346e0299bb96c893b43746bff7a70d7a50efdd105ba9ded3cfe5d5f"

# Dataset commits AFTER the freeze. Each MUST preserve FROZEN_PROMPT_HASH, i.e.
# it touched only metadata / exclusion flags, never input or target. Verified by
# build/verify; if any ever fails to preserve the hash, that is a real prompt
# edit and the freeze claim is broken.
POST_FREEZE_DATASET_COMMITS = [
    ("b754d3441079cd57de6cd8c8182d8b670be6449a", "2026-03-15 Fix 4 data quality issues"),
    ("d119c0f398db364df131bbcd12497ad6814ec805", "2026-03-23 Rename career-guidance -> workplace, fix VP tag"),
    ("79cdbcb3c415112bd491bfbe29efdc220575d0ed", "2026-03-23 Rename domain workplace -> career"),
    ("ef43b81ea32dd21d26637fc78a89ac8eafc26bca", "2026-04-19 Tag 12 excluded items; make metadata canonical"),
]

# Fixed judge ensemble for the reported runs (see src/baseline_task.py).
JUDGE_ENSEMBLE = {
    "models": [
        "openrouter/anthropic/claude-4.5-sonnet",
        "openrouter/openai/gpt-5.1",
        "openrouter/google/gemini-2.5-pro",
    ],
    "temperature": 0.0,
    "score_attempts": 3,
}

# Zenodo DOI for the archived logs deposit (reserved on a draft deposition).
ZENODO_DOI = "10.5281/zenodo.21046964"

# Field/record separators for canonical hashing (control chars that cannot occur
# in the prompt text), so concatenation is unambiguous.
_FIELD_SEP = "\x1f"
_REC_SEP = "\x1e"


# Paper: the content-binding hash. Every reported run's scored prompts hash to
# FROZEN_PROMPT_HASH above, which is what the reproducibility claim rests on:
# the scenarios could not have been tuned after the results were seen
# (supplement, "Scenario Construction Pipeline").
def canonical_prompt_hash(pairs: Iterable[tuple[str, str, str]]) -> tuple[str, int]:
    """SHA-256 over the sorted ``(id, input, target)`` triples.

    Metadata is intentionally excluded: this captures exactly what a model is
    shown and scored on, nothing else. Sorting by id makes the hash independent
    of dataset row order and of sample-file order inside a ``.eval`` zip.
    Returns ``(hexdigest, n_records)``.
    """
    items = sorted((str(i), str(inp), str(tgt)) for i, inp, tgt in pairs)
    h = hashlib.sha256()
    for _id, inp, tgt in items:
        h.update(f"{_id}{_FIELD_SEP}{inp}{_FIELD_SEP}{tgt}{_REC_SEP}".encode("utf-8"))
    return h.hexdigest(), len(items)


def prompt_hash_from_jsonl(text: str) -> tuple[str, int]:
    """Canonical prompt hash of a humane_bench.jsonl document (string contents)."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        pairs.append((row["id"], row["input"], row.get("target")))
    return canonical_prompt_hash(pairs)


def iter_eval_samples(eval_path: Path) -> Iterator[dict]:
    """Yield each sample dict from an Inspect ``.eval`` file (a ZIP archive)."""
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                with zf.open(name) as fh:
                    yield json.load(fh)


def read_eval_header(eval_path: Path) -> dict:
    """Return the parsed ``header.json`` of an Inspect ``.eval`` file."""
    with zipfile.ZipFile(eval_path, "r") as zf:
        with zf.open("header.json") as fh:
            return json.load(fh)


def eval_prompt_hash(eval_path: Path) -> tuple[str, int]:
    """Canonical prompt hash over the prompts a run actually scored."""
    pairs = [
        (s["id"], s["input"], s.get("target"))
        for s in iter_eval_samples(eval_path)
    ]
    return canonical_prompt_hash(pairs)


def file_sha256(path: Path, _bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_bufsize):
            h.update(chunk)
    return h.hexdigest()


# Paper: defines the reported run set -- one eval log per (model, persona) over
# the three conditions of the main paper, "Three-Condition Evaluation Design".
# Every published aggregate is conditioned on exactly this set.
def reported_runs(logs_dir: Path = LOGS_DIR) -> list[tuple[str, str, Path]]:
    """Return sorted ``(persona, model, path)`` for the canonical reported runs.

    The reported set is exactly one ``.eval`` per (model, persona) under
    ``logs/{baseline,good_persona,bad_persona}/<model>/``. ``logs/old_gold`` and
    ``logs/golden_questions_eval`` are deliberately excluded -- the former is
    superseded, the latter is the human-agreement validation run on a different
    dataset (documented separately in PROVENANCE.md).
    """
    out: list[tuple[str, str, Path]] = []
    for persona in PERSONAS:
        pdir = logs_dir / persona
        if not pdir.is_dir():
            continue
        for evalf in sorted(pdir.rglob("*.eval")):
            out.append((persona, evalf.parent.name, evalf))
    return out


def decomposition_runs(logs_dir: Path = LOGS_DIR) -> list[tuple[str, str, Path]]:
    """Return sorted ``(condition, model, path)`` for the decomposition runs.

    Separate from :func:`reported_runs` on purpose. The decomposition conditions
    are a robustness analysis of the adversarial condition, not part of the
    reported three-condition design, and every published aggregate -- the item
    counts, Krippendorff's alpha, the design effects -- is conditioned on those
    three. Keeping the two run sets in separate functions means no caller can
    silently widen a published number by walking one extra directory.

    Raises if a model directory holds more than one ``.eval``: downstream
    discovery assumes exactly one per cell, and a stray retry artifact would
    otherwise be picked up by lexicographic luck. ``attic/`` subdirectories hold
    deliberately superseded files and are ignored.
    """
    from humanebench.decomposition import TASK_TYPES as _DECOMP_TASK_TYPES

    out: list[tuple[str, str, Path]] = []
    for condition in _DECOMP_TASK_TYPES:
        cdir = logs_dir / condition
        if not cdir.is_dir():
            continue
        for model_dir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            evals = sorted(model_dir.glob("*.eval"))
            if not evals:
                continue
            if len(evals) > 1:
                names = ", ".join(p.name for p in evals)
                raise RuntimeError(
                    f"{condition}/{model_dir.name}: expected one .eval, found "
                    f"{len(evals)} ({names}). Move superseded runs to an attic/ "
                    "subdirectory before building provenance."
                )
            out.append((condition, model_dir.name, evals[0]))
    return out


def frozen_triples(dataset_path: Path = DATASET_PATH) -> dict[str, tuple[str, str]]:
    """``id -> (input, target)`` for the frozen dataset.

    Used to prove that a subset run scored prompts byte-identical to the frozen
    set. A subset can never reproduce ``FROZEN_PROMPT_HASH`` -- the hash covers
    the whole set -- so the equivalent guarantee is per-triple identity plus a
    subset-membership test.
    """
    triples: dict[str, tuple[str, str]] = {}
    with dataset_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            triples[row["id"]] = (row["input"], row.get("target"))
    return triples


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def created_after_freeze(created_iso: str) -> bool:
    return parse_iso(created_iso) > parse_iso(FREEZE_COMMIT_DATE)


# --- git helpers (best-effort; provenance must not crash off-repo) ----------
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
    )


def git_available() -> bool:
    return _git("rev-parse", "--git-dir").returncode == 0


def git_commit_present(sha: str) -> bool:
    """True if ``sha`` (full or abbreviated) resolves to a commit in this repo."""
    return _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def git_file_at_commit(sha: str, repo_rel_path: str = DATASET_REPO_REL) -> str | None:
    res = _git("cat-file", "-p", f"{sha}:{repo_rel_path}")
    return res.stdout if res.returncode == 0 else None
