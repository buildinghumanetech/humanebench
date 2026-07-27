#!/usr/bin/env python3
"""Build the judge-drift canary: 96 of the 2,304 multi-label calls, re-judged.

The discriminant matrix was scored by `openrouter/openai/gpt-5.1`, an *unpinned*
slug. Whatever sits behind it can move between the run that produced
`tables/discriminant/matrix_long.csv` and any later run whose results get pooled
with it. Pooling two conditions judged by two different models, both called
"gpt-5.1", would put a judge change inside a number the paper attributes to the
design.

The canary is the cheap way to find out first. It re-judges a 96-call slice of
the *already published* condition -- same template, same prompts, same responses,
byte-identical bytes -- so any difference between the new scores and the archived
ones is the judge moving and nothing else. `compute_discriminant_canary.py`
compares the two and gates pooling on the result.

The rows are copied out of `data/discriminant/multilabel_<model>.jsonl` verbatim,
not rebuilt from the archives, and the source files are sha-checked against
`data/discriminant/manifest.json` first. Rebuilding would re-derive the response
text and give the canary a way to differ from the published run for a reason
other than the judge -- which is the one thing it must not be able to do.

THE DRAW
--------
4 scenarios for each of the 24 (source model x scored principle) cells: 96 calls,
32 per model, every principle's rubric exercised on every model. Drawn from the
frozen 96 with `default_rng(BOOTSTRAP_SEED)`, so the slice is reproducible from
the seed alone and was not chosen after seeing which cells agreed.

Cells are drawn independently, so a scenario may appear in several of them. That
is deliberate: constraining the draw to a Latin-square-like cover would trade the
per-cell coverage the gate needs for a property it does not use.

This script makes no API call.

Inputs (read-only):
  - data/discriminant/manifest.json
  - data/discriminant/multilabel_<model>.jsonl
  - data/discriminant/expected_prompt_hashes.csv
  - data/decomposition/discriminant_96_ids.txt

Outputs (written to --output-dir, default data/discriminant_canary/):
  - multilabel_<model>.jsonl    32 rows each
  - expected_prompt_hashes.csv  96 rows
  - manifest.json

Run from repo root:
    python scripts/build_discriminant_canary.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import BOOTSTRAP_SEED, PRINCIPLES  # noqa: E402
from humanebench.discriminant import (  # noqa: E402
    CONDITIONS,
    IDS_PATH,
    SOURCE_MODELS,
)
from humanebench.provenance import file_sha256  # noqa: E402

CONDITION = "discriminant_canary"
SOURCE_CONDITION = "discriminant"
N_PER_CELL = 4


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_source_rows(path: Path, expected_sha: str) -> dict[tuple[str, str], str]:
    """``{(scenario_id, scored_principle): raw_line}`` for one source model.

    The raw line is kept rather than the parsed object so the canary dataset is a
    byte copy of the rows the published run scored. A re-serialised row would
    differ in key order or unicode escaping, and the whole point of the canary is
    that the only thing that can have changed is the judge.
    """
    actual = file_sha256(path)
    if actual != expected_sha:
        raise SystemExit(
            f"{_rel(path)} has sha256 {actual[:16]}, manifest says "
            f"{expected_sha[:16]}. The canary must be drawn from the dataset "
            "that was actually judged."
        )
    rows: dict[tuple[str, str], str] = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            meta = json.loads(line)["metadata"]
            rows[(meta["scenario_id"], meta["scored_principle"])] = line
    return rows


def load_source_hashes(
    path: Path, expected_sha: str,
) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    actual = file_sha256(path)
    if actual != expected_sha:
        raise SystemExit(
            f"{_rel(path)} has sha256 {actual[:16]}, manifest says "
            f"{expected_sha[:16]}."
        )
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return {
            (r["scenario_id"], r["source_model"], r["scored_principle"]): r
            for r in reader
        }, reader.fieldnames


def draw_cells(scenario_ids: list[str], models: list[str], seed: int) -> list[dict]:
    """4 scenarios per (source model, scored principle), in a fixed order.

    The loop order is part of the seed's meaning: change it and the same seed
    produces a different slice, so it is model-outer / canonical-principle-inner
    and stays that way.
    """
    rng = np.random.default_rng(seed)
    pool = np.array(scenario_ids)
    cells = []
    for model in models:
        for principle in PRINCIPLES:
            picked = rng.choice(pool, size=N_PER_CELL, replace=False)
            cells.append({
                "source_model": model,
                "scored_principle": principle,
                "scenario_ids": [str(s) for s in picked],
            })
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", type=Path,
                    default=CONDITIONS[SOURCE_CONDITION].data_dir)
    ap.add_argument("--output-dir", type=Path,
                    default=CONDITIONS[CONDITION].data_dir)
    ap.add_argument("--models", nargs="+", default=list(SOURCE_MODELS))
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    source_manifest_path = args.source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    per_model_src = source_manifest["per_model"]

    scenario_ids = sorted(ln.strip() for ln in IDS_PATH.read_text().splitlines()
                          if ln.strip())
    if len(scenario_ids) != source_manifest["n_scenarios"]:
        raise SystemExit(
            f"{_rel(IDS_PATH)} has {len(scenario_ids)} ids but the source "
            f"manifest was built on {source_manifest['n_scenarios']}"
        )
    if len(scenario_ids) < N_PER_CELL:
        raise SystemExit("frame is smaller than one cell")

    hashes, hash_fields = load_source_hashes(
        args.source_dir / "expected_prompt_hashes.csv",
        source_manifest["expected_prompt_hashes_sha256"])

    models = sorted(args.models)
    missing_models = [m for m in models if m not in per_model_src]
    if missing_models:
        raise SystemExit(f"source manifest has no entry for {missing_models}")

    source_rows = {
        m: load_source_rows(args.source_dir / f"multilabel_{m}.jsonl",
                            per_model_src[m]["dataset_file_sha256"])
        for m in models
    }

    cells = draw_cells(scenario_ids, models, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[str]] = {m: [] for m in models}
    selected_hashes: list[dict] = []
    for cell in cells:
        model = cell["source_model"]
        for sid in cell["scenario_ids"]:
            key = (sid, cell["scored_principle"])
            line = source_rows[model].get(key)
            if line is None:
                raise SystemExit(
                    f"{model}: no published row for {sid} scored against "
                    f"{cell['scored_principle']}"
                )
            hash_row = hashes.get((sid, model, cell["scored_principle"]))
            if hash_row is None:
                raise SystemExit(
                    f"{model}: no expected prompt hash for {sid} scored against "
                    f"{cell['scored_principle']}"
                )
            selected[model].append(line)
            selected_hashes.append(hash_row)

    per_model: dict[str, dict] = {}
    for model in models:
        out = args.output_dir / f"multilabel_{model}.jsonl"
        out.write_text("".join(selected[model]))
        per_model[model] = {
            "dataset_file": _rel(out),
            "dataset_file_sha256": file_sha256(out),
            "n_rows": len(selected[model]),
            "source_dataset_file": per_model_src[model]["dataset_file"],
            "source_dataset_file_sha256": per_model_src[model]["dataset_file_sha256"],
            "source_eval": per_model_src[model]["source_eval"],
            "source_eval_sha256": per_model_src[model]["source_eval_sha256"],
        }
        print(f"  {model:22s} {len(selected[model]):3d} rows -> {_rel(out)}")

    hashes_path = args.output_dir / "expected_prompt_hashes.csv"
    with hashes_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hash_fields)
        w.writeheader()
        w.writerows(selected_hashes)

    n_calls = len(selected_hashes)
    manifest = {
        "schema": "humanebench-discriminant-canary/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "condition": CONDITION,
        "design": (
            "judge-drift canary: a slice of the published discriminant condition "
            "re-judged with the same template, so any score difference is the "
            "unpinned gpt-5.1 slug moving rather than the procedure changing"
        ),
        "source_condition": SOURCE_CONDITION,
        "source_manifest_file": _rel(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "n_judge_calls": n_calls,
        "n_cells": len(cells),
        "n_per_cell": N_PER_CELL,
        "n_source_models": len(models),
        "n_principles": len(PRINCIPLES),
        "one_call_per_principle": True,
        "selection_seed": args.seed,
        "selection_rule": (
            "numpy.random.default_rng(seed).choice(sorted frame ids, size=4, "
            "replace=False) per cell, models sorted, principles in canonical "
            "order; cells drawn independently so a scenario may recur"
        ),
        "frame_ids_file": _rel(IDS_PATH),
        "frame_ids_sha256": file_sha256(IDS_PATH),
        "frame_subset_prompt_hash": source_manifest.get("frame_subset_prompt_hash"),
        "source_dataset_sha256": source_manifest.get("source_dataset_sha256"),
        "judge": source_manifest["judge"],
        "expected_prompt_hashes_file": _rel(hashes_path),
        "expected_prompt_hashes_sha256": file_sha256(hashes_path),
        "per_model": per_model,
        "cells": cells,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{n_calls} judge calls will be made "
          f"({len(cells)} cells x {N_PER_CELL} scenarios)")
    print(f"wrote {_rel(hashes_path)} and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
