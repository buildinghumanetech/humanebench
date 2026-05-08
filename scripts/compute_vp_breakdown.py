#!/usr/bin/env python3
"""Extract per-sample VP-tagged scores from .eval files into a flat CSV.

Walks logs/{baseline,good_persona,bad_persona}/<model>/*.eval and writes
tables/vp_sample_scores.csv with one row per sample per model per persona.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanebench.excluded import load_excluded_ids  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "humane_bench.jsonl"


def _load_jsonl_metadata(jsonl_path: Path) -> dict[str, dict]:
    """Load principle and VP tags from the canonical JSONL dataset."""
    lookup: dict[str, dict] = {}
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            lookup[row["id"]] = {
                "principle": meta.get("principle", ""),
                "vulnerable-population": meta.get("vulnerable-population", ""),
            }
    return lookup

PERSONAS = ["baseline", "good_persona", "bad_persona"]


def _pick_latest_eval(model_dir: Path) -> tuple[Path | None, int]:
    """Return (latest .eval by mtime, count of older files skipped)."""
    eval_files = list(model_dir.glob("*.eval"))
    if not eval_files:
        return None, 0
    latest = max(eval_files, key=lambda p: p.stat().st_mtime)
    return latest, len(eval_files) - 1


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-sample VP-tagged scores from .eval files into a flat CSV."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "logs",
        help="Directory containing persona subdirectories (baseline/good_persona/bad_persona)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Canonical JSONL dataset for VP/principle metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables" / "vp_sample_scores.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    excluded = load_excluded_ids()
    jsonl_meta = _load_jsonl_metadata(dataset_path)
    print(f"Loaded metadata for {len(jsonl_meta)} scenarios from {dataset_path.name}")
    print(f"Loaded {len(excluded)} excluded IDs")

    rows = []
    files_scanned = 0
    rows_excluded = 0
    nan_dropped = 0

    for persona in PERSONAS:
        persona_dir = logs_dir / persona
        if not persona_dir.is_dir():
            print(f"[warn] missing persona dir: {persona_dir}")
            continue
        for model_dir in sorted(persona_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            eval_path, skipped = _pick_latest_eval(model_dir)
            if eval_path is None:
                print(f"[warn] no .eval files in {persona}/{model_name}")
                continue
            files_scanned += 1
            print(f"  {persona}/{model_name}: picked {eval_path.name}"
                  + (f" (skipped {skipped} stale)" if skipped else ""))

            log = read_eval_log(str(eval_path))
            for sample in log.samples or []:
                sample_id = sample.id
                if sample_id in excluded:
                    rows_excluded += 1
                    continue
                canonical = jsonl_meta.get(sample_id, {})
                overseer = (sample.scores or {}).get("overseer")
                score = overseer.value if overseer is not None else None
                if score is None or _is_nan(score):
                    nan_dropped += 1
                    continue
                rows.append({
                    "sample_id": sample_id,
                    "model": model_name,
                    "persona": persona,
                    "principle": canonical.get("principle", ""),
                    "vulnerable_population": canonical.get("vulnerable-population", ""),
                    "score": score,
                })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nFiles scanned: {files_scanned}")
    print(f"Sample rows excluded: {rows_excluded}")
    print(f"Dropped {nan_dropped} NaN/None scores (ensemble judge failures)")
    print(f"Total rows written: {len(df)}")
    print(f"\nVP distribution (unique scenario IDs per VP):")
    vp_counts = (
        df[df["vulnerable_population"] != ""]
        .groupby("vulnerable_population")["sample_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    for vp, n in vp_counts.items():
        print(f"  {vp}: {n} scenarios")
    print(f"  (blank/general population): "
          f"{df[df['vulnerable_population'] == '']['sample_id'].nunique()} scenarios")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
