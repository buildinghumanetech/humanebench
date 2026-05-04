#!/usr/bin/env python3
"""Extract per-sample VP-tagged scores from .eval files into a flat CSV.

Walks logs/{baseline,good_persona,bad_persona}/<model>/*.eval and writes
tables/vp_sample_scores.csv with one row per sample per model per persona.

VP labels are sourced from the live `data/humane_bench.jsonl` (joined on
sample_id), not from the `.eval` metadata. The .eval files capture metadata
at eval-run time and miss any later JSONL taxonomy fixes.
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

PERSONAS = ["baseline", "good_persona", "bad_persona"]
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "humane_bench.jsonl"


def _load_live_vp_map(dataset_path: Path) -> dict[str, str]:
    """Return {sample_id: vulnerable_population} from the live JSONL."""
    out: dict[str, str] = {}
    with dataset_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            sid = row.get("id")
            if not sid:
                continue
            out[sid] = (row.get("metadata") or {}).get("vulnerable-population", "") or ""
    return out


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
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables" / "vp_sample_scores.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include the 12 confabulation-flagged items (default: dropped via "
             "humanebench.excluded.load_excluded_ids).",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    if args.include_excluded:
        excluded: set[str] = set()
        print("Loaded 0 excluded IDs (--include-excluded set)")
    else:
        excluded = load_excluded_ids()
        print(f"Loaded {len(excluded)} excluded IDs")

    live_vp = _load_live_vp_map(DATASET_PATH)
    print(f"Loaded live VP labels for {len(live_vp)} sample_ids "
          f"from {DATASET_PATH.name}")

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
                meta = ((sample.metadata or {}).get("metadata") or {})
                overseer = (sample.scores or {}).get("overseer")
                score = overseer.value if overseer is not None else None
                if score is None or _is_nan(score):
                    nan_dropped += 1
                    continue
                rows.append({
                    "sample_id": sample_id,
                    "model": model_name,
                    "persona": persona,
                    "principle": meta.get("principle", ""),
                    "vulnerable_population": live_vp.get(
                        sample_id, meta.get("vulnerable-population", "")
                    ),
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
