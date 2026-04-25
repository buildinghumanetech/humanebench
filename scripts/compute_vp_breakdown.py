#!/usr/bin/env python3
"""Extract per-sample VP-tagged scores from .eval files into a flat CSV.

Walks logs/{baseline,good_persona,bad_persona}/<model>/*.eval and writes
tables/vp_sample_scores.csv with one row per sample per model per persona.
"""

import json
import zipfile
from pathlib import Path

import pandas as pd

from humanebench.excluded import load_excluded_ids

PERSONAS = ["baseline", "good_persona", "bad_persona"]
LOGS_DIR = Path("logs")
OUTPUT_CSV = Path("tables/vp_sample_scores.csv")


def _iter_eval_samples(eval_path: Path):
    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = [
            f for f in zf.namelist()
            if f.startswith("samples/") and f.endswith(".json")
        ]
        for sf in sample_files:
            with zf.open(sf) as fh:
                yield json.load(fh)


def main():
    excluded = load_excluded_ids()
    print(f"Loaded {len(excluded)} excluded IDs")

    rows = []
    files_scanned = 0
    samples_excluded = 0

    for persona in PERSONAS:
        persona_dir = LOGS_DIR / persona
        if not persona_dir.is_dir():
            print(f"[warn] missing persona dir: {persona_dir}")
            continue
        for model_dir in sorted(persona_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for eval_path in sorted(model_dir.glob("*.eval")):
                files_scanned += 1
                for sample in _iter_eval_samples(eval_path):
                    sample_id = sample.get("id")
                    if sample_id in excluded:
                        samples_excluded += 1
                        continue
                    meta = (sample.get("metadata") or {}).get("metadata") or {}
                    overseer = (sample.get("scores") or {}).get("overseer") or {}
                    rows.append({
                        "sample_id": sample_id,
                        "model": model_name,
                        "persona": persona,
                        "principle": meta.get("principle", ""),
                        "vulnerable_population": meta.get("vulnerable-population", ""),
                        "score": overseer.get("value"),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFiles scanned: {files_scanned}")
    print(f"Samples excluded (cut list): {samples_excluded}")
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
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
