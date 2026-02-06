#!/usr/bin/env python3
"""
Generate a composite ScoreGrid SVG by averaging per-model × per-principle
scores across all three personas (baseline, good_persona, bad_persona).

Reuses build_svg() from create_scoregrid_svg.py.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import pandas as pd

from create_scoregrid_svg import PRINCIPLES, build_svg

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPTS_DIR.parent
FIGURES_DIR = REPO_DIR / "figures"
MODEL_MAP_PATH = FIGURES_DIR / "model_display_names.json"

PERSONA_CSV_FILES = [
    REPO_DIR / "baseline_scores.csv",
    REPO_DIR / "good_persona_scores.csv",
    REPO_DIR / "bad_persona_scores.csv",
]

PRINCIPLE_IDS = [p["id"] for p in PRINCIPLES]
# The CSV uses "overall" for HumaneScore
CSV_METRIC_COLS = [p for p in PRINCIPLE_IDS if p != "HumaneScore"] + ["overall"]


def load_and_average() -> Dict[str, Dict[str, float]]:
    """Load 3 persona CSVs and average per model × principle."""
    frames = []
    for csv_name in PERSONA_CSV_FILES:
        df = pd.read_csv(csv_name)
        frames.append(df.set_index("model")[CSV_METRIC_COLS])

    # Only keep models present in all 3 personas
    common_models = frames[0].index
    for f in frames[1:]:
        common_models = common_models.intersection(f.index)

    averaged: Dict[str, Dict[str, float]] = {}
    for model in common_models:
        metrics: Dict[str, float] = {}
        for col in CSV_METRIC_COLS:
            vals = [f.loc[model, col] for f in frames]
            # Skip if any value is NaN
            if any(pd.isna(v) for v in vals):
                metrics[col] = 0.0
            else:
                metrics[col] = sum(vals) / len(vals)

        # Map "overall" back to "HumaneScore" for the SVG builder
        metrics["HumaneScore"] = metrics.pop("overall")
        averaged[model] = metrics

    # Sort by HumaneScore descending
    def sort_key(item: tuple) -> tuple:
        return (-item[1].get("HumaneScore", 0.0), item[0])

    return OrderedDict(sorted(averaged.items(), key=sort_key))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate composite ScoreGrid SVG (average of 3 personas).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to write the SVG file into.",
    )
    parser.add_argument(
        "--model-map",
        type=Path,
        default=MODEL_MAP_PATH,
        help="Path to JSON mapping of model key -> human-friendly name.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_map.exists():
        raise SystemExit(f"Model map not found: {args.model_map}")

    with args.model_map.open() as f:
        model_map: Dict[str, str] = json.load(f)

    models = load_and_average()
    if not models:
        raise SystemExit("No models found after averaging personas.")

    missing = [m for m in models if m not in model_map]
    if missing:
        raise SystemExit(f"Missing model display names for: {', '.join(missing)}")

    svg = build_svg("composite", models, PRINCIPLES, model_map)
    output_path = args.output_dir / "scoregrid_composite.svg"
    output_path.write_text(svg)
    print(f"[ok] Wrote {output_path}")


if __name__ == "__main__":
    main()
