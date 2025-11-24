#!/usr/bin/env python3
"""
Generate static ScoreGrid SVGs from eval_results headers.

This mirrors the layout/color rules in humanebench-website/src/components/ScoreGrid.vue
so the site can embed SVGs without bundling the raw eval data. Each cell includes
data-* attributes to make it easy to reattach hover tooltips client-side.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.sax.saxutils import escape

# Layout constants (must match the Vue component)
MODEL_NAME_WIDTH = 180
CELL_WIDTH = 80
CELL_HEIGHT = 30
CELL_MARGIN = 4
LABEL_HEIGHT = 150

# Principle ordering for column layout and labels (no details needed; website handles tooltips).
PRINCIPLES: List[Dict[str, str]] = [
    {"id": "HumaneScore", "name": "HumaneScore"},
    {"id": "respect-user-attention", "name": "Respect User Attention"},
    {"id": "enable-meaningful-choices", "name": "Enable Meaningful Choices"},
    {"id": "enhance-human-capabilities", "name": "Enhance Human Capabilities"},
    {"id": "protect-dignity-and-safety", "name": "Protect Dignity and Safety"},
    {"id": "foster-healthy-relationships", "name": "Foster Healthy Relationships"},
    {"id": "prioritize-long-term-wellbeing", "name": "Prioritize Long-term Wellbeing"},
    {"id": "be-transparent-and-honest", "name": "Be Transparent and Honest"},
    {"id": "design-for-equity-and-inclusion", "name": "Design for Equity and Inclusion"},
]

@dataclass
class CellPosition:
    x: float
    y: float


def human_model_name(model_key: str, model_map: Dict[str, str]) -> str:
    pretty = model_map.get(model_key)
    if pretty:
        return pretty
    cleaned = model_key.replace("-", " ")
    return cleaned[:1].upper() + cleaned[1:]


def clamp_score(value: float) -> float:
    return max(-1.0, min(1.0, value))


def score_to_color(score: float) -> str:
    """Interpolate the pink→yellow→green gradient used in the Vue component."""
    clamped = clamp_score(score)
    pink = (0xE5, 0x1A, 0x62)
    yellow = (0xCE, 0xD9, 0x26)
    green = (0x40, 0xBF, 0x4F)

    if clamped < 0:
        t = clamped + 1
        base, target = pink, yellow
    else:
        t = clamped
        base, target = yellow, green

    r = round(base[0] + (target[0] - base[0]) * t)
    g = round(base[1] + (target[1] - base[1]) * t)
    b = round(base[2] + (target[2] - base[2]) * t)
    return f"rgb({r}, {g}, {b})"


def load_models(data_dir: Path, dataset: str) -> Dict[str, Dict[str, float]]:
    """Return model -> metric values dict for a dataset by reading zipped .eval logs."""
    models: Dict[str, Dict[str, float]] = {}
    dataset_dir = data_dir / dataset
    if not dataset_dir.exists():
        return models

    for eval_file in sorted(dataset_dir.rglob("*.eval")):
        # Expect logs/<dataset>/<model>/<timestamp>.eval
        relative_parts = eval_file.relative_to(dataset_dir).parts
        if len(relative_parts) < 2:
            continue
        model_key = relative_parts[0]

        try:
            import zipfile

            with zipfile.ZipFile(eval_file) as zf:
                with zf.open("header.json") as header_fp:
                    payload = json.load(header_fp)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Skipping {eval_file}: {exc}")
            continue

        metrics = payload["results"]["scores"][0]["metrics"]
        models[model_key] = {k: v.get("value", 0) for k, v in metrics.items()}

    def sort_key(item: tuple[str, Dict[str, float]]) -> tuple[float, str]:
        model_key, metrics = item
        score = metrics.get("HumaneScore", 0.0)
        # Sort by HumaneScore desc, then model key asc for stability.
        return (-score, model_key)

    return OrderedDict(sorted(models.items(), key=sort_key))


def position_for_cell(row: int, col: int) -> CellPosition:
    x = MODEL_NAME_WIDTH + col * (CELL_WIDTH + CELL_MARGIN)
    y = row * (CELL_HEIGHT + CELL_MARGIN)
    return CellPosition(x=x, y=y)


def build_svg(
    dataset: str,
    models: Dict[str, Dict[str, float]],
    principles: Iterable[Dict[str, str]],
    model_map: Dict[str, str],
) -> str:
    principles_list = list(principles)
    models_list: List[Tuple[str, Dict[str, float]]] = list(models.items())
    models_count = len(models_list)
    principles_count = len(principles_list)

    svg_width = MODEL_NAME_WIDTH + principles_count * (CELL_WIDTH + CELL_MARGIN)
    svg_height = models_count * (CELL_HEIGHT + CELL_MARGIN) + LABEL_HEIGHT

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {svg_width} {svg_height}" '
        f'role="img" aria-label="Score grid for {escape(dataset)}">'
    )
    lines.append("  <style>")
    lines.append(
        '    .text-caption { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; '
        "font-size: 12px; fill: #1f1f1f; }"
    )
    lines.append("    .font-weight-medium { font-weight: 500; }")
    lines.append("    .font-weight-bold { font-weight: 700; }")
    lines.append("  </style>")
    lines.append(f"  <desc>Generated from eval_results/{escape(dataset)}</desc>")

    for row_index, (model_key, metrics) in enumerate(models_list):
        row_y = row_index * (CELL_HEIGHT + CELL_MARGIN)
        label_y = row_y + CELL_HEIGHT / 2
        lines.append(
            f'  <text x="0" y="{label_y}" '
            f'class="font-weight-medium text-caption" dominant-baseline="middle">'
            f"{escape(human_model_name(model_key, model_map))}"
            "</text>"
        )

        for col_index, principle in enumerate(principles_list):
            position = position_for_cell(row_index, col_index)
            score = metrics.get(principle["id"], 0.0)
            score_str = f"{score:.2f}"
            fill = score_to_color(score)

            lines.append(
                f'  <rect x="{position.x}" y="{position.y}" width="{CELL_WIDTH}" '
                f'height="{CELL_HEIGHT}" rx="8" ry="8" fill="{fill}" '
                f'class="score-cell cursor-pointer" '
                f'data-dataset="{escape(dataset)}" '
                f'data-model="{escape(model_key)}" '
                f'data-principle-id="{escape(principle["id"])}" '
                f'data-score="{score_str}" />'
            )

            text_x = position.x + CELL_WIDTH / 2
            text_y = position.y + CELL_HEIGHT / 2
            lines.append(
                f'  <text x="{text_x}" y="{text_y}" '
                f'class="font-weight-bold text-caption" text-anchor="middle" '
                f'dominant-baseline="middle">{score_str}</text>'
            )

    label_base_y = models_count * (CELL_HEIGHT + CELL_MARGIN) + 10
    for index, category in enumerate(principles_list):
        text_x = MODEL_NAME_WIDTH + index * (CELL_WIDTH + CELL_MARGIN) + CELL_WIDTH / 2
        lines.append(
            f'  <text x="{text_x}" y="{label_base_y}" '
            f'text-anchor="end" dominant-baseline="middle" '
            f'transform="rotate(-45, {text_x}, {label_base_y})" '
            f'class="text-caption font-weight-medium">'
            f"{escape(category['name'])}"
            "</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate static ScoreGrid SVGs from eval_results headers.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["bad_persona", "good_persona", "baseline"],
        help="Dataset folders under eval_results to render.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "logs",
        help="Path to the logs directory (contains task folders like bad_persona/good_persona/baseline).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "figures",
        help="Directory to write the SVG files into.",
    )
    parser.add_argument(
        "--prefix",
        default="scoregrid_",
        help="Filename prefix for generated SVGs.",
    )
    parser.add_argument(
        "--model-map",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "figures" / "model_display_names.json",
        help="Path to JSON mapping of model key -> human-friendly name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_map.exists():
        raise SystemExit(f"Model map not found: {args.model_map}")

    with args.model_map.open() as f:
        model_map: Dict[str, str] = json.load(f)

    # Keep a copy of the mapping alongside the figures for downstream consumers (e.g., website)
    model_map_out = args.output_dir / "model_display_names.json"
    model_map_out.write_text(json.dumps(model_map, indent=2))

    for dataset in args.datasets:
        models = load_models(args.data_dir, dataset)
        if not models:
            print(f"[warn] No models found for dataset '{dataset}', skipping.")
            continue
        missing = [m for m in models.keys() if m not in model_map]
        if missing:
            raise SystemExit(f"Missing model display names for: {', '.join(missing)}")

        svg = build_svg(dataset, models, PRINCIPLES, model_map)
        output_path = args.output_dir / f"{args.prefix}{dataset}.svg"
        output_path.write_text(svg)
        print(f"[ok] Wrote {output_path}")


if __name__ == "__main__":
    main()
