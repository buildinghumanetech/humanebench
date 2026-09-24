"""Export per-model, per-principle scores to the website's model_scores.json shape.

Reads canonical post-exclusion CSVs (baseline_scores.csv, good_persona_scores.csv,
bad_persona_scores.csv) and writes JSON consumed by the humanebench-website
ScoreGrid / ScoreCarousel / ModelDetailPage components. displayName and provider
fields are preserved from the existing JSON so they stay single-sourced there.
"""

# Paper: utility -- no paper-facing output; exports the post-exclusion per-persona score CSVs to the model_scores.json consumed by the web front end.

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PRINCIPLES = [
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
]

PERSONAS = {
    "baseline": "baseline_scores.csv",
    "good_persona": "good_persona_scores.csv",
    "bad_persona": "bad_persona_scores.csv",
}


def round2(x: float) -> float:
    # Round to 2 decimals; round() returns float and json.dumps renders cleanly.
    return round(float(x), 2)


def load_scores(csv_path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            model = row["model"]
            entry = {"HumaneScore": round2(row["overall"])}
            for p in PRINCIPLES:
                entry[p] = round2(row[p])
            out[model] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT.parent / "humanebench-website" / "public" / "data" / "model_scores.json",
        help="Destination JSON path (default: ../humanebench-website/public/data/model_scores.json)",
    )
    ap.add_argument(
        "--csv-dir",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing the per-persona score CSVs (default: repo root)",
    )
    args = ap.parse_args()

    if not args.output.exists():
        raise SystemExit(
            f"Existing JSON not found at {args.output}; cannot read displayName/provider."
        )

    existing = json.loads(args.output.read_text())
    existing_models = existing.get("models", {})

    persona_scores = {
        persona: load_scores(args.csv_dir / fname) for persona, fname in PERSONAS.items()
    }

    # Use the model set present in baseline (canonical) and intersect with the existing JSON
    # so we keep ordering and surface any drift (added/removed models).
    csv_models = set(persona_scores["baseline"].keys())
    json_models = set(existing_models.keys())
    missing_in_csv = json_models - csv_models
    missing_in_json = csv_models - json_models
    if missing_in_csv:
        print(f"WARNING: models in JSON but missing from CSVs: {sorted(missing_in_csv)}")
    if missing_in_json:
        print(f"WARNING: models in CSVs but missing from JSON: {sorted(missing_in_json)}")

    out_models: dict[str, dict] = {}
    for model_id, meta in existing_models.items():
        if model_id not in csv_models:
            # Preserve untouched (won't happen in practice but avoids accidental data loss).
            out_models[model_id] = meta
            continue
        out_models[model_id] = {
            "displayName": meta.get("displayName", model_id),
            "provider": meta.get("provider", ""),
            "scores": {persona: persona_scores[persona][model_id] for persona in PERSONAS},
        }

    args.output.write_text(json.dumps({"models": out_models}, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
