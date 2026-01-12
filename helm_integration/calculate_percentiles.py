#!/usr/bin/env python3
"""
HELM Capability Percentile Calculator

Calculates capability percentiles for target models based on HELM benchmark data.
Uses scipy.stats.percentileofscore with 'rank' method to handle ties properly.

Usage:
    python calculate_percentiles.py

Inputs:
    - model_mapping.json: Manual mapping of your models to HELM model names
    - data/helm_aggregate_scores.json: HELM aggregate scores from scraper

Outputs:
    - output/model_coverage_report.txt: Which models found/missing
    - output/capability_percentiles.csv: Final percentiles for plotting
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

try:
    from scipy.stats import percentileofscore
except ImportError:
    print("✗ scipy is not installed")
    print("  Install with: pip install scipy")
    sys.exit(1)


# File paths
SCRIPT_DIR = Path(__file__).parent
MODEL_MAPPING_FILE = SCRIPT_DIR / "model_mapping.json"
AGGREGATE_FILE = SCRIPT_DIR / "data" / "helm_aggregate_scores.json"
OUTPUT_DIR = SCRIPT_DIR / "output"
COVERAGE_REPORT_FILE = OUTPUT_DIR / "model_coverage_report.txt"
PERCENTILES_CSV_FILE = OUTPUT_DIR / "capability_percentiles.csv"


def load_model_mapping() -> List[Dict[str, str]]:
    """Load model mapping from JSON file."""
    if not MODEL_MAPPING_FILE.exists():
        print(f"✗ Model mapping file not found: {MODEL_MAPPING_FILE}")
        print("  Please ensure model_mapping.json exists")
        sys.exit(1)

    with open(MODEL_MAPPING_FILE, 'r') as f:
        data = json.load(f)

    return data.get("mappings", [])


def load_helm_scores() -> Dict[str, float]:
    """
    Load HELM aggregate scores from JSON file.

    Returns:
        Dict mapping model_name -> mean_score
    """
    if not AGGREGATE_FILE.exists():
        print(f"✗ HELM aggregate scores file not found: {AGGREGATE_FILE}")
        print("  Please run helm_scraper.py first")
        sys.exit(1)

    with open(AGGREGATE_FILE, 'r') as f:
        data = json.load(f)

    # Convert list of models to dict
    scores = {}
    for model in data.get("models", []):
        model_name = model.get("model_name")
        mean_score = model.get("mean_score")
        if model_name and mean_score is not None:
            scores[model_name] = mean_score

    return scores


def calculate_percentile(model_score: float, all_scores: List[float]) -> float:
    """
    Calculate percentile ranking for a model score.

    Uses scipy.stats.percentileofscore with kind='rank' to handle ties properly.
    Returns percentile on 0-100 scale.

    Args:
        model_score: The score for the target model
        all_scores: List of all HELM model scores

    Returns:
        Percentile (0-100) indicating where model_score ranks
    """
    return percentileofscore(all_scores, model_score, kind='rank')


def find_helm_model_name(
    your_model_name: str,
    helm_models: List[str]
) -> List[str]:
    """
    Find potential HELM model name matches for a given model name.

    Returns list of potential matches for manual review.
    """
    your_lower = your_model_name.lower()
    potential_matches = []

    for helm_model in helm_models:
        helm_lower = helm_model.lower()

        # Check for substring matches
        if your_lower in helm_lower or helm_lower in your_lower:
            potential_matches.append(helm_model)

    return potential_matches


def generate_coverage_report(
    mappings: List[Dict[str, str]],
    helm_scores: Dict[str, float],
    results: List[Dict[str, any]]
) -> str:
    """Generate a coverage report showing which models were found/missing."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("HELM CAPABILITY PERCENTILE CALCULATION")
    report_lines.append("Model Coverage Report")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Summary statistics
    total_models = len(mappings)
    found_models = sum(1 for r in results if r["in_helm"])
    missing_models = total_models - found_models

    report_lines.append("SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Total target models: {total_models}")
    report_lines.append(f"Found in HELM: {found_models} ({found_models/total_models*100:.1f}%)")
    report_lines.append(f"Not found in HELM: {missing_models} ({missing_models/total_models*100:.1f}%)")
    report_lines.append(f"Total HELM models: {len(helm_scores)}")
    report_lines.append("")

    # Found models
    report_lines.append("MODELS FOUND IN HELM")
    report_lines.append("-" * 70)
    found_results = [r for r in results if r["in_helm"]]
    if found_results:
        for result in found_results:
            report_lines.append(f"✓ {result['model_name']}")
            report_lines.append(f"  HELM name: {result['helm_model_name']}")
            report_lines.append(f"  Mean score: {result['helm_aggregate_score']:.4f}")
            report_lines.append(f"  Percentile: {result['capability_percentile']:.1f}")
            report_lines.append("")
    else:
        report_lines.append("  (none)")
        report_lines.append("")

    # Missing models
    report_lines.append("MODELS NOT FOUND IN HELM")
    report_lines.append("-" * 70)
    missing_results = [r for r in results if not r["in_helm"]]
    if missing_results:
        helm_model_names = list(helm_scores.keys())
        for result in missing_results:
            report_lines.append(f"✗ {result['model_name']}")

            # Try to find potential matches
            potential_matches = find_helm_model_name(result['model_name'], helm_model_names)
            if potential_matches:
                report_lines.append("  Potential HELM matches:")
                for match in potential_matches[:3]:  # Show top 3
                    report_lines.append(f"    - {match}")
            else:
                report_lines.append("  No similar HELM model names found")
            report_lines.append("")
    else:
        report_lines.append("  (none)")
        report_lines.append("")

    # Additional HELM models not in your list
    report_lines.append("ADDITIONAL HELM MODELS (Not in your target list)")
    report_lines.append("-" * 70)
    your_helm_names = {m.get("helm_model_name") for m in mappings if m.get("helm_model_name")}
    additional_models = [name for name in helm_scores.keys() if name not in your_helm_names]
    if additional_models:
        report_lines.append(f"Found {len(additional_models)} additional models in HELM:")
        for model in sorted(additional_models)[:10]:  # Show first 10
            report_lines.append(f"  - {model} ({helm_scores[model]:.4f})")
        if len(additional_models) > 10:
            report_lines.append(f"  ... and {len(additional_models) - 10} more")
    else:
        report_lines.append("  (none)")

    report_lines.append("")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


def save_percentiles_csv(results: List[Dict[str, any]]):
    """Save capability percentiles to CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PERCENTILES_CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name",
            "helm_model_name",
            "helm_aggregate_score",
            "capability_percentile",
            "in_helm"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Saved capability percentiles to {PERCENTILES_CSV_FILE}")


def main():
    print("=" * 70)
    print("HELM Capability Percentile Calculator")
    print("=" * 70)
    print()

    # Load model mapping
    print("Loading model mappings...")
    mappings = load_model_mapping()
    print(f"✓ Loaded {len(mappings)} model mappings")

    # Load HELM scores
    print("Loading HELM aggregate scores...")
    helm_scores = load_helm_scores()
    print(f"✓ Loaded scores for {len(helm_scores)} HELM models")
    print()

    # Get all HELM scores for percentile calculation
    all_helm_scores = list(helm_scores.values())

    # Calculate percentiles
    print("Calculating capability percentiles...")
    results = []

    for mapping in mappings:
        your_model_name = mapping.get("your_model_name")
        helm_model_name = mapping.get("helm_model_name", "").strip()

        if not helm_model_name:
            # Model not mapped
            results.append({
                "model_name": your_model_name,
                "helm_model_name": "",
                "helm_aggregate_score": None,
                "capability_percentile": None,
                "in_helm": False
            })
            continue

        # Get HELM score
        if helm_model_name not in helm_scores:
            print(f"  Warning: '{helm_model_name}' not found in HELM scores")
            results.append({
                "model_name": your_model_name,
                "helm_model_name": helm_model_name,
                "helm_aggregate_score": None,
                "capability_percentile": None,
                "in_helm": False
            })
            continue

        model_score = helm_scores[helm_model_name]
        percentile = calculate_percentile(model_score, all_helm_scores)

        results.append({
            "model_name": your_model_name,
            "helm_model_name": helm_model_name,
            "helm_aggregate_score": model_score,
            "capability_percentile": percentile,
            "in_helm": True
        })

        print(f"  ✓ {your_model_name}: {percentile:.1f}th percentile")

    print()

    # Generate coverage report
    print("Generating coverage report...")
    report_text = generate_coverage_report(mappings, helm_scores, results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(COVERAGE_REPORT_FILE, 'w') as f:
        f.write(report_text)
    print(f"✓ Saved coverage report to {COVERAGE_REPORT_FILE}")

    # Save CSV
    save_percentiles_csv(results)

    # Print report to console
    print()
    print(report_text)

    # Summary
    found_count = sum(1 for r in results if r["in_helm"])
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully calculated percentiles for {found_count}/{len(mappings)} models")
    print()
    print("Output files:")
    print(f"  - {COVERAGE_REPORT_FILE}")
    print(f"  - {PERCENTILES_CSV_FILE}")
    print()
    print("Next step: Use capability_percentiles.csv to plot capability vs. humaneness")

    return 0


if __name__ == "__main__":
    sys.exit(main())
