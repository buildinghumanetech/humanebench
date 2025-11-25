#!/usr/bin/env python3
"""
HELM Capabilities Data Scraper

Downloads and parses HELM Capabilities benchmark data from Google Cloud Storage.
Extracts scores for MMLU-Pro, GPQA Diamond, IFEval, WildBench, and Omni-MATH.
Calculates mean scores across benchmarks (with WildBench rescaling).

Usage:
    python helm_scraper.py [--force-download]

Outputs:
    - data/helm_raw_data.json: All models with individual benchmark scores
    - data/helm_aggregate_scores.json: All models with mean scores
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


# Target benchmarks (HELM scenario names may differ)
TARGET_BENCHMARKS = [
    "mmlu_pro",
    "gpqa_diamond",
    "ifeval",
    "wildbench",
    "omni_math"
]

# GCS bucket path for HELM data
GCS_BUCKET = "gs://crfm-helm-public/capabilities/benchmark_output"

# Local paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
GCS_CACHE_DIR = DATA_DIR / "gcs_cache"
RAW_DATA_FILE = DATA_DIR / "helm_raw_data.json"
AGGREGATE_FILE = DATA_DIR / "helm_aggregate_scores.json"


def check_gcloud_installed() -> bool:
    """Check if gcloud CLI is installed and accessible."""
    try:
        result = subprocess.run(
            ["gcloud", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def download_helm_data(force: bool = False) -> bool:
    """
    Download HELM data from GCS using gcloud storage rsync.

    Args:
        force: If True, re-download even if cache exists

    Returns:
        True if download successful, False otherwise
    """
    if GCS_CACHE_DIR.exists() and not force:
        print(f"✓ Using cached HELM data from {GCS_CACHE_DIR}")
        print("  (Use --force-download to re-download)")
        return True

    print(f"Downloading HELM data from {GCS_BUCKET}...")
    print("This may take a few minutes...")

    GCS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "gcloud", "storage", "rsync", "-r",
                GCS_BUCKET,
                str(GCS_CACHE_DIR)
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"✗ Error downloading HELM data:")
            print(result.stderr)
            return False

        print(f"✓ Successfully downloaded HELM data to {GCS_CACHE_DIR}")
        return True

    except subprocess.TimeoutExpired:
        print("✗ Download timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False


def find_stats_files() -> List[Path]:
    """Find all stats.json files in the GCS cache."""
    stats_files = list(GCS_CACHE_DIR.rglob("stats.json"))
    print(f"Found {len(stats_files)} stats.json files")
    return stats_files


def parse_stats_file(stats_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a stats.json file to extract benchmark scores.

    Returns dict with structure:
    {
        "model_name": str,
        "scenario": str,
        "metrics": {...}
    }
    """
    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not parse {stats_path}: {e}")
        return None


def extract_benchmark_scores(stats_files: List[Path]) -> Dict[str, Dict[str, float]]:
    """
    Extract scores for target benchmarks from stats files.

    Returns:
    {
        "model_name": {
            "mmlu_pro": 0.75,
            "gpqa_diamond": 0.65,
            "ifeval": 0.80,
            "wildbench": 7.5,  # Note: 1-10 scale
            "omni_math": 0.55
        }
    }
    """
    model_scores = {}

    for stats_path in stats_files:
        data = parse_stats_file(stats_path)
        if not data:
            continue

        # Extract model name and scenario from path or data
        # The structure may vary, so we'll need to inspect the actual data
        # For now, this is a placeholder that will need adjustment based on actual HELM data structure

        # TODO: Adjust based on actual HELM data structure
        # This will need to be updated once we see the actual JSON structure
        pass

    return model_scores


def rescale_wildbench(score: float) -> float:
    """Rescale WildBench score from 1-10 range to 0-1 range."""
    return (score - 1.0) / 9.0


def calculate_mean_scores(model_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate mean score across benchmarks for each model.
    Rescales WildBench before averaging.

    Returns:
    {
        "model_name": mean_score
    }
    """
    mean_scores = {}

    for model, scores in model_scores.items():
        if len(scores) != 5:
            print(f"Warning: {model} missing some benchmark scores ({len(scores)}/5)")
            continue

        # Rescale WildBench
        rescaled_scores = scores.copy()
        if "wildbench" in rescaled_scores:
            rescaled_scores["wildbench"] = rescale_wildbench(rescaled_scores["wildbench"])

        # Calculate mean
        mean_score = sum(rescaled_scores.values()) / len(rescaled_scores)
        mean_scores[model] = mean_score

    return mean_scores


def save_data(model_scores: Dict[str, Dict[str, float]], mean_scores: Dict[str, float]):
    """Save extracted data to JSON files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw scores
    raw_data = {
        "benchmarks": TARGET_BENCHMARKS,
        "models": model_scores,
        "note": "WildBench scores are on 1-10 scale (not yet rescaled)"
    }

    with open(RAW_DATA_FILE, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"✓ Saved raw data to {RAW_DATA_FILE}")

    # Save aggregate scores
    aggregate_data = {
        "models": [
            {
                "model_name": model,
                "mean_score": score,
                "note": "Mean across 5 benchmarks with WildBench rescaled to 0-1"
            }
            for model, score in sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        ]
    }

    with open(AGGREGATE_FILE, 'w') as f:
        json.dump(aggregate_data, f, indent=2)
    print(f"✓ Saved aggregate scores to {AGGREGATE_FILE}")

    print(f"\n✓ Extracted data for {len(model_scores)} models")
    print(f"  Top model: {max(mean_scores, key=mean_scores.get)} ({max(mean_scores.values()):.3f})")


def main():
    parser = argparse.ArgumentParser(description="Download and parse HELM Capabilities data")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if cached data exists"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HELM Capabilities Data Scraper")
    print("=" * 60)

    # Check gcloud CLI
    if not check_gcloud_installed():
        print("\n✗ gcloud CLI is not installed or not in PATH")
        print("\nTo install gcloud CLI:")
        print("  https://cloud.google.com/sdk/docs/install")
        print("\nAlternatively, you can manually download HELM data from:")
        print(f"  {GCS_BUCKET}")
        return 1

    print("✓ gcloud CLI is installed\n")

    # Download data
    if not download_helm_data(force=args.force_download):
        return 1

    print("\n" + "=" * 60)
    print("Parsing HELM data...")
    print("=" * 60 + "\n")

    # Find stats files
    stats_files = find_stats_files()
    if not stats_files:
        print("✗ No stats.json files found in cache")
        print(f"  Expected location: {GCS_CACHE_DIR}")
        return 1

    # Extract scores
    print("Extracting benchmark scores...")
    model_scores = extract_benchmark_scores(stats_files)

    if not model_scores:
        print("✗ No model scores extracted")
        print("  The HELM data structure may have changed.")
        print("  Manual inspection of GCS cache may be required.")
        return 1

    # Calculate mean scores
    print("Calculating mean scores...")
    mean_scores = calculate_mean_scores(model_scores)

    # Save data
    save_data(model_scores, mean_scores)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Inspect data/helm_raw_data.json to see all HELM model names")
    print("2. Edit model_mapping.json to map your models to HELM models")
    print("3. Run calculate_percentiles.py to generate capability percentiles")

    return 0


if __name__ == "__main__":
    sys.exit(main())
