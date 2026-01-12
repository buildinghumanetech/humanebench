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
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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


def parse_stats_file(stats_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a stats.json file to load the metrics array.

    Returns:
        List of metric objects, or None if parsing fails
    """
    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not parse {stats_path}: {e}")
        return None


def parse_stats_path(stats_path: Path) -> Optional[Tuple[str, str]]:
    """
    Parse HELM stats file path to extract scenario name and model name.

    The path format is:
    .../runs/v1.0.0/{scenario}:{params},model={provider}_{model}/stats.json

    Args:
        stats_path: Path to stats.json file

    Returns:
        Tuple of (scenario_name, raw_model_name) if successful, None otherwise

    Examples:
        >>> path = Path("mmlu_pro:subset=all,model=openai_gpt-4o-2024-08-06/stats.json")
        >>> parse_stats_path(path)
        ('mmlu_pro', 'openai_gpt-4o-2024-08-06')
    """
    # Get parent directory name which contains scenario and model info
    dir_name = stats_path.parent.name

    # Pattern: {scenario}:{params}...model={model_name}
    # We need to extract scenario (before first colon) and model (after model=)
    pattern = r'^([^:]+):.*model=([^,]+)(?:,|$)'
    match = re.search(pattern, dir_name)

    if match:
        scenario_name = match.group(1)
        raw_model_name = match.group(2)
        return (scenario_name, raw_model_name)

    return None


def identify_benchmark(scenario_name: str) -> Optional[str]:
    """
    Map HELM scenario name to benchmark name.

    Args:
        scenario_name: HELM scenario name (e.g., "mmlu_pro", "gpqa")

    Returns:
        Benchmark name or None if not a target benchmark
    """
    SCENARIO_TO_BENCHMARK = {
        "mmlu_pro": "mmlu_pro",
        "gpqa": "gpqa_diamond",
        "ifeval": "ifeval",
        "wildbench": "wildbench",
        "omni_math": "omni_math"
    }

    return SCENARIO_TO_BENCHMARK.get(scenario_name)


def extract_metric_value(stats_data: List[Dict], metric_name: str) -> Optional[float]:
    """
    Extract a specific metric value from HELM stats data array.

    Args:
        stats_data: List of metric objects from stats.json
        metric_name: Name of metric to extract (e.g., "chain_of_thought_correctness")

    Returns:
        Metric mean value if found, None otherwise

    Notes:
        - Only extracts from "test" split (not "train")
        - Skips perturbation variants (robustness, fairness)
    """
    for metric in stats_data:
        try:
            name_obj = metric.get("name", {})

            # Check if this is the metric we're looking for
            if name_obj.get("name") != metric_name:
                continue

            # Only use test split
            if name_obj.get("split") != "test":
                continue

            # Skip perturbation variants
            if name_obj.get("perturbation") is not None:
                continue

            # Extract mean value
            mean_value = metric.get("mean")
            if mean_value is not None:
                return float(mean_value)

        except (KeyError, TypeError, ValueError):
            continue

    return None


def clean_model_name(raw_model_name: str) -> str:
    """
    Clean and format model name for display.

    Args:
        raw_model_name: Raw model name from HELM (e.g., "openai_gpt-4o-2024-08-06")

    Returns:
        Cleaned model name (e.g., "OpenAI GPT-4o (2024-08-06)")

    Examples:
        >>> clean_model_name("openai_gpt-4o-2024-08-06")
        'OpenAI GPT-4o (2024-08-06)'
        >>> clean_model_name("anthropic_claude-3-5-sonnet-20240620")
        'Anthropic Claude 3.5 Sonnet (2024-06-20)'
    """
    # Handle special provider names
    provider_map = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google",
        "meta": "Meta",
        "mistralai": "Mistral AI",
        "deepseek-ai": "DeepSeek AI",
        "qwen": "Qwen",
        "amazon": "Amazon"
    }

    # Split on first underscore to separate provider from model
    parts = raw_model_name.split("_", 1)
    if len(parts) != 2:
        # No underscore, return as-is with capitalization
        return raw_model_name.replace("-", " ").title()

    provider, model_part = parts

    # Get formatted provider name
    provider_formatted = provider_map.get(provider, provider.replace("-", " ").title())

    # Format model part
    # Look for date patterns: YYYY-MM-DD or YYYYMMDD
    date_pattern_dash = r'(\d{4})-(\d{2})-(\d{2})'
    date_pattern_compact = r'(\d{4})(\d{2})(\d{2})$'

    # Try YYYY-MM-DD pattern
    date_match = re.search(date_pattern_dash, model_part)
    if date_match:
        model_name = model_part[:date_match.start()].rstrip("-")
        date_str = f"({date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)})"
        return f"{provider_formatted} {model_name.replace('-', ' ').upper()} {date_str}"

    # Try YYYYMMDD pattern
    date_match = re.search(date_pattern_compact, model_part)
    if date_match:
        model_name = model_part[:date_match.start()].rstrip("-")
        date_str = f"({date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)})"
        return f"{provider_formatted} {model_name.replace('-', ' ').upper()} {date_str}"

    # No date pattern, just format as-is
    return f"{provider_formatted} {model_part.replace('-', ' ').title()}"


def extract_benchmark_scores(stats_files: List[Path]) -> Dict[str, Dict[str, float]]:
    """
    Extract scores for target benchmarks from stats files.

    Returns:
    {
        "model_name": {
            "mmlu_pro": 0.75,
            "gpqa_diamond": 0.65,
            "ifeval": 0.80,
            "wildbench": 0.79,  # Note: Using rescaled 0-1 scale
            "omni_math": 0.55
        }
    }
    """
    # Mapping of benchmarks to their metric names in HELM data
    BENCHMARK_METRICS = {
        "mmlu_pro": "chain_of_thought_correctness",
        "gpqa_diamond": "chain_of_thought_correctness",
        "ifeval": "ifeval_strict_accuracy",
        "wildbench": "wildbench_score_rescaled",  # Using rescaled 0-1 version
        "omni_math": "omni_math_accuracy"
    }

    model_scores = {}
    skipped_non_target = 0
    skipped_parse_error = 0
    skipped_missing_metric = 0

    for stats_path in stats_files:
        # Parse path to get scenario and model
        parsed = parse_stats_path(stats_path)
        if not parsed:
            skipped_parse_error += 1
            continue

        scenario_name, raw_model_name = parsed

        # Check if this is a target benchmark
        benchmark = identify_benchmark(scenario_name)
        if not benchmark:
            skipped_non_target += 1
            continue

        # Load and parse stats file
        stats_data = parse_stats_file(stats_path)
        if not stats_data:
            continue

        # Extract the metric value
        metric_name = BENCHMARK_METRICS[benchmark]
        metric_value = extract_metric_value(stats_data, metric_name)
        if metric_value is None:
            skipped_missing_metric += 1
            continue

        # Clean model name for output
        model_name = clean_model_name(raw_model_name)

        # Initialize model entry if not exists
        if model_name not in model_scores:
            model_scores[model_name] = {}

        # Check for duplicates (shouldn't happen, but handle gracefully)
        if benchmark in model_scores[model_name]:
            print(f"Warning: Duplicate benchmark '{benchmark}' for model '{model_name}'")
            print(f"  Previous: {model_scores[model_name][benchmark]:.4f}")
            print(f"  New: {metric_value:.4f}")
            print(f"  Keeping previous value")
        else:
            model_scores[model_name][benchmark] = metric_value

    # Print extraction summary
    print(f"Extraction summary:")
    print(f"  Models found: {len(model_scores)}")
    print(f"  Skipped (non-target benchmarks): {skipped_non_target}")
    print(f"  Skipped (parse errors): {skipped_parse_error}")
    print(f"  Skipped (missing metrics): {skipped_missing_metric}")

    # Count models by completeness
    complete_models = sum(1 for scores in model_scores.values() if len(scores) == 5)
    print(f"  Models with all 5 benchmarks: {complete_models}/{len(model_scores)}")

    return model_scores


def rescale_wildbench(score: float) -> float:
    """Rescale WildBench score from 1-10 range to 0-1 range."""
    return (score - 1.0) / 9.0


def calculate_mean_scores(model_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate mean score across benchmarks for each model.
    All scores are already on 0-1 scale (including WildBench rescaled).

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

        # All scores already on 0-1 scale, just calculate mean
        mean_score = sum(scores.values()) / len(scores)
        mean_scores[model] = mean_score

    return mean_scores


def save_data(model_scores: Dict[str, Dict[str, float]], mean_scores: Dict[str, float]):
    """Save extracted data to JSON files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw scores
    raw_data = {
        "benchmarks": TARGET_BENCHMARKS,
        "models": model_scores,
        "note": "All scores on 0-1 scale (WildBench uses wildbench_score_rescaled)"
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
                "note": "Mean across 5 benchmarks (all on 0-1 scale)"
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
