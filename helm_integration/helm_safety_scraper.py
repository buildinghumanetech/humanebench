#!/usr/bin/env python3
"""
HELM Safety Data Scraper

Downloads and parses HELM Safety benchmark data from Google Cloud Storage.
Mirrors helm_scraper.py (capabilities). Extracts scores for five sub-scenarios
(BBQ, SimpleSafetyTests, HarmBench, XSTest, AnthropicRedTeam) and computes the
mean across them. All five are on a 0-1 scale with higher = safer.

Run --discover first to enumerate the (scenario, metric) pairs actually present
in the bucket; that report tells you what to put in TARGET_SCENARIOS and
SCENARIO_METRICS if HELM has renamed anything.

Usage:
    python helm_safety_scraper.py [--force-download] [--discover]

Outputs:
    - data/helm_safety_raw_data.json: per-scenario scores per model
    - data/helm_safety_aggregate_scores.json: per-model mean across scenarios
    - output/safety_coverage_report.txt: matched / partial / unmapped models
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Target HELM Safety scenarios. Keys are HELM scenario names (left of ':' in
# the run directory). Values are best-guess primary-metric names; --discover
# will surface the actual names if these are wrong.
TARGET_SCENARIOS = [
    "bbq",
    "simple_safety_tests",
    "harm_bench",
    "xstest",
    "anthropic_red_team",
]

# Map scenario -> ordered list of candidate metric names. The first one present
# in the stats.json wins. Higher = safer for all five.
SCENARIO_METRICS: Dict[str, List[str]] = {
    "bbq":                 ["bbq_metric_ambiguous_bias",
                            "bbq_accuracy",
                            "exact_match",
                            "quasi_exact_match"],
    "simple_safety_tests": ["safety_score",
                            "safe_response_rate",
                            "exact_match"],
    "harm_bench":          ["safety_score",
                            "refusal_rate",
                            "safe_response_rate"],
    "xstest":              ["safety_score",
                            "exact_match",
                            "refusal_rate"],
    "anthropic_red_team":  ["safety_score",
                            "safe_response_rate"],
}

GCS_BUCKET = "gs://crfm-helm-public/safety/benchmark_output"

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
GCS_CACHE_DIR = DATA_DIR / "gcs_safety_cache"
RAW_DATA_FILE = DATA_DIR / "helm_safety_raw_data.json"
AGGREGATE_FILE = DATA_DIR / "helm_safety_aggregate_scores.json"
OUTPUT_DIR = SCRIPT_DIR / "output"
COVERAGE_FILE = OUTPUT_DIR / "safety_coverage_report.txt"


def check_gcloud_installed() -> bool:
    try:
        result = subprocess.run(
            ["gcloud", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def download_helm_data(force: bool = False) -> bool:
    if GCS_CACHE_DIR.exists() and not force:
        print(f"✓ Using cached HELM Safety data from {GCS_CACHE_DIR}")
        print("  (Use --force-download to re-download)")
        return True

    print(f"Downloading HELM Safety data from {GCS_BUCKET}...")
    GCS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["gcloud", "storage", "rsync", "-r", GCS_BUCKET, str(GCS_CACHE_DIR)],
            capture_output=True, text=True, timeout=900,
        )
        if result.returncode != 0:
            print("✗ Error downloading HELM Safety data:")
            print(result.stderr)
            return False
        print(f"✓ Downloaded HELM Safety data to {GCS_CACHE_DIR}")
        return True
    except subprocess.TimeoutExpired:
        print("✗ Download timed out after 15 minutes")
        return False
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False


def find_stats_files() -> List[Path]:
    stats_files = list(GCS_CACHE_DIR.rglob("stats.json"))
    print(f"Found {len(stats_files)} stats.json files")
    return stats_files


def parse_stats_file(stats_path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(stats_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not parse {stats_path}: {e}")
        return None


def parse_stats_path(stats_path: Path) -> Optional[Tuple[str, str]]:
    """Return (scenario_name, raw_model_name) or None."""
    dir_name = stats_path.parent.name
    pattern = r'^([^:]+):.*model=([^,]+)(?:,|$)'
    match = re.search(pattern, dir_name)
    if not match:
        return None
    return match.group(1), match.group(2)


def extract_first_available_metric(
    stats_data: List[Dict[str, Any]], candidate_names: List[str]
) -> Tuple[Optional[float], Optional[str]]:
    """
    Walk the candidate list and return the first metric found on the test split
    with no perturbation. Returns (mean_value, resolved_metric_name).
    """
    available_by_name: Dict[str, float] = {}
    for metric in stats_data:
        try:
            name_obj = metric.get("name", {})
            if name_obj.get("split") != "test":
                continue
            if name_obj.get("perturbation") is not None:
                continue
            mname = name_obj.get("name")
            mean_value = metric.get("mean")
            if mname is None or mean_value is None:
                continue
            available_by_name.setdefault(mname, float(mean_value))
        except (KeyError, TypeError, ValueError):
            continue

    for candidate in candidate_names:
        if candidate in available_by_name:
            return available_by_name[candidate], candidate
    return None, None


def clean_model_name(raw: str) -> str:
    """Format provider_model into a display name (mirrors helm_scraper.py)."""
    provider_map = {
        "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
        "meta": "Meta", "mistralai": "Mistral AI", "deepseek-ai": "DeepSeek AI",
        "qwen": "Qwen", "amazon": "Amazon",
    }
    parts = raw.split("_", 1)
    if len(parts) != 2:
        return raw.replace("-", " ").title()
    provider, model_part = parts
    provider_formatted = provider_map.get(provider, provider.replace("-", " ").title())

    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', model_part)
    if date_match:
        model_name = model_part[:date_match.start()].rstrip("-")
        date_str = f"({date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)})"
        return f"{provider_formatted} {model_name.replace('-', ' ').upper()} {date_str}"

    date_match = re.search(r'(\d{4})(\d{2})(\d{2})$', model_part)
    if date_match:
        model_name = model_part[:date_match.start()].rstrip("-")
        date_str = f"({date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)})"
        return f"{provider_formatted} {model_name.replace('-', ' ').upper()} {date_str}"

    return f"{provider_formatted} {model_part.replace('-', ' ').title()}"


def discover_report(stats_files: List[Path]) -> None:
    """Print every (scenario, metric_name) pair found in the cache."""
    print("\n--- discovery report ---")
    pairs: Dict[str, set] = defaultdict(set)
    scenario_models: Dict[str, set] = defaultdict(set)
    for path in stats_files:
        parsed = parse_stats_path(path)
        if not parsed:
            continue
        scenario, raw_model = parsed
        scenario_models[scenario].add(raw_model)
        stats = parse_stats_file(path)
        if not stats:
            continue
        for metric in stats:
            try:
                name_obj = metric.get("name", {})
                if name_obj.get("split") != "test":
                    continue
                if name_obj.get("perturbation") is not None:
                    continue
                mname = name_obj.get("name")
                if mname:
                    pairs[scenario].add(mname)
            except (KeyError, TypeError):
                continue
    for scenario in sorted(pairs):
        print(f"\nscenario: {scenario}  ({len(scenario_models[scenario])} models)")
        for mname in sorted(pairs[scenario]):
            print(f"  - {mname}")


def extract_safety_scores(
    stats_files: List[Path],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Returns (model_scores, resolved_metrics):
      model_scores  -> { display_model_name: { scenario: score } }
      resolved_metrics -> { scenario: metric_name_used }
    """
    model_scores: Dict[str, Dict[str, float]] = {}
    resolved_metrics: Dict[str, str] = {}
    skipped_non_target = 0
    skipped_parse = 0
    skipped_no_metric = 0

    for path in stats_files:
        parsed = parse_stats_path(path)
        if not parsed:
            skipped_parse += 1
            continue
        scenario, raw_model = parsed

        if scenario not in SCENARIO_METRICS:
            skipped_non_target += 1
            continue

        stats = parse_stats_file(path)
        if not stats:
            continue

        value, used_metric = extract_first_available_metric(
            stats, SCENARIO_METRICS[scenario]
        )
        if value is None:
            skipped_no_metric += 1
            continue

        # Lock in the metric the first time we resolve it; warn if it changes.
        if scenario not in resolved_metrics:
            resolved_metrics[scenario] = used_metric
        elif resolved_metrics[scenario] != used_metric:
            print(
                f"WARNING: scenario {scenario} resolved to "
                f"{resolved_metrics[scenario]} earlier but {used_metric} now; "
                f"keeping the earlier metric."
            )

        if resolved_metrics[scenario] != used_metric:
            # Re-extract using the locked metric only.
            value, _ = extract_first_available_metric(
                stats, [resolved_metrics[scenario]]
            )
            if value is None:
                skipped_no_metric += 1
                continue

        display = clean_model_name(raw_model)
        model_scores.setdefault(display, {})
        if scenario in model_scores[display]:
            print(
                f"Warning: duplicate scenario {scenario!r} for {display!r}; "
                f"keeping previous ({model_scores[display][scenario]:.4f}), "
                f"new was {value:.4f}"
            )
        else:
            model_scores[display][scenario] = value

    print(f"Extraction summary:")
    print(f"  Models encountered: {len(model_scores)}")
    print(f"  Skipped (non-target scenarios): {skipped_non_target}")
    print(f"  Skipped (parse errors): {skipped_parse}")
    print(f"  Skipped (no metric matched): {skipped_no_metric}")
    print(f"  Resolved metrics: {resolved_metrics}")
    return model_scores, resolved_metrics


def calculate_mean_scores(
    model_scores: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    mean_scores: Dict[str, float] = {}
    n_required = len(TARGET_SCENARIOS)
    partial = []
    for model, scores in model_scores.items():
        if len(scores) != n_required:
            partial.append((model, len(scores)))
            continue
        mean_scores[model] = sum(scores.values()) / n_required
    if partial:
        print("Models dropped for incomplete coverage:")
        for m, k in sorted(partial):
            print(f"  - {m} ({k}/{n_required})")
    print(f"  Models with all {n_required} scenarios: "
          f"{len(mean_scores)}/{len(model_scores)}")
    return mean_scores


def detect_duplicate_vectors(model_scores: Dict[str, Dict[str, float]]) -> List[Tuple[str, str]]:
    """Flag pairs of models with byte-identical score vectors across scenarios."""
    by_vector: Dict[Tuple[Tuple[str, float], ...], List[str]] = defaultdict(list)
    for model, scores in model_scores.items():
        if len(scores) != len(TARGET_SCENARIOS):
            continue
        key = tuple(sorted(scores.items()))
        by_vector[key].append(model)
    dupes: List[Tuple[str, str]] = []
    for models in by_vector.values():
        if len(models) > 1:
            print("WARNING: byte-identical score vectors:")
            for m in models:
                print(f"  - {m}")
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    dupes.append((models[i], models[j]))
    return dupes


def save_data(
    model_scores: Dict[str, Dict[str, float]],
    mean_scores: Dict[str, float],
    resolved_metrics: Dict[str, str],
    duplicate_pairs: List[Tuple[str, str]],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = {
        "scenarios": TARGET_SCENARIOS,
        "resolved_metrics": resolved_metrics,
        "models": model_scores,
        "note": "All scores on 0-1 scale, higher = safer (HELM convention).",
    }
    with open(RAW_DATA_FILE, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"✓ Saved raw safety data to {RAW_DATA_FILE}")

    aggregate = {
        "models": [
            {"model_name": m, "mean_score": s,
             "note": f"Mean across {len(TARGET_SCENARIOS)} HELM Safety scenarios"}
            for m, s in sorted(mean_scores.items(), key=lambda kv: kv[1], reverse=True)
        ]
    }
    with open(AGGREGATE_FILE, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"✓ Saved aggregate safety scores to {AGGREGATE_FILE}")

    lines = [
        "HELM Safety coverage report",
        "=" * 40,
        f"Scenarios: {TARGET_SCENARIOS}",
        f"Resolved metrics: {resolved_metrics}",
        f"Models with all {len(TARGET_SCENARIOS)} scenarios: {len(mean_scores)}",
        "",
        "Aggregate scores (sorted desc):",
    ]
    for m, s in sorted(mean_scores.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {s:.3f}  {m}")
    lines.append("")
    lines.append("Partial-coverage models (dropped from aggregate):")
    for m, scores in sorted(model_scores.items()):
        if len(scores) != len(TARGET_SCENARIOS):
            missing = sorted(set(TARGET_SCENARIOS) - set(scores))
            lines.append(f"  {m}  ({len(scores)}/{len(TARGET_SCENARIOS)}; missing {missing})")
    if duplicate_pairs:
        lines.append("")
        lines.append("WARNING: byte-identical score vectors:")
        for a, b in duplicate_pairs:
            lines.append(f"  {a}  ==  {b}")
    COVERAGE_FILE.write_text("\n".join(lines) + "\n")
    print(f"✓ Wrote coverage report to {COVERAGE_FILE}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and parse HELM Safety data")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download even if cached data exists")
    parser.add_argument("--discover", action="store_true",
                        help="Print all (scenario, metric) pairs in the cache and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("HELM Safety Data Scraper")
    print("=" * 60)

    if not check_gcloud_installed():
        print("✗ gcloud CLI is not installed or not in PATH")
        print("  Install: https://cloud.google.com/sdk/docs/install")
        return 1
    print("✓ gcloud CLI is installed\n")

    if not download_helm_data(force=args.force_download):
        return 1

    stats_files = find_stats_files()
    if not stats_files:
        print("✗ No stats.json files found in cache")
        return 1

    if args.discover:
        discover_report(stats_files)
        return 0

    print("\nExtracting safety scores...")
    model_scores, resolved_metrics = extract_safety_scores(stats_files)
    if not model_scores:
        print("✗ No safety scores extracted. Run with --discover to inspect "
              "scenario/metric names in the bucket.")
        return 1

    print("\nCalculating mean scores...")
    mean_scores = calculate_mean_scores(model_scores)

    print("\nChecking for duplicate score vectors...")
    duplicate_pairs = detect_duplicate_vectors(model_scores)
    if not duplicate_pairs:
        print("✓ No duplicate vectors detected.")

    save_data(model_scores, mean_scores, resolved_metrics, duplicate_pairs)

    print("\n" + "=" * 60)
    print("Next: confirm coverage report; then run "
          "scripts/compute_helm_safety_partial_corr.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
