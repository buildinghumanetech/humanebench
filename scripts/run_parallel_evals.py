#!/usr/bin/env python3
"""
Run parallel evaluations across multiple models and task types.

This script orchestrates running Inspect AI evaluations in parallel,
organizing logs into subdirectories by task type and model name.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import concurrent.futures
import time

# Available task types
TASK_TYPES = ["baseline", "good_persona", "bad_persona"]

# Default models (can be overridden via command line)
DEFAULT_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4-5-20250929",
]


def run_evaluation(task_type: str, model: str, log_dir: Path) -> dict:
    """
    Run a single evaluation task.

    Args:
        task_type: Type of task (baseline, good_persona, bad_persona)
        model: Model identifier
        log_dir: Directory to store logs

    Returns:
        Dict with execution details
    """
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build inspect command
    task_file = f"src/{task_type}_task.py"
    cmd = [
        "inspect", "eval",
        task_file,
        f"--model={model}",
        f"--log-dir={log_dir}",
    ]

    start_time = time.time()
    result = {
        "task_type": task_type,
        "model": model,
        "log_dir": str(log_dir),
        "success": False,
        "duration": 0,
        "error": None,
    }

    try:
        print(f"Starting: {task_type} with {model}")
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        result["success"] = True
        result["duration"] = time.time() - start_time
        print(f"✓ Completed: {task_type} with {model} ({result['duration']:.1f}s)")

    except subprocess.CalledProcessError as e:
        result["duration"] = time.time() - start_time
        result["error"] = f"Exit code {e.returncode}: {e.stderr[:200]}"
        print(f"✗ Failed: {task_type} with {model} - {result['error']}")
    except Exception as e:
        result["duration"] = time.time() - start_time
        result["error"] = str(e)
        print(f"✗ Error: {task_type} with {model} - {result['error']}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel evaluations across models and task types"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=f"Model identifiers to evaluate (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        choices=TASK_TYPES,
        default=TASK_TYPES,
        help=f"Task types to run (default: all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--log-base-dir",
        type=Path,
        default=Path("logs"),
        help="Base directory for logs (default: logs)",
    )

    args = parser.parse_args()

    # Use provided models or defaults
    models = args.models if args.models else DEFAULT_MODELS

    # Build list of evaluations to run
    evaluations = []
    for task_type in args.task_types:
        for model in models:
            # Extract model name for directory (remove provider prefix)
            model_name = model.split("/")[-1] if "/" in model else model
            log_dir = args.log_base_dir / task_type / model_name
            evaluations.append((task_type, model, log_dir))

    print(f"\n{'='*60}")
    print(f"Running {len(evaluations)} evaluations with {args.max_workers} workers")
    print(f"Task types: {', '.join(args.task_types)}")
    print(f"Models: {', '.join(models)}")
    print(f"{'='*60}\n")

    # Run evaluations in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_evaluation, task_type, model, log_dir)
            for task_type, model, log_dir in evaluations
        ]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed evaluations:")
        for r in failed:
            print(f"  - {r['task_type']} / {r['model']}: {r['error']}")

    total_duration = sum(r["duration"] for r in results)
    print(f"\nTotal execution time: {total_duration:.1f}s")
    print(f"{'='*60}\n")

    # Exit with error if any failed
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
