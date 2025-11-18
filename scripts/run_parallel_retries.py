#!/usr/bin/env python3
"""
Run parallel retry operations for incomplete evaluations.

This script auto-discovers incomplete evaluations and retries them in parallel,
preserving completed samples to minimize cost. Uses subprocess to call the
Inspect AI CLI for parallel execution.

Usage Examples:
    # Auto-discover and retry all incomplete evaluations
    python scripts/run_parallel_retries.py

    # Preview what would run without executing
    python scripts/run_parallel_retries.py --dry-run

    # Use more parallel workers for faster completion
    python scripts/run_parallel_retries.py --max-workers 8

    # Only retry specific task type
    python scripts/run_parallel_retries.py --task-types good_persona

    # Retry specific files
    python scripts/run_parallel_retries.py --files logs/good_persona/grok-4/*.eval
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

try:
    from inspect_ai.log import read_eval_log
except ImportError:
    print("Error: inspect_ai package not found. Please install it:")
    print("  pip install inspect-ai")
    sys.exit(1)


@dataclass
class RetryTask:
    """Represents a single evaluation retry task."""
    task_type: str
    model: str
    log_path: str
    expected_samples: int
    actual_samples: int
    remaining_samples: int
    completion_pct: float

    @property
    def display_name(self):
        """Format for display: [task_type/model]"""
        return f"[{self.task_type}/{self.model}]"

    @property
    def estimated_cost(self):
        """Estimate cost based on remaining samples."""
        # Based on actual cost data: $29.44 per 800-sample evaluation
        cost_per_sample = 29.44 / 800
        return self.remaining_samples * cost_per_sample


def find_incomplete_evaluations(
    log_dir: Path,
    task_types: Optional[List[str]] = None
) -> List[RetryTask]:
    """
    Find all incomplete evaluations by scanning the log directory.

    Args:
        log_dir: Root directory containing evaluation logs
        task_types: Optional list of task types to filter

    Returns:
        List of RetryTask objects for incomplete evaluations
    """
    if task_types is None:
        task_types = ["baseline", "bad_persona", "good_persona"]

    retry_tasks = []

    for task_type in task_types:
        task_log_dir = log_dir / task_type
        if not task_log_dir.exists():
            print(f"Warning: Directory not found: {task_log_dir}")
            continue

        # Find all model directories
        for model_dir in task_log_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Find most recent .eval file in this directory
            eval_files = list(model_dir.glob("*.eval"))
            if not eval_files:
                continue

            # Use most recent file (sorted by modification time)
            most_recent = max(eval_files, key=lambda p: p.stat().st_mtime)

            # Check if evaluation is incomplete
            try:
                log = read_eval_log(str(most_recent))

                expected_samples = log.eval.dataset.samples if hasattr(log.eval, 'dataset') else 0
                actual_samples = len(log.samples) if log.samples else 0
                is_complete = (actual_samples == expected_samples and log.status == "success")

                if not is_complete:
                    retry_task = RetryTask(
                        task_type=task_type,
                        model=model_name,
                        log_path=str(most_recent),
                        expected_samples=expected_samples,
                        actual_samples=actual_samples,
                        remaining_samples=max(0, expected_samples - actual_samples),
                        completion_pct=(actual_samples / expected_samples * 100) if expected_samples > 0 else 0
                    )
                    retry_tasks.append(retry_task)

            except Exception as e:
                print(f"Warning: Failed to read {most_recent}: {e}")
                continue

    return retry_tasks


def run_retry(
    retry_task: RetryTask,
    max_connections: int = 10
) -> dict:
    """
    Execute a single retry operation using subprocess.

    Args:
        retry_task: The retry task to execute
        max_connections: Maximum API connections per retry

    Returns:
        Dictionary with result information
    """
    start_time = time.time()
    result = {
        'task': retry_task,
        'success': False,
        'error': None,
        'duration': 0,
        'new_log_path': None
    }

    # Build inspect eval-retry command
    cmd = [
        "inspect", "eval-retry",
        retry_task.log_path,
        f"--max-connections={max_connections}"
    ]

    prefix = retry_task.display_name

    try:
        print(f"{prefix} Starting retry: {retry_task.remaining_samples} samples remaining...")

        # Execute retry using subprocess (avoids concurrent eval_async limitation)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered for streaming
        )

        # Stream output line by line with prefix
        for line in process.stdout:
            print(f"{prefix} {line.rstrip()}")

        # Wait for process to complete
        process.wait()

        duration = time.time() - start_time
        result['duration'] = duration

        if process.returncode == 0:
            result['success'] = True
            print(f"{prefix} ✓ Completed successfully ({duration:.1f}s)")
        else:
            result['error'] = f"Exit code {process.returncode}"
            print(f"{prefix} ✗ Failed with exit code {process.returncode}")

    except Exception as e:
        result['error'] = str(e)
        duration = time.time() - start_time
        result['duration'] = duration
        print(f"{prefix} ✗ Error: {e}")

    return result


def print_summary(retry_tasks: List[RetryTask]):
    """Print a summary of tasks to be retried."""
    print("\n" + "="*80)
    print("PARALLEL RETRY SUMMARY")
    print("="*80)

    # Group by task type
    by_task_type = {}
    for task in retry_tasks:
        if task.task_type not in by_task_type:
            by_task_type[task.task_type] = []
        by_task_type[task.task_type].append(task)

    total_samples = sum(t.remaining_samples for t in retry_tasks)
    total_cost = sum(t.estimated_cost for t in retry_tasks)

    for task_type in sorted(by_task_type.keys()):
        tasks = by_task_type[task_type]
        task_samples = sum(t.remaining_samples for t in tasks)
        task_cost = sum(t.estimated_cost for t in tasks)

        print(f"\n{task_type.upper()}: {len(tasks)} tasks, {task_samples} samples, ~${task_cost:.2f}")
        for task in sorted(tasks, key=lambda t: t.remaining_samples):
            print(f"  • {task.model:<30} {task.remaining_samples:>4} samples "
                  f"({task.completion_pct:>5.1f}% done)")

    print(f"\n{'='*80}")
    print(f"TOTAL: {len(retry_tasks)} tasks, {total_samples:,} samples, ~${total_cost:.2f}")
    print(f"{'='*80}\n")


def print_results(results: List[dict]):
    """Print final results summary."""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print("\n" + "="*80)
    print("RETRY RESULTS")
    print("="*80)

    if successful:
        print(f"\n✓ SUCCESSFUL ({len(successful)} tasks):")
        total_duration = sum(r['duration'] for r in successful)
        for result in successful:
            task = result['task']
            print(f"  ✓ {task.display_name:<40} {result['duration']:.1f}s")
        print(f"\nTotal time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")

    if failed:
        print(f"\n✗ FAILED ({len(failed)} tasks):")
        for result in failed:
            task = result['task']
            error_preview = result['error'][:80] if result['error'] else "Unknown error"
            print(f"  ✗ {task.display_name:<40} {error_preview}")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(successful)}/{len(results)} successful, {len(failed)}/{len(results)} failed")
    print(f"{'='*80}\n")

    if failed:
        print("To retry failed tasks, run:")
        for result in failed:
            task = result['task']
            print(f"  inspect eval-retry {task.log_path}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel retry operations for incomplete evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and retry all incomplete evaluations
  %(prog)s

  # Preview what would run (dry-run mode)
  %(prog)s --dry-run

  # Use 8 parallel workers for faster completion
  %(prog)s --max-workers 8

  # Only retry specific task type
  %(prog)s --task-types good_persona

  # Retry specific files
  %(prog)s --files logs/good_persona/grok-4/*.eval
        """
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing evaluation logs (default: logs)"
    )

    parser.add_argument(
        "--task-types",
        action="append",
        dest="task_types",
        choices=["baseline", "bad_persona", "good_persona"],
        help="Task type to retry (can be specified multiple times). "
             "Default: all task types"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel retry workers (default: 4)"
    )

    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Maximum API connections per retry (default: 10)"
    )

    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit .eval files to retry (overrides auto-discovery)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be executed without running retries"
    )

    args = parser.parse_args()

    print("="*80)
    print("PARALLEL EVALUATION RETRY")
    print("="*80)
    print()

    # Find incomplete evaluations
    if args.files:
        # Use explicit files
        print(f"Using {len(args.files)} explicit file(s)...")
        retry_tasks = []
        for file_path in args.files:
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            try:
                log = read_eval_log(str(path))
                # Extract task type and model from path
                parts = path.parts
                if len(parts) >= 2:
                    task_type = parts[-2]
                    model = parts[-1].split('/')[0] if '/' in parts[-1] else "unknown"
                else:
                    task_type = "unknown"
                    model = "unknown"

                expected_samples = log.eval.dataset.samples if hasattr(log.eval, 'dataset') else 0
                actual_samples = len(log.samples) if log.samples else 0

                retry_task = RetryTask(
                    task_type=task_type,
                    model=model,
                    log_path=str(path),
                    expected_samples=expected_samples,
                    actual_samples=actual_samples,
                    remaining_samples=max(0, expected_samples - actual_samples),
                    completion_pct=(actual_samples / expected_samples * 100) if expected_samples > 0 else 0
                )
                retry_tasks.append(retry_task)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
    else:
        # Auto-discover
        print(f"Scanning {args.log_dir} for incomplete evaluations...")
        retry_tasks = find_incomplete_evaluations(args.log_dir, args.task_types)

    if not retry_tasks:
        print("\n✓ No incomplete evaluations found! All tasks are complete.")
        print("\nTo verify, run: python scripts/check_eval_status.py")
        return

    # Print summary
    print_summary(retry_tasks)

    if args.dry_run:
        print("DRY-RUN MODE: No retries will be executed.")
        print("\nTo execute retries, run without --dry-run flag:")
        print(f"  python {sys.argv[0]}")
        return

    # Confirm before proceeding
    print(f"Ready to retry {len(retry_tasks)} evaluations with {args.max_workers} parallel workers.")
    print("This will:")
    print("  • Preserve all completed samples (only retry remaining)")
    print("  • Create new .eval files (originals preserved)")
    print(f"  • Cost approximately ${sum(t.estimated_cost for t in retry_tasks):.2f}")
    print()

    response = input("Proceed with retries? [y/N] ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return

    # Execute retries in parallel
    print("\n" + "="*80)
    print(f"EXECUTING RETRIES ({args.max_workers} parallel workers)")
    print("="*80 + "\n")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all retry tasks
        future_to_task = {
            executor.submit(run_retry, task, args.max_connections): task
            for task in retry_tasks
        }

        # Process results as they complete
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)

    total_duration = time.time() - start_time

    # Print results
    print_results(results)
    print(f"Total execution time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)\n")

    # Suggest verification
    print("Next steps:")
    print("  1. Verify completion: python scripts/check_eval_status.py")
    if any(not r['success'] for r in results):
        print("  2. Retry failed tasks (see commands above)")


if __name__ == "__main__":
    main()
