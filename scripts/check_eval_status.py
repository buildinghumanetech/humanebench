#!/usr/bin/env python3
"""
Check evaluation completion status across all task types.

This script uses the Inspect AI Python API to precisely determine which
evaluations completed successfully and which need to be retried.

Usage:
    python scripts/check_eval_status.py
    python scripts/check_eval_status.py --task-type good_persona
    python scripts/check_eval_status.py --show-errors
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

try:
    from inspect_ai.log import read_eval_log, list_eval_logs
except ImportError:
    print("Error: inspect_ai package not found. Please install it:")
    print("  pip install inspect-ai")
    sys.exit(1)


class EvaluationStatus:
    """Store status information for a single evaluation."""

    def __init__(self, task_type: str, model: str, log_path: str):
        self.task_type = task_type
        self.model = model
        self.log_path = log_path
        self.status = None
        self.expected_samples = None
        self.actual_samples = None
        self.error_message = None
        self.is_complete = False

    def check(self):
        """Read the log file and populate status information."""
        try:
            log = read_eval_log(self.log_path)

            self.status = log.status
            self.expected_samples = log.eval.dataset.samples if hasattr(log.eval, 'dataset') else 0
            self.actual_samples = len(log.samples) if log.samples else 0
            self.is_complete = (self.actual_samples == self.expected_samples and
                              self.status == "success")

            # Get error message if evaluation failed
            if self.status == "error" and hasattr(log, 'error') and log.error:
                if hasattr(log.error, 'message'):
                    self.error_message = log.error.message
                else:
                    self.error_message = str(log.error)

        except Exception as e:
            self.status = "parse_error"
            self.error_message = str(e)

    @property
    def completion_pct(self) -> float:
        """Calculate completion percentage."""
        if self.expected_samples == 0:
            return 0.0
        return (self.actual_samples / self.expected_samples) * 100

    @property
    def remaining_samples(self) -> int:
        """Calculate number of remaining samples."""
        return max(0, self.expected_samples - self.actual_samples)

    def __repr__(self):
        status_icon = "✓" if self.is_complete else "✗"
        return (f"{status_icon} {self.task_type}/{self.model}: "
                f"{self.actual_samples}/{self.expected_samples} samples "
                f"({self.completion_pct:.1f}%) - {self.status}")


def find_evaluations(log_dir: Path, task_types: Optional[List[str]] = None) -> List[EvaluationStatus]:
    """Find all evaluation log files and create status objects."""
    if task_types is None:
        task_types = ["baseline", "bad_persona", "good_persona"]

    evaluations = []

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
                print(f"Warning: No .eval files found in {model_dir}")
                continue

            # Use most recent file (sorted by modification time)
            most_recent = max(eval_files, key=lambda p: p.stat().st_mtime)

            eval_status = EvaluationStatus(
                task_type=task_type,
                model=model_name,
                log_path=str(most_recent)
            )
            evaluations.append(eval_status)

    return evaluations


def print_summary(evaluations: List[EvaluationStatus], show_errors: bool = False):
    """Print a comprehensive summary of evaluation status."""

    # Group by task type
    by_task_type: Dict[str, List[EvaluationStatus]] = {}
    for eval_status in evaluations:
        if eval_status.task_type not in by_task_type:
            by_task_type[eval_status.task_type] = []
        by_task_type[eval_status.task_type].append(eval_status)

    # Print results by task type
    for task_type in sorted(by_task_type.keys()):
        evals = by_task_type[task_type]
        print(f"\n{'='*80}")
        print(f"TASK TYPE: {task_type.upper()}")
        print(f"{'='*80}")

        complete = [e for e in evals if e.is_complete]
        incomplete = [e for e in evals if not e.is_complete]

        print(f"\nStatus: {len(complete)}/{len(evals)} complete")

        if complete:
            print(f"\n✓ COMPLETED ({len(complete)} models):")
            for eval_status in sorted(complete, key=lambda e: e.model):
                print(f"  ✓ {eval_status.model:<30} {eval_status.actual_samples}/{eval_status.expected_samples} samples")

        if incomplete:
            print(f"\n✗ INCOMPLETE ({len(incomplete)} models):")
            # Sort by completion percentage (descending)
            for eval_status in sorted(incomplete, key=lambda e: e.completion_pct, reverse=True):
                print(f"  ✗ {eval_status.model:<30} "
                      f"{eval_status.actual_samples}/{eval_status.expected_samples} samples "
                      f"({eval_status.completion_pct:>5.1f}% done, {eval_status.remaining_samples} remaining)")

                if show_errors and eval_status.error_message:
                    error_preview = eval_status.error_message[:100]
                    if len(eval_status.error_message) > 100:
                        error_preview += "..."
                    print(f"      Error: {error_preview}")

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    total = len(evaluations)
    total_complete = sum(1 for e in evaluations if e.is_complete)
    total_incomplete = total - total_complete
    total_samples = sum(e.expected_samples for e in evaluations)
    completed_samples = sum(e.actual_samples for e in evaluations)
    remaining_samples = total_samples - completed_samples

    print(f"Total evaluations: {total}")
    print(f"  Complete: {total_complete} ({total_complete/total*100:.1f}%)")
    print(f"  Incomplete: {total_incomplete} ({total_incomplete/total*100:.1f}%)")
    print(f"\nSamples:")
    print(f"  Total expected: {total_samples:,}")
    print(f"  Completed: {completed_samples:,} ({completed_samples/total_samples*100:.1f}%)")
    print(f"  Remaining: {remaining_samples:,} ({remaining_samples/total_samples*100:.1f}%)")

    # Cost estimate
    if total_incomplete > 0:
        # Based on actual cost data: $942 for 32 evals = $29.44/eval
        avg_cost_per_eval = 29.44
        avg_cost_per_sample = avg_cost_per_eval / 800
        estimated_cost = remaining_samples * avg_cost_per_sample

        print(f"\nEstimated cost to complete remaining work:")
        print(f"  ~${estimated_cost:.2f} (based on $29.44 per 800-sample evaluation)")

    # Generate retry commands
    incomplete_by_task = {}
    for eval_status in evaluations:
        if not eval_status.is_complete:
            if eval_status.task_type not in incomplete_by_task:
                incomplete_by_task[eval_status.task_type] = []
            incomplete_by_task[eval_status.task_type].append(eval_status)

    if incomplete_by_task:
        print(f"\n{'='*80}")
        print("RETRY COMMANDS")
        print(f"{'='*80}")

        for task_type in sorted(incomplete_by_task.keys()):
            incomplete_evals = incomplete_by_task[task_type]
            print(f"\n# {task_type.upper()} ({len(incomplete_evals)} models to retry)")

            # Sort by remaining work (least to most)
            for eval_status in sorted(incomplete_evals, key=lambda e: e.remaining_samples):
                print(f"inspect eval-retry {eval_status.log_path}  # {eval_status.remaining_samples} samples remaining")


def main():
    parser = argparse.ArgumentParser(
        description="Check evaluation completion status across all task types."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing evaluation logs (default: logs)"
    )
    parser.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        help="Task type to check (can be specified multiple times). "
             "Default: all task types (baseline, bad_persona, good_persona)"
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show error messages for failed evaluations"
    )

    args = parser.parse_args()

    # Find all evaluations
    print(f"Scanning for evaluations in {args.log_dir}...")
    evaluations = find_evaluations(args.log_dir, args.task_types)

    if not evaluations:
        print("No evaluations found!")
        sys.exit(1)

    print(f"Found {len(evaluations)} evaluations. Checking status...\n")

    # Check status of each evaluation
    for eval_status in evaluations:
        eval_status.check()

    # Print summary
    print_summary(evaluations, show_errors=args.show_errors)


if __name__ == "__main__":
    main()
