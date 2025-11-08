#!/usr/bin/env python3
"""
Export Inspect AI evaluation logs to Langfuse for human annotation.

This script extracts input/output pairs from Inspect AI .eval files and uploads them
to Langfuse for human review and annotation. Only the user prompt and AI response are
exported (system prompts, overseer scoring, and explanations are excluded).

Setup:
1. Install dependencies: pip install langfuse python-dotenv

2. Create a .env file in the project root with:
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL

3. (Optional) For annotation queues, create .langfuse_queues.json mapping principles to queue IDs:
   Copy .langfuse_queues.example.json to .langfuse_queues.json and fill in your queue IDs

Usage:
    # Export all traces from all .eval files (no filtering)
    python inspect_logs_to_langfuse.py

    # Export only traces for a specific principle
    python inspect_logs_to_langfuse.py --principle respect-user-attention

    # Export specific principle AND add to its annotation queue
    python inspect_logs_to_langfuse.py --principle respect-user-attention --enqueue

    # Export specific eval file(s), filtered by principle
    python inspect_logs_to_langfuse.py --principle enable-meaningful-choices logs/specific.eval

Valid principles:
    - be-transparent-and-honest
    - design-for-equity-and-inclusion
    - enable-meaningful-choices
    - enhance-human-capabilities
    - foster-healthy-relationships
    - prioritize-long-term-wellbeing
    - protect-dignity-and-safety
    - respect-user-attention
"""

import zipfile
import json
import os
import sys
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv package not installed.")
    print("Install with: pip install python-dotenv")
    sys.exit(1)

try:
    from langfuse import get_client
except ImportError:
    print("Error: langfuse package not installed.")
    print("Install with: pip install langfuse")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# List of valid humane principles
VALID_PRINCIPLES = [
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "protect-dignity-and-safety",
    "respect-user-attention"
]


def load_queue_config(config_path: str = ".langfuse_queues.json") -> Dict[str, str]:
    """
    Load the principle-to-queue mapping from config file.

    Returns a dictionary mapping principle names to queue IDs.
    """
    config_file = Path(__file__).parent.parent / config_path
    if not config_file.exists():
        return {}

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Remove 'description' field if present
        if "description" in config:
            del config["description"]

        return config
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse config file {config_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error: Failed to load config file {config_path}: {e}")
        return {}


def extract_user_input(sample: Dict[str, Any]) -> str:
    """Extract only the user input from sample messages (excluding system prompts)."""
    # First try the direct input field
    if "input" in sample and sample["input"]:
        return sample["input"]

    # Otherwise, extract from messages with role="user"
    if "messages" in sample:
        for msg in sample["messages"]:
            if msg.get("role") == "user":
                return msg.get("content", "")

    return ""


def extract_ai_output(sample: Dict[str, Any]) -> str:
    """Extract only the AI's response (excluding overseer/scoring info)."""
    # Use the completion field from output
    if "output" in sample and "completion" in sample["output"]:
        return sample["output"]["completion"]

    # Fallback: extract from messages with role="assistant"
    if "messages" in sample:
        for msg in sample["messages"]:
            if msg.get("role") == "assistant" and msg.get("source") == "generate":
                return msg.get("content", "")

    return ""


def extract_sample_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all relevant metadata from the sample."""
    metadata = {
        "sample_id": sample.get("id", ""),
        "target": sample.get("target", ""),
    }

    # Add metadata from the sample's metadata field
    if "metadata" in sample and "metadata" in sample["metadata"]:
        sample_meta = sample["metadata"]["metadata"]
        metadata.update({
            "principle": sample_meta.get("principle", ""),
            "domain": sample_meta.get("domain", ""),
            "vulnerable_population": sample_meta.get("vulnerable-population", ""),
        })

    # Add model information
    if "output" in sample and "model" in sample["output"]:
        metadata["model"] = sample["output"]["model"]

    # Add token usage if available
    if "output" in sample and "usage" in sample["output"]:
        usage = sample["output"]["usage"]
        metadata["input_tokens"] = usage.get("input_tokens", 0)
        metadata["output_tokens"] = usage.get("output_tokens", 0)
        metadata["total_tokens"] = usage.get("total_tokens", 0)

    return metadata


def add_to_annotation_queue(
    span_id: str,
    queue_id: str,
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com"
) -> bool:
    """
    Add a span/generation to an annotation queue via Langfuse API.

    Returns True if successful, False otherwise.
    """
    url = f"{host}/api/public/annotation-queues/{queue_id}/items"

    try:
        response = requests.post(
            url,
            auth=(public_key, secret_key),
            json={
                "objectId": span_id,
                "objectType": "OBSERVATION"
            },
            timeout=10
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"    Warning: Failed to add to annotation queue: {e}")
        return False


def export_eval_to_langfuse(
    eval_file_path: str,
    langfuse,
    filter_principle: Optional[str] = None,
    enqueue: bool = False,
    queue_id: Optional[str] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: str = "https://cloud.langfuse.com"
) -> int:
    """
    Export a single Inspect AI .eval file to Langfuse.

    Args:
        filter_principle: If provided, only export samples matching this principle.

    Returns the number of samples exported.
    """
    eval_path = Path(eval_file_path)
    if not eval_path.exists():
        print(f"Warning: {eval_file_path} not found, skipping.")
        return 0

    print(f"Processing {eval_path.name}...")

    try:
        with zipfile.ZipFile(eval_file_path, 'r') as z:
            # Read header for evaluation metadata
            with z.open('header.json') as f:
                header = json.load(f)

            eval_info = header.get("eval", {})
            task_name = eval_info.get("task", "unknown")
            model_name = eval_info.get("model", "unknown")
            eval_id = eval_info.get("eval_id", eval_path.stem)
            created = eval_info.get("created", "")

            # Create a trace for this evaluation run
            trace_metadata = {
                "task": task_name,
                "model": model_name,
                "created": created,
                "eval_file": eval_path.name,
                "eval_id": eval_id,
            }
            if filter_principle:
                trace_metadata["filter_principle"] = filter_principle

            with langfuse.start_as_current_span(
                name=f"inspect-eval-{task_name}",
                metadata=trace_metadata
            ) as trace:
                # Process each sample
                sample_files = [f for f in z.namelist() if f.startswith('samples/') and f.endswith('.json')]
                sample_count = 0
                skipped_count = 0

                for sample_file in sample_files:
                    with z.open(sample_file) as f:
                        sample = json.load(f)

                    # Extract all metadata first (needed for filtering)
                    metadata = extract_sample_metadata(sample)

                    # Filter by principle if specified
                    if filter_principle:
                        sample_principle = metadata.get("principle", "")
                        if sample_principle != filter_principle:
                            skipped_count += 1
                            continue

                    # Extract input and output (excluding system prompts and scoring)
                    user_input = extract_user_input(sample)
                    ai_output = extract_ai_output(sample)

                    if not user_input or not ai_output:
                        print(f"  Warning: Skipping sample {sample.get('id', 'unknown')} - missing input or output")
                        continue

                    # Create a generation for this sample
                    with langfuse.start_as_current_observation(
                        name=f"sample-{sample.get('id', sample_count)}",
                        as_type="generation",
                        model=metadata.get("model", model_name),
                        input=user_input,
                        output=ai_output,
                        metadata=metadata,
                    ) as generation:
                        generation.update(
                            usage_details={
                                "input": metadata.get("input_tokens", 0),
                                "output": metadata.get("output_tokens", 0),
                                "total": metadata.get("total_tokens", 0),
                            }
                        )

                        # Add to annotation queue if requested
                        if enqueue and queue_id:
                            # Get the span ID from the generation object
                            span_id = generation.id
                            if span_id:
                                add_to_annotation_queue(
                                    span_id=span_id,
                                    queue_id=queue_id,
                                    public_key=public_key,
                                    secret_key=secret_key,
                                    host=host
                                )

                    sample_count += 1

            # Print summary
            if filter_principle:
                print(f"  Exported {sample_count} samples from {eval_path.name} (skipped {skipped_count} not matching '{filter_principle}')")
            else:
                print(f"  Exported {sample_count} samples from {eval_path.name}")
            return sample_count

    except Exception as e:
        print(f"Error processing {eval_file_path}: {e}")
        return 0


def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Export Inspect AI evaluation logs to Langfuse for human annotation."
    )
    parser.add_argument(
        "--principle",
        type=str,
        choices=VALID_PRINCIPLES,
        help="Filter traces by humane principle (required when using --enqueue)"
    )
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Add exported items to annotation queue (requires --principle and .langfuse_queues.json)"
    )
    parser.add_argument(
        "eval_files",
        nargs="*",
        help="Specific .eval files to process (if not provided, processes all files in logs/)"
    )

    args = parser.parse_args()

    # Validate --enqueue requires --principle
    if args.enqueue and not args.principle:
        parser.error("--enqueue requires --principle to be specified")

    # Validate principle if provided
    if args.principle and args.principle not in VALID_PRINCIPLES:
        print(f"Error: Invalid principle '{args.principle}'")
        print(f"Valid principles: {', '.join(VALID_PRINCIPLES)}")
        sys.exit(1)

    # Check for required environment variables
    required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables before running the script.")
        print("See the script docstring for setup instructions.")
        sys.exit(1)

    # Get credentials for API calls
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Load queue configuration and determine queue ID if --enqueue is used
    queue_id = None
    if args.enqueue:
        queue_config = load_queue_config()
        if not queue_config:
            print("Error: --enqueue requires .langfuse_queues.json config file")
            print("Copy .langfuse_queues.example.json to .langfuse_queues.json and fill in your queue IDs.")
            sys.exit(1)

        # Get queue ID for the specified principle
        queue_id = queue_config.get(args.principle)
        if not queue_id:
            print(f"Error: No queue ID found for principle '{args.principle}' in .langfuse_queues.json")
            print(f"Please add a queue ID for this principle in the config file.")
            sys.exit(1)

        print(f"Annotation queue enabled for principle '{args.principle}' (Queue ID: {queue_id})\n")

    # Initialize Langfuse client
    try:
        langfuse = get_client()
        print("Connected to Langfuse successfully.\n")
    except Exception as e:
        print(f"Error connecting to Langfuse: {e}")
        sys.exit(1)

    # Determine which files to process
    if args.eval_files:
        # Process specific files provided as arguments
        eval_files = args.eval_files
        print(f"Processing {len(eval_files)} specified file(s)...\n")
    else:
        # Process all .eval files in logs/ directory
        logs_dir = Path(__file__).parent.parent / "logs"
        if not logs_dir.exists():
            print(f"Error: logs directory not found at {logs_dir}")
            sys.exit(1)

        eval_files = list(logs_dir.glob("*.eval"))
        if not eval_files:
            print(f"No .eval files found in {logs_dir}")
            sys.exit(0)

        eval_files = [str(f) for f in eval_files]
        print(f"Processing all {len(eval_files)} .eval file(s) from logs/...\n")

    # Export each file
    total_samples = 0
    for eval_file in eval_files:
        samples_exported = export_eval_to_langfuse(
            eval_file,
            langfuse,
            filter_principle=args.principle,
            enqueue=args.enqueue,
            queue_id=queue_id,
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        total_samples += samples_exported

    # Ensure all data is sent to Langfuse
    langfuse.flush()

    print(f"\n{'='*60}")
    print(f"Export complete!")
    if args.principle:
        print(f"Principle filter: {args.principle}")
    print(f"Total samples exported: {total_samples}")
    if args.enqueue:
        print(f"Items added to annotation queue: {queue_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
