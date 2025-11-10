#!/usr/bin/env python3
"""
Export Inspect AI evaluation logs to Langfuse for multiple users' annotation queues.

This script extends inspect_logs_to_langfuse.py to populate multiple users' queues
with the SAME sampled data for inter-annotator reliability.

Setup:
1. Install dependencies: pip install langfuse python-dotenv

2. Create a .env file in the project root with:
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL

3. Create queue_ids.csv with columns: user, principle1_queue_id, principle2_queue_id, ...
   Example:
   user,be-transparent-and-honest,design-for-equity-and-inclusion,enable-meaningful-choices,...
   alice,queue-id-1,queue-id-2,queue-id-3,...
   bob,queue-id-4,queue-id-5,queue-id-6,...

Usage:
    # Upload to all users' queues for one principle with sampling
    python scripts/inspect_logs_to_langfuse_multi_user.py --principle respect-user-attention --sample-limit 100 --queue-csv queue_ids.csv --enqueue

    # Upload without sampling (all matching samples)
    python scripts/inspect_logs_to_langfuse_multi_user.py --principle respect-user-attention --queue-csv queue_ids.csv --enqueue

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
import random
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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


def load_user_queue_config(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load user queue configuration from CSV file.

    Expected format (wide):
    Name,Email,Domain expertise,QUEUE IDs -->,principle1,principle2,...
    alice,alice@example.com,Expert,,queue-id-1,queue-id-2,...
    bob,bob@example.com,Researcher,,queue-id-3,queue-id-4,...

    Or simple format:
    user,principle1,principle2,...
    alice,queue-id-1,queue-id-2,...

    Returns:
        Dictionary mapping username to {principle: queue_id}
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Queue CSV file not found: {csv_path}")
        sys.exit(1)

    try:
        user_queues = {}
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try 'Name' first (real format), then fall back to 'user' (simple format)
                username = row.get('Name', row.get('user', '')).strip()
                if not username:
                    continue

                # Build principle -> queue_id mapping for this user
                user_queues[username] = {}
                for principle in VALID_PRINCIPLES:
                    queue_id = row.get(principle, '').strip()
                    # Only add if queue_id exists and looks valid (at least 5 chars)
                    if queue_id and len(queue_id) >= 5:
                        user_queues[username][principle] = queue_id

        return user_queues
    except Exception as e:
        print(f"Error: Failed to load queue CSV file {csv_path}: {e}")
        sys.exit(1)


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


def get_sample_ids_from_queue(
    queue_id: str,
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com"
) -> set:
    """
    Query an existing annotation queue to extract sample IDs.

    Returns a set of sample IDs that were already uploaded to this queue.
    """
    print(f"  Querying queue {queue_id} to extract existing sample IDs...")

    sample_ids = set()
    page = 1
    limit = 100

    try:
        langfuse = get_client()

        while True:
            # Get queue items (paginated)
            url = f"{host}/api/public/annotation-queues/{queue_id}/items"
            response = requests.get(
                url,
                auth=(public_key, secret_key),
                params={"page": page, "limit": limit},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Handle both array response and paginated response
            items = data if isinstance(data, list) else data.get("data", [])

            if not items:
                break

            # Extract sample IDs from each queue item
            for item in items:
                object_id = item.get("objectId")
                object_type = item.get("objectType")

                if not object_id:
                    continue

                # Fetch the actual observation/trace to get metadata
                try:
                    if object_type == "OBSERVATION":
                        # Get observation details
                        obs_url = f"{host}/api/public/observations/{object_id}"
                        obs_response = requests.get(
                            obs_url,
                            auth=(public_key, secret_key),
                            timeout=10
                        )
                        obs_response.raise_for_status()
                        observation = obs_response.json()

                        # Extract sample_id from metadata
                        metadata = observation.get("metadata", {})
                        sample_id = metadata.get("sample_id")
                        if sample_id:
                            sample_ids.add(sample_id)

                except requests.exceptions.RequestException as e:
                    print(f"    Warning: Failed to fetch object {object_id}: {e}")
                    continue

            # Check if there are more pages
            if isinstance(data, dict) and "meta" in data:
                total_pages = data["meta"].get("totalPages", 1)
                if page >= total_pages:
                    break
            else:
                # If no pagination metadata, assume we got all items
                break

            page += 1

        print(f"  Found {len(sample_ids)} sample IDs in reference queue")
        return sample_ids

    except requests.exceptions.RequestException as e:
        print(f"  Error querying annotation queue: {e}")
        print(f"  Falling back to empty sample set")
        return set()
    except Exception as e:
        print(f"  Unexpected error querying queue: {e}")
        print(f"  Falling back to empty sample set")
        return set()


def collect_samples_for_sampling(
    eval_files: List[str],
    filter_principle: Optional[str] = None,
    filter_sample_ids: Optional[set] = None
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """
    Collect all samples from eval files, grouped by model.

    Args:
        filter_principle: Only include samples matching this principle
        filter_sample_ids: If provided, only include samples with IDs in this set

    Returns:
        Dictionary mapping model names to list of (eval_file, sample) tuples
    """
    samples_by_model = {}

    for eval_file_path in eval_files:
        eval_path = Path(eval_file_path)
        if not eval_path.exists():
            continue

        try:
            with zipfile.ZipFile(eval_file_path, 'r') as z:
                # Read header for model info
                with z.open('header.json') as f:
                    header = json.load(f)

                eval_info = header.get("eval", {})
                model_name = eval_info.get("model", "unknown")

                # Process each sample
                sample_files = [f for f in z.namelist() if f.startswith('samples/') and f.endswith('.json')]

                for sample_file in sample_files:
                    with z.open(sample_file) as f:
                        sample = json.load(f)

                    # Extract metadata for filtering
                    metadata = extract_sample_metadata(sample)

                    # Filter by principle if specified
                    if filter_principle:
                        sample_principle = metadata.get("principle", "")
                        if sample_principle != filter_principle:
                            continue

                    # Filter by sample IDs if specified (for reference user mode)
                    if filter_sample_ids is not None:
                        sample_id = sample.get("id", "")
                        if sample_id not in filter_sample_ids:
                            continue

                    # Validate input and output exist
                    user_input = extract_user_input(sample)
                    ai_output = extract_ai_output(sample)
                    if not user_input or not ai_output:
                        continue

                    # Skip samples without scores
                    if not sample.get('scores') or len(sample.get('scores', {})) == 0:
                        continue

                    # Add to samples by model
                    if model_name not in samples_by_model:
                        samples_by_model[model_name] = []

                    samples_by_model[model_name].append((eval_file_path, sample))

        except Exception as e:
            print(f"Warning: Error collecting samples from {eval_file_path}: {e}")
            continue

    return samples_by_model


def sample_evenly_across_models(
    samples_by_model: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    target_count: int
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """
    Sample evenly across models to get target_count total samples.

    Returns:
        Dictionary with sampled items (same structure as input)
    """
    if not samples_by_model:
        return {}

    # Calculate samples per model (distribute evenly)
    num_models = len(samples_by_model)
    samples_per_model = target_count // num_models
    remainder = target_count % num_models

    sampled = {}
    models = sorted(samples_by_model.keys())  # Sort for deterministic ordering

    for i, model in enumerate(models):
        available = samples_by_model[model]

        # Give one extra sample to first 'remainder' models
        target_for_model = samples_per_model + (1 if i < remainder else 0)

        # Sample randomly from available samples
        num_to_sample = min(target_for_model, len(available))
        if num_to_sample > 0:
            sampled[model] = random.sample(available, num_to_sample)
        else:
            sampled[model] = []

    return sampled


def export_samples_for_user(
    username: str,
    eval_files_and_samples: List[Tuple[str, Dict[str, Any]]],
    langfuse,
    filter_principle: str,
    enqueue: bool,
    queue_id: str,
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com"
) -> int:
    """
    Export sampled data to Langfuse for a specific user.
    Creates one trace per eval file with generations for matching samples.

    Returns the number of samples exported.
    """
    # Group samples by eval file
    samples_by_eval = {}
    for eval_file, sample in eval_files_and_samples:
        if eval_file not in samples_by_eval:
            samples_by_eval[eval_file] = []
        samples_by_eval[eval_file].append(sample)

    total_exported = 0

    for eval_file_path, samples in samples_by_eval.items():
        eval_path = Path(eval_file_path)

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

                # Create a trace for this user's evaluation
                trace_metadata = {
                    "task": task_name,
                    "model": model_name,
                    "created": created,
                    "eval_file": eval_path.name,
                    "eval_id": eval_id,
                    "filter_principle": filter_principle,
                    "annotator": username,
                }

                with langfuse.start_as_current_span(
                    name=f"inspect-eval-{task_name}-user-{username}",
                    metadata=trace_metadata
                ) as trace:
                    # Process each sample for this user
                    for sample in samples:
                        # Extract metadata
                        metadata = extract_sample_metadata(sample)

                        # Extract input and output
                        user_input = extract_user_input(sample)
                        ai_output = extract_ai_output(sample)

                        if not user_input or not ai_output:
                            continue

                        # Create a generation for this sample
                        with langfuse.start_as_current_observation(
                            name=f"sample-{sample.get('id', 'unknown')}",
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
                                span_id = generation.id
                                if span_id:
                                    add_to_annotation_queue(
                                        span_id=span_id,
                                        queue_id=queue_id,
                                        public_key=public_key,
                                        secret_key=secret_key,
                                        host=host
                                    )

                        total_exported += 1

        except Exception as e:
            print(f"  Warning: Error processing {eval_file_path} for user {username}: {e}")
            continue

    return total_exported


def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Export Inspect AI evaluation logs to Langfuse for multiple users' annotation queues."
    )
    parser.add_argument(
        "--principle",
        type=str,
        choices=VALID_PRINCIPLES,
        required=True,
        help="Filter traces by humane principle (required)"
    )
    parser.add_argument(
        "--queue-csv",
        type=str,
        required=True,
        help="Path to CSV file with user queue IDs (wide format)"
    )
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Add exported items to annotation queues"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Limit total samples exported with even distribution across models (e.g., 100)"
    )
    parser.add_argument(
        "--reference-user",
        type=str,
        help="Query this user's queue to use their exact sample IDs (for late-arriving evaluators)"
    )
    parser.add_argument(
        "eval_files",
        nargs="*",
        help="Specific .eval files to process (if not provided, recursively processes all files in logs/)"
    )

    args = parser.parse_args()

    # Check for required environment variables
    required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables before running the script.")
        sys.exit(1)

    # Get credentials for API calls
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Load user queue configuration
    print(f"Loading user queue configuration from {args.queue_csv}...")
    user_queues = load_user_queue_config(args.queue_csv)

    if not user_queues:
        print("Error: No users found in queue CSV file")
        sys.exit(1)

    # Validate that users have queue IDs for this principle
    users_with_queue = []
    for username, queues in user_queues.items():
        if args.principle in queues:
            users_with_queue.append(username)

    if not users_with_queue:
        print(f"Error: No users have queue IDs for principle '{args.principle}'")
        sys.exit(1)

    print(f"Found {len(users_with_queue)} user(s) with queues for '{args.principle}':")
    for username in users_with_queue:
        queue_id = user_queues[username][args.principle]
        print(f"  - {username}: {queue_id}")
    print()

    # Initialize Langfuse client
    try:
        langfuse = get_client()
        print("Connected to Langfuse successfully.\n")
    except Exception as e:
        print(f"Error connecting to Langfuse: {e}")
        sys.exit(1)

    # Determine which files to process
    if args.eval_files:
        eval_files = args.eval_files
        print(f"Processing {len(eval_files)} specified file(s)...\n")
    else:
        logs_dir = Path(__file__).parent.parent / "logs"
        if not logs_dir.exists():
            print(f"Error: logs directory not found at {logs_dir}")
            sys.exit(1)

        eval_files = list(logs_dir.glob("**/*.eval"))
        if not eval_files:
            print(f"No .eval files found in {logs_dir}")
            sys.exit(0)

        eval_files = [str(f) for f in eval_files]
        print(f"Processing all {len(eval_files)} .eval file(s) from logs/ (recursive)...\n")

    # Check if using reference user mode
    reference_sample_ids = None
    if args.reference_user:
        # Validate reference user exists in CSV
        if args.reference_user not in user_queues:
            print(f"Error: Reference user '{args.reference_user}' not found in queue CSV")
            sys.exit(1)

        # Validate reference user has queue for this principle
        if args.principle not in user_queues[args.reference_user]:
            print(f"Error: Reference user '{args.reference_user}' has no queue for principle '{args.principle}'")
            sys.exit(1)

        # Warn if both --reference-user and --sample-limit are provided
        if args.sample_limit:
            print(f"Warning: --sample-limit ignored when using --reference-user\n")

        # Get reference user's queue ID and query for sample IDs
        reference_queue_id = user_queues[args.reference_user][args.principle]
        print(f"Reference user mode: Using samples from '{args.reference_user}'")
        reference_sample_ids = get_sample_ids_from_queue(
            queue_id=reference_queue_id,
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )

        if not reference_sample_ids:
            print(f"Error: No samples found in reference user's queue")
            sys.exit(1)

        print()

    # Collect and sample data ONCE (same samples for all users)
    print(f"Collecting samples for principle '{args.principle}'...\n")
    samples_by_model = collect_samples_for_sampling(
        eval_files,
        args.principle,
        filter_sample_ids=reference_sample_ids
    )

    # Display pre-sampling statistics
    total_available = sum(len(samples) for samples in samples_by_model.values())
    print(f"Found {total_available} matching samples across {len(samples_by_model)} model(s):")
    for model, samples in sorted(samples_by_model.items()):
        print(f"  - {model}: {len(samples)} samples")
    print()

    # Sample if requested (only applies when NOT using reference user)
    if args.sample_limit and not args.reference_user:
        print(f"Sampling {args.sample_limit} samples evenly across models...\n")
        sampled = sample_evenly_across_models(samples_by_model, args.sample_limit)

        total_sampled = sum(len(samples) for samples in sampled.values())
        print(f"Sampled {total_sampled} samples for export:")
        for model, samples in sorted(sampled.items()):
            print(f"  - {model}: {len(samples)} samples")
        print()
    else:
        sampled = samples_by_model
        if not args.reference_user:
            print(f"No sampling limit specified, using all {total_available} samples\n")

    # Flatten sampled data into a list of (eval_file, sample) tuples
    eval_files_and_samples = []
    for samples in sampled.values():
        eval_files_and_samples.extend(samples)

    # Export to each user's queue
    print(f"Exporting to {len(users_with_queue)} user queue(s)...\n")
    total_samples_per_user = {}

    for username in users_with_queue:
        queue_id = user_queues[username][args.principle]
        print(f"Processing user: {username} (Queue: {queue_id})")

        samples_exported = export_samples_for_user(
            username=username,
            eval_files_and_samples=eval_files_and_samples,
            langfuse=langfuse,
            filter_principle=args.principle,
            enqueue=args.enqueue,
            queue_id=queue_id,
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )

        total_samples_per_user[username] = samples_exported
        print(f"  Exported {samples_exported} samples for {username}\n")

    # Ensure all data is sent to Langfuse
    langfuse.flush()

    # Summary
    print(f"{'='*60}")
    print(f"Multi-user export complete!")
    print(f"Principle: {args.principle}")
    if args.reference_user:
        print(f"Reference user: {args.reference_user} (using their exact samples)")
    elif args.sample_limit:
        print(f"Sample limit: {args.sample_limit} (evenly distributed across models)")
    print(f"Users processed: {len(users_with_queue)}")
    for username, count in total_samples_per_user.items():
        print(f"  - {username}: {count} samples")
    if args.enqueue:
        print(f"Items added to annotation queues: Yes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
