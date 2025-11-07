#!/usr/bin/env python3
"""
Import humane_bench.jsonl dataset into Langfuse.

This script reads the humane_bench.jsonl file and imports each item as a dataset
item in Langfuse, mapping the fields appropriately.

Requirements:
    - langfuse package installed: pip install langfuse
    - python-dotenv package (optional): pip install python-dotenv
    - Credentials configured either via:
        1. .env file in the project root (preferred), or
        2. Environment variables:
            - LANGFUSE_PUBLIC_KEY
            - LANGFUSE_SECRET_KEY
            - LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)

Usage:
    python scripts/jsonl_dataset_to_langfuse.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from langfuse import Langfuse
except ImportError:
    print("Error: langfuse package not installed.")
    print("Please install it with: pip install langfuse")
    sys.exit(1)

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        print("Note: No .env file found, using system environment variables")
except ImportError:
    print("Note: python-dotenv not installed, using system environment variables")
    print("      Install with: pip install python-dotenv")


def validate_environment() -> None:
    """Validate that required environment variables are set."""
    required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables before running the script.")
        sys.exit(1)


def import_dataset(
    jsonl_path: str = "data/humane_bench.jsonl",
    dataset_name: str = "humane_bench"
) -> None:
    """
    Import JSONL dataset into Langfuse.

    Args:
        jsonl_path: Path to the JSONL file
        dataset_name: Name of the dataset in Langfuse
    """
    # Validate environment
    validate_environment()

    # Initialize Langfuse client
    print("Initializing Langfuse client...")
    langfuse = Langfuse()

    # Verify the JSONL file exists
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)

    # Create or get the dataset
    print(f"Creating/accessing dataset: {dataset_name}")
    try:
        langfuse.create_dataset(name=dataset_name)
        print(f"✓ Dataset '{dataset_name}' created successfully")
    except Exception as e:
        # Dataset might already exist, which is fine
        print(f"Note: {e}")
        print(f"Continuing with existing dataset...")

    # Read and import each item
    print(f"\nImporting items from {jsonl_path}...")
    success_count = 0
    error_count = 0

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                item = json.loads(line.strip())

                # Extract fields
                item_id = item.get("id", f"item-{line_num}")
                input_data = item.get("input", "")
                target = item.get("target", "")
                metadata = item.get("metadata", {})

                # Combine id with metadata for complete metadata
                combined_metadata = {
                    "item_id": item_id,
                    **metadata
                }

                # Create dataset item in Langfuse
                langfuse.create_dataset_item(
                    dataset_name=dataset_name,
                    input=input_data,
                    expected_output=target,
                    metadata=combined_metadata
                )

                success_count += 1
                if success_count % 10 == 0:
                    print(f"  Imported {success_count} items...")

            except json.JSONDecodeError as e:
                print(f"✗ Error parsing line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"✗ Error importing line {line_num}: {e}")
                error_count += 1

    # Flush any pending operations
    langfuse.flush()

    # Print summary
    print(f"\n{'='*50}")
    print(f"Import Summary:")
    print(f"  Total items imported: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Dataset name: {dataset_name}")
    print(f"{'='*50}")

    if error_count > 0:
        print(f"\n⚠ Warning: {error_count} items failed to import")
        sys.exit(1)
    else:
        print(f"\n✓ All items imported successfully!")


if __name__ == "__main__":
    import_dataset()
