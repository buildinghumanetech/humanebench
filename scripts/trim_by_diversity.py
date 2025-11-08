#!/usr/bin/env python3
"""
Trim humane_bench.jsonl by semantic diversity.

For all principles with over 100 entries:
1. Calculate semantic similarity using text-embedding-3-large
2. Select the 100 most diverse entries using greedy diversity selection
3. Renumber IDs sequentially (001-100)

Usage:
    python trim_by_diversity.py
"""

import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
INPUT_FILE = "data/humane_bench.jsonl"
OUTPUT_FILE = "data/humane_bench.jsonl"
BACKUP_FILE = "data/humane_bench.jsonl.backup"
TARGET_COUNT = 100
EMBEDDING_MODEL = "text-embedding-3-large"


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def save_jsonl(filepath: str, entries: List[Dict[str, Any]]) -> None:
    """Save list of dictionaries to JSONL file."""
    with open(filepath, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def get_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    print(f"  Getting embeddings for {len(texts)} texts...")

    # OpenAI API has a limit, so batch if needed
    batch_size = 2048
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return np.array(all_embeddings)


def greedy_diversity_selection(embeddings: np.ndarray, n: int, seed: int = 42) -> List[int]:
    """
    Select n most diverse items using greedy selection.

    Algorithm:
    1. Start with a random item
    2. Iteratively select the item that maximizes minimum distance to already selected items

    Returns:
        List of indices of selected items
    """
    np.random.seed(seed)
    num_items = len(embeddings)

    if n >= num_items:
        return list(range(num_items))

    # Start with a random item
    selected_indices = [np.random.randint(num_items)]

    print(f"  Selecting {n} diverse items from {num_items}...")

    for step in range(1, n):
        if step % 10 == 0:
            print(f"  Selected {step}/{n} items")

        # Calculate similarity of all items to selected items
        selected_embeddings = embeddings[selected_indices]
        similarities = cosine_similarity(embeddings, selected_embeddings)

        # For each candidate, find its maximum similarity to any selected item
        # (we want to minimize this to maximize diversity)
        max_similarities = similarities.max(axis=1)

        # Exclude already selected items
        max_similarities[selected_indices] = np.inf

        # Select the item with minimum max similarity (most diverse)
        next_idx = max_similarities.argmin()
        selected_indices.append(next_idx)

    print(f"  Completed diversity selection")
    return selected_indices


def renumber_ids(entries: List[Dict[str, Any]], principle: str) -> List[Dict[str, Any]]:
    """Renumber IDs sequentially for a principle."""
    renumbered = []
    for i, entry in enumerate(entries, start=1):
        new_entry = entry.copy()
        new_entry['id'] = f"{principle}-{i:03d}"
        renumbered.append(new_entry)
    return renumbered


def main():
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Loading {INPUT_FILE}...")
    entries = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(entries)} entries")

    # Group by principle
    by_principle = defaultdict(list)
    for entry in entries:
        principle = entry['metadata']['principle']
        by_principle[principle].append(entry)

    print(f"\nFound {len(by_principle)} unique principles")
    for principle, items in sorted(by_principle.items()):
        print(f"  {principle}: {len(items)} entries")

    # Process each principle
    trimmed_entries = []

    for principle, items in sorted(by_principle.items()):
        print(f"\n{'='*60}")
        print(f"Processing: {principle} ({len(items)} entries)")
        print(f"{'='*60}")

        if len(items) <= TARGET_COUNT:
            print(f"  Principle has ≤{TARGET_COUNT} entries, keeping all and renumbering")
            renumbered = renumber_ids(items, principle)
            trimmed_entries.extend(renumbered)
        else:
            print(f"  Principle has {len(items)} entries, trimming to {TARGET_COUNT}")

            # Get embeddings
            input_texts = [item['input'] for item in items]
            embeddings = get_embeddings(input_texts, client)

            # Select diverse subset
            selected_indices = greedy_diversity_selection(embeddings, TARGET_COUNT)
            selected_items = [items[i] for i in selected_indices]

            print(f"  Trimmed from {len(items)} to {len(selected_items)} entries")

            # Renumber IDs
            renumbered = renumber_ids(selected_items, principle)
            trimmed_entries.extend(renumbered)

    # Create backup
    print(f"\n{'='*60}")
    print(f"Creating backup: {BACKUP_FILE}")
    import shutil
    shutil.copy2(INPUT_FILE, BACKUP_FILE)

    # Save trimmed data
    print(f"Saving trimmed data to: {OUTPUT_FILE}")
    save_jsonl(OUTPUT_FILE, trimmed_entries)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Original entries: {len(entries)}")
    print(f"Trimmed entries: {len(trimmed_entries)}")
    print(f"Removed: {len(entries) - len(trimmed_entries)}")

    # Verify counts per principle
    final_counts = defaultdict(int)
    for entry in trimmed_entries:
        final_counts[entry['metadata']['principle']] += 1

    print("\nFinal counts per principle:")
    for principle, count in sorted(final_counts.items()):
        print(f"  {principle}: {count}")

    print(f"\nBackup saved to: {BACKUP_FILE}")
    print(f"Trimmed data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
