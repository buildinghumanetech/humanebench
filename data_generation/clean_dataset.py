#!/usr/bin/env python3
"""
Dataset cleanup script for humane_bench.jsonl
Removes exact duplicates, fixes duplicate IDs, and removes non-standard domains.
"""

import json
from pathlib import Path
from collections import defaultdict

# Define standard domains
STANDARD_DOMAINS = {
    "relationships",
    "mental-health",
    "financial-decisions",
    "career-guidance",
    "parenting",
    "health-wellness",
    "technology-use",
    "politics-society",
    "education",
    "moral-ambiguity",
    "crisis-situations",
    "everyday-decisions"
}

def load_jsonl(filepath):
    """Load JSONL file into a list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, filepath):
    """Save list of dictionaries to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def remove_exact_duplicates(data):
    """Remove scenarios with identical input text."""
    seen_inputs = {}
    unique_data = []
    duplicates_removed = 0

    for item in data:
        input_text = item['input']
        if input_text in seen_inputs:
            print(f"  Removing exact duplicate: {item['id']} (duplicate of {seen_inputs[input_text]})")
            duplicates_removed += 1
        else:
            seen_inputs[input_text] = item['id']
            unique_data.append(item)

    print(f"Removed {duplicates_removed} exact duplicate(s)")
    return unique_data

def fix_duplicate_ids(data):
    """Fix duplicate IDs by renumbering the second occurrence."""
    seen_ids = {}
    principle_counters = defaultdict(int)
    fixed_data = []
    fixes_made = 0

    # First pass: collect all existing IDs and find max numbers per principle
    for item in data:
        item_id = item['id']
        principle = item['metadata']['principle']

        # Extract the number from the ID
        if '-' in item_id:
            parts = item_id.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit():
                num = int(parts[1])
                principle_counters[principle] = max(principle_counters[principle], num)

    # Second pass: fix duplicates
    seen_ids = {}
    for item in data:
        item_id = item['id']
        principle = item['metadata']['principle']

        if item_id in seen_ids:
            # This is a duplicate ID - generate a new one
            principle_counters[principle] += 1
            new_id = f"{principle}-{principle_counters[principle]:03d}"
            print(f"  Renumbering duplicate ID: {item_id} → {new_id}")
            print(f"    Input: {item['input'][:80]}...")
            item['id'] = new_id
            fixes_made += 1

        seen_ids[item_id] = True
        fixed_data.append(item)

    print(f"Fixed {fixes_made} duplicate ID(s)")
    return fixed_data

def remove_nonstandard_domains(data):
    """Remove scenarios with non-standard domains."""
    clean_data = []
    removed = 0

    for item in data:
        domain = item['metadata']['domain']
        if domain in STANDARD_DOMAINS:
            clean_data.append(item)
        else:
            print(f"  Removing non-standard domain '{domain}': {item['id']}")
            removed += 1

    print(f"Removed {removed} scenario(s) with non-standard domains")
    return clean_data

def main():
    # File paths
    input_file = Path(__file__).parent.parent / "data" / "humane_bench.jsonl"
    output_file = input_file  # Overwrite the original file

    print(f"Loading dataset from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} scenarios\n")

    # Step 1: Remove exact duplicates
    print("Step 1: Removing exact duplicates...")
    data = remove_exact_duplicates(data)
    print(f"Dataset now has {len(data)} scenarios\n")

    # Step 2: Fix duplicate IDs
    print("Step 2: Fixing duplicate IDs...")
    data = fix_duplicate_ids(data)
    print(f"Dataset still has {len(data)} scenarios\n")

    # Step 3: Remove non-standard domains (safety check)
    print("Step 3: Removing any remaining non-standard domains...")
    data = remove_nonstandard_domains(data)
    print(f"Dataset now has {len(data)} scenarios\n")

    # Save the cleaned dataset
    print(f"Saving cleaned dataset to {output_file}...")
    save_jsonl(data, output_file)
    print("Done!")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Final dataset size: {len(data)} scenarios")
    print(f"Cleaned dataset saved to: {output_file}")

if __name__ == "__main__":
    main()
