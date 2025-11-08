#!/usr/bin/env python3
"""
Script to replace all enhance-human-capabilities entries in data/humane_bench.jsonl
with new data from enhance-human-capabilities.csv
"""

import json
import csv
import shutil
from pathlib import Path


def read_csv_data(csv_path):
    """Read CSV and convert to JSONL format entries."""
    entries = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create JSONL entry in the exact format
            entry = {
                "id": row['id'],
                "input": row['input'],
                "target": row['target'],
                "metadata": {
                    "principle": row['principle'],
                    "domain": row['domain'],
                    "vulnerable-population": row['vulnerable-population']
                }
            }
            entries.append(entry)

    return entries


def process_jsonl(jsonl_path, new_entries):
    """
    Process JSONL file:
    - Replace entries 1-70 in their original positions
    - Append entries 71-133 at the end
    """
    # Create a lookup dictionary for new entries by ID
    new_entries_dict = {entry['id']: entry for entry in new_entries}

    # Read all existing entries and replace old enhance-human-capabilities entries
    all_entries = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())

            # If this is a enhance-human-capabilities entry, replace with new version
            if entry['target'] == 'enhance-human-capabilities':
                entry_id = entry['id']
                if entry_id in new_entries_dict:
                    all_entries.append(new_entries_dict[entry_id])
                else:
                    # This shouldn't happen, but keep old entry if no match
                    all_entries.append(entry)
            else:
                # Keep all other entries as-is
                all_entries.append(entry)

    # Add entries 71-133 at the end
    for i in range(0, 151):
        entry_id = f'enhance-human-capabilities-{i:03d}'
        if entry_id in new_entries_dict:
            all_entries.append(new_entries_dict[entry_id])

    return all_entries


def write_jsonl(entries, output_path):
    """Write entries to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def main():
    csv_path = Path('enhance-human-capabilities.csv')
    jsonl_path = Path('data/humane_bench.jsonl')
    backup_path = Path('data/humane_bench.jsonl.backup')

    print(f"Reading CSV data from {csv_path}...")
    new_entries = read_csv_data(csv_path)
    print(f"  Loaded {len(new_entries)} new entries")

    print(f"\nCreating backup of {jsonl_path} to {backup_path}...")
    shutil.copy2(jsonl_path, backup_path)
    print("  Backup created")

    print(f"\nProcessing {jsonl_path}...")
    all_entries = process_jsonl(jsonl_path, new_entries)
    print(f"  Total entries after processing: {len(all_entries)}")

    print(f"\nWriting updated data to {jsonl_path}...")
    write_jsonl(all_entries, jsonl_path)
    print("  Done!")

    # Verify counts
    rua_count = sum(1 for e in all_entries if e['target'] == 'enhance-human-capabilities')
    print(f"\nVerification:")
    print(f"  Total entries: {len(all_entries)}")
    print(f"  enhance-human-capabilities entries: {rua_count}")
    print(f"  Other entries: {len(all_entries) - rua_count}")


if __name__ == '__main__':
    main()
