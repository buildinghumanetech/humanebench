#!/usr/bin/env python3
"""
Export filtered CSVs from humane_bench.jsonl based on IDs in GQ_models.txt.
Splits the 24 IDs (3 per principle) into three separate CSV files.
"""

import json
import csv
from collections import defaultdict

def read_ids_from_file(file_path):
    """Read IDs from GQ_models.txt."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def group_ids_by_principle(ids):
    """
    Group IDs by principle in the order they appear.
    Returns a dict: {principle: [id1, id2, id3, ...]}
    """
    principle_ids = defaultdict(list)

    for id_str in ids:
        # Extract principle from ID (everything before the last hyphen and number)
        # e.g., "respect-user-attention-012" -> "respect-user-attention"
        parts = id_str.rsplit('-', 1)
        if len(parts) == 2:
            principle = parts[0]
            principle_ids[principle].append(id_str)

    return principle_ids

def distribute_ids_to_sets(principle_ids):
    """
    Distribute IDs across 3 sets: 1st ID per principle, 2nd ID, 3rd ID.
    Returns three lists of IDs.
    """
    set1_ids = []
    set2_ids = []
    set3_ids = []

    # Sort principles for consistent ordering
    for principle in sorted(principle_ids.keys()):
        ids = principle_ids[principle]
        if len(ids) >= 1:
            set1_ids.append(ids[0])
        if len(ids) >= 2:
            set2_ids.append(ids[1])
        if len(ids) >= 3:
            set3_ids.append(ids[2])

    return set1_ids, set2_ids, set3_ids

def load_jsonl_records(jsonl_path, id_filter):
    """
    Load records from JSONL file that match the given ID filter.
    Returns a list of flattened records ready for CSV export.
    """
    records = []
    id_set = set(id_filter)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            if record['id'] in id_set:
                # Flatten the structure for CSV
                flat_record = {
                    'id': record['id'],
                    'input': record['input'],
                    'target': record['target'],
                    'principle': record['metadata']['principle'],
                    'domain': record['metadata']['domain'],
                    'vulnerable-population': record['metadata']['vulnerable-population']
                }
                records.append(flat_record)

    return records

def write_csv(records, output_path):
    """Write records to CSV file."""
    fieldnames = ['id', 'input', 'target', 'principle', 'domain', 'vulnerable-population']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(records)

def main():
    # Configuration
    gq_file = 'GQ_models.txt'
    jsonl_file = 'data/humane_bench.jsonl'
    output_files = [
        'data/humane_bench_set1.csv',
        'data/humane_bench_set2.csv',
        'data/humane_bench_set3.csv'
    ]

    # Read and process IDs
    print(f"Reading IDs from {gq_file}...")
    all_ids = read_ids_from_file(gq_file)
    print(f"  Found {len(all_ids)} IDs")

    # Group by principle
    principle_ids = group_ids_by_principle(all_ids)
    print(f"  Grouped into {len(principle_ids)} principles")

    # Distribute to 3 sets
    set1_ids, set2_ids, set3_ids = distribute_ids_to_sets(principle_ids)
    print(f"\nDistribution:")
    print(f"  Set 1: {len(set1_ids)} IDs (1st per principle)")
    print(f"  Set 2: {len(set2_ids)} IDs (2nd per principle)")
    print(f"  Set 3: {len(set3_ids)} IDs (3rd per principle)")

    # Process each set
    id_sets = [set1_ids, set2_ids, set3_ids]

    for i, (ids, output_path) in enumerate(zip(id_sets, output_files), 1):
        print(f"\nProcessing Set {i}...")
        records = load_jsonl_records(jsonl_file, ids)
        write_csv(records, output_path)
        print(f"  ✓ Wrote {len(records)} records to {output_path}")

    print(f"\n✓ Successfully created {len(output_files)} CSV files")

if __name__ == '__main__':
    main()
