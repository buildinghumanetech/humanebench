#!/usr/bin/env python3
import json
import csv
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python convert_jsonl_to_csv.py <input_jsonl_file>")
    sys.exit(1)

input_file = sys.argv[1]

# Generate output filename by replacing .jsonl with .csv
if input_file.endswith('.jsonl'):
    output_file = input_file[:-6] + '.csv'
else:
    output_file = input_file + '.csv'

# Read all records from JSONL
records = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            # Flatten metadata
            flattened = {
                'id': record['id'],
                'input': record['input'],
                'target': record['target'],
                'meta_principle': record['metadata']['principle'],
                'meta_domain': record['metadata']['domain'],
                'meta_vulnerable_population': record['metadata']['vulnerable-population']
            }
            records.append(flattened)

# Write to CSV
if records:
    fieldnames = ['id', 'input', 'target', 'meta_principle', 'meta_domain', 'meta_vulnerable_population']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Converted {len(records)} records from {input_file} to {output_file}")
else:
    print("No records found!")
