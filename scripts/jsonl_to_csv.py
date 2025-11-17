#!/usr/bin/env python3
"""
Convert humane_bench.jsonl to CSV format for Google Sheets upload.
"""

import json
import csv

def convert_jsonl_to_csv(input_file, output_file):
    """
    Convert JSONL file to CSV, flattening the metadata structure.

    Args:
        input_file: Path to the JSONL input file
        output_file: Path to the CSV output file
    """
    # Define CSV columns in desired order
    fieldnames = ['id', 'input', 'target', 'principle', 'domain', 'vulnerable-population']

    records_processed = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for line in infile:
            # Parse JSON line
            record = json.loads(line.strip())

            # Flatten the structure
            flat_record = {
                'id': record['id'],
                'input': record['input'],
                'target': record['target'],
                'principle': record['metadata']['principle'],
                'domain': record['metadata']['domain'],
                'vulnerable-population': record['metadata']['vulnerable-population']
            }

            writer.writerow(flat_record)
            records_processed += 1

    print(f"✓ Successfully converted {records_processed} records")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

if __name__ == '__main__':
    input_path = 'data/humane_bench.jsonl'
    output_path = 'data/humane_bench.csv'

    convert_jsonl_to_csv(input_path, output_path)
