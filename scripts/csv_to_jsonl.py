#!/usr/bin/env python3
"""
Convert humane_bench CSV back to JSONL format.
"""

import json
import csv

def convert_csv_to_jsonl(input_file, output_file):
    """
    Convert CSV file to JSONL, reconstructing the metadata structure.

    Args:
        input_file: Path to the CSV input file
        output_file: Path to the JSONL output file
    """
    records_processed = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)

        for row in reader:
            # Reconstruct the nested structure
            record = {
                'id': row['id'],
                'input': row['input'],
                'target': row['target'],
                'metadata': {
                    'principle': row['principle'],
                    'domain': row['domain'],
                    'vulnerable-population': row['vulnerable-population']
                }
            }

            # Write as JSON line
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            records_processed += 1

    print(f"✓ Successfully converted {records_processed} records")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

if __name__ == '__main__':
    input_path = 'data/humane_bench.csv'
    output_path = 'data/humane_bench.jsonl'

    convert_csv_to_jsonl(input_path, output_path)
