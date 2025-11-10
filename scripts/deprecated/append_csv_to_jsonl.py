#!/usr/bin/env python3
"""
Script to append enhance-human-capabilities.csv to data/humane_bench.jsonl
Converts CSV format to JSONL format with metadata structure.
"""

import csv
import json

def append_csv_to_jsonl():
    csv_file = "enhance-human-capabilities.csv"
    jsonl_file = "data/humane_bench.jsonl"

    # Read CSV and append to JSONL
    with open(csv_file, 'r', encoding='utf-8') as csv_f, \
         open(jsonl_file, 'a', encoding='utf-8') as jsonl_f:

        reader = csv.DictReader(csv_f)

        for row in reader:
            # Convert CSV row to JSONL format
            jsonl_entry = {
                "id": row["id"],
                "input": row["input"],
                "target": row["target"],
                "metadata": {
                    "principle": row["principle"],
                    "domain": row["domain"],
                    "vulnerable-population": row["vulnerable-population"]
                }
            }

            # Write as single line JSON
            jsonl_f.write(json.dumps(jsonl_entry) + '\n')

    print(f"Successfully appended {csv_file} to {jsonl_file}")

if __name__ == "__main__":
    append_csv_to_jsonl()
