#!/usr/bin/env python3
"""
Filter humane_bench.jsonl to only include IDs from GQ_models.txt
and write to humane_bench_test.jsonl
"""

import json

def read_ids_from_file(file_path):
    """Read IDs from GQ_models.txt."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def filter_jsonl(input_file, output_file, id_filter):
    """
    Filter JSONL file to only include records with IDs in id_filter.
    """
    records_written = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            record = json.loads(line.strip())
            if record['id'] in id_filter:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                records_written += 1

    return records_written

def main():
    gq_file = 'GQ_models.txt'
    input_file = 'data/humane_bench.jsonl'
    output_file = 'data/humane_bench_test.jsonl'

    print(f"Reading IDs from {gq_file}...")
    ids = read_ids_from_file(gq_file)
    print(f"  Found {len(ids)} IDs")

    print(f"\nFiltering {input_file}...")
    count = filter_jsonl(input_file, output_file, ids)

    print(f"\n✓ Successfully wrote {count} records to {output_file}")

if __name__ == '__main__':
    main()
