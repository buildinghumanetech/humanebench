#!/usr/bin/env python3
"""
Extract AI outputs from Inspect AI evaluation logs for human rating.
Creates three CSV files with outputs from 2 models x 3 personas.
"""

import json
import csv
import zipfile
import random
from pathlib import Path
from collections import defaultdict

def extract_model_short_name(model_full_name):
    """
    Extract short model name from full path.
    'openai/gpt-4o' -> 'gpt-4o'
    'google/gemini-2.5-flash' -> 'gemini-2.5-flash'
    """
    if '/' in model_full_name:
        return model_full_name.split('/')[-1]
    return model_full_name

def extract_persona_from_path(eval_file_path):
    """
    Extract persona from file path.
    logs/baseline/... -> 'baseline'
    logs/good_persona/... -> 'good'
    logs/bad_persona/... -> 'bad'
    """
    parts = Path(eval_file_path).parts
    if 'baseline' in parts:
        return 'baseline'
    elif 'good_persona' in parts:
        return 'good'
    elif 'bad_persona' in parts:
        return 'bad'
    return 'unknown'

def extract_samples_from_eval(eval_file_path):
    """
    Extract all samples from a .eval file (ZIP archive).
    Returns list of sample data dictionaries.
    """
    samples = []
    persona = extract_persona_from_path(eval_file_path)

    with zipfile.ZipFile(eval_file_path, 'r') as zf:
        # List all sample JSON files
        sample_files = [f for f in zf.namelist() if f.startswith('samples/') and f.endswith('.json')]

        for sample_file in sample_files:
            with zf.open(sample_file) as f:
                sample_data = json.load(f)

                # Extract required fields
                sample_id = sample_data.get('id', '')
                sample_text = sample_data.get('input', '')

                # AI output is the last message (assistant role)
                messages = sample_data.get('messages', [])
                ai_output = ''
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        ai_output = msg.get('content', '')
                        break

                # Model name (short form)
                model_full = sample_data.get('output', {}).get('model', '')
                ai_model = extract_model_short_name(model_full)

                samples.append({
                    'sample_id': sample_id,
                    'sample_text': sample_text,
                    'ai_output': ai_output,
                    'ai_model': ai_model,
                    'ai_persona': persona,
                    'score': '',
                    'reasoning': '',
                    'misc_comments': ''
                })

    return samples

def collect_all_samples(logs_dir):
    """
    Collect samples from all .eval files in logs directory.
    Returns list of all samples.
    """
    all_samples = []
    logs_path = Path(logs_dir)

    # Find all .eval files
    eval_files = list(logs_path.rglob('*.eval'))

    print(f"Found {len(eval_files)} evaluation files:")
    for eval_file in eval_files:
        print(f"  {eval_file.relative_to(logs_path)}")
        samples = extract_samples_from_eval(eval_file)
        all_samples.extend(samples)
        print(f"    Extracted {len(samples)} samples")

    return all_samples

def group_by_sample_id(samples):
    """
    Group samples by sample_id.
    Returns dict: {sample_id: [sample_data, ...]}
    """
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['sample_id']].append(sample)
    return grouped

def split_into_sets(grouped_samples):
    """
    Split grouped samples into 3 sets of 8 sample_ids each.
    Randomizes the order of rows within each sample_id group.
    Returns three lists of samples.
    """
    # Sort sample_ids alphabetically
    sorted_ids = sorted(grouped_samples.keys())

    # Split into 3 groups of 8
    set1_ids = sorted_ids[0:8]
    set2_ids = sorted_ids[8:16]
    set3_ids = sorted_ids[16:24]

    # Collect samples for each set
    set1_samples = []
    set2_samples = []
    set3_samples = []

    for sample_id in set1_ids:
        # Randomize the order of the 6 variations for this sample_id
        sample_group = grouped_samples[sample_id].copy()
        random.shuffle(sample_group)
        set1_samples.extend(sample_group)

    for sample_id in set2_ids:
        # Randomize the order of the 6 variations for this sample_id
        sample_group = grouped_samples[sample_id].copy()
        random.shuffle(sample_group)
        set2_samples.extend(sample_group)

    for sample_id in set3_ids:
        # Randomize the order of the 6 variations for this sample_id
        sample_group = grouped_samples[sample_id].copy()
        random.shuffle(sample_group)
        set3_samples.extend(sample_group)

    return set1_samples, set2_samples, set3_samples

def write_csv(samples, output_path):
    """
    Write samples to CSV file.
    """
    fieldnames = [
        'sample_id',
        'sample_text',
        'ai_output',
        'ai_model',
        'ai_persona',
        'score',
        'reasoning',
        'misc_comments'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(samples)

def main():
    logs_dir = 'logs'
    output_files = [
        'human_ratings_set1.csv',
        'human_ratings_set2.csv',
        'human_ratings_set3.csv'
    ]

    print("=" * 60)
    print("Extracting AI outputs for human rating")
    print("=" * 60)
    print()

    # Collect all samples
    print("Step 1: Collecting samples from evaluation logs...")
    all_samples = collect_all_samples(logs_dir)
    print(f"  Total samples collected: {len(all_samples)}")
    print()

    # Group by sample_id
    print("Step 2: Grouping by sample_id...")
    grouped = group_by_sample_id(all_samples)
    print(f"  Unique sample_ids: {len(grouped)}")

    # Verify each sample_id has 6 variations (2 models x 3 personas)
    for sample_id, samples in grouped.items():
        if len(samples) != 6:
            print(f"  WARNING: {sample_id} has {len(samples)} variations (expected 6)")
    print()

    # Split into 3 sets
    print("Step 3: Splitting into 3 sets...")
    set1, set2, set3 = split_into_sets(grouped)
    print(f"  Set 1: {len(set1)} samples")
    print(f"  Set 2: {len(set2)} samples")
    print(f"  Set 3: {len(set3)} samples")
    print()

    # Write CSV files
    print("Step 4: Writing CSV files...")
    for samples, output_file in zip([set1, set2, set3], output_files):
        write_csv(samples, output_file)
        print(f"  ✓ Wrote {len(samples)} rows to {output_file}")

    print()
    print("=" * 60)
    print("✓ Successfully created 3 CSV files for human rating")
    print("=" * 60)

if __name__ == '__main__':
    main()
