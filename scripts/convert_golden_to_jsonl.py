"""
Convert golden questions CSV to Inspect AI JSONL format.

This script converts the golden_questions.csv file (which contains pre-generated
AI responses and human ratings) into a JSONL format that can be consumed by
Inspect AI for scoring validation.
"""

import pandas as pd
import json
from pathlib import Path


def convert_golden_questions_to_jsonl(
    input_csv: Path,
    output_jsonl: Path
):
    """
    Convert golden questions CSV to Inspect AI JSONL format.

    Args:
        input_csv: Path to golden_questions.csv
        output_jsonl: Path where JSONL should be saved
    """
    # Read CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} golden questions from {input_csv}")

    # Validate required columns
    required_columns = [
        'observation_id', 'input_text', 'principle', 'input_id',
        'ai_model', 'ai_persona', 'ai_output', 'num_raters',
        'rater_names', 'ratings', 'rating_values', 'all_positive'
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Convert each row to JSONL format
    samples = []

    for idx, row in df.iterrows():
        # Create sample in Inspect AI format
        sample = {
            "id": row['observation_id'],
            "input": row['input_text'],
            "target": row['principle'],
            "metadata": {
                "principle": row['principle'],
                "domain": row.get('domain', ''),
                "vulnerable_population": row.get('vulnerable_population', ''),
                "input_id": row['input_id'],
                "ai_model": row['ai_model'],
                "ai_persona": row['ai_persona'],
                "ai_output": row['ai_output'],  # Pre-generated response
                "human_num_raters": int(row['num_raters']) if pd.notna(row['num_raters']) else 0,
                "human_rater_names": row['rater_names'],
                "human_ratings": row['ratings'],
                "human_rating_values": row['rating_values'],
                "human_all_positive": (
                    row['all_positive']
                    if isinstance(row['all_positive'], bool)
                    else str(row['all_positive']).lower() in ('true', '1', 'yes')
                ),
                "is_golden_question": True
            }
        }

        samples.append(sample)

    # Write to JSONL
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"\nConverted {len(samples)} samples to {output_jsonl}")

    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")

    df_summary = df.copy()

    print(f"\nSamples by model:")
    for model, count in df_summary['ai_model'].value_counts().items():
        print(f"  {model}: {count}")

    print(f"\nSamples by persona:")
    for persona, count in df_summary['ai_persona'].value_counts().items():
        print(f"  {persona}: {count}")

    print(f"\nSamples by principle (top 5):")
    for principle, count in df_summary['principle'].value_counts().head(5).items():
        print(f"  {principle}: {count}")

    print(f"\nSamples by number of human raters:")
    for n_raters, count in df_summary['num_raters'].value_counts().sort_index().items():
        print(f"  {n_raters} raters: {count}")

    print(f"\nHuman rating direction:")
    positive = df_summary['all_positive'].sum()
    negative = len(df_summary) - positive
    print(f"  All positive: {positive}")
    print(f"  All negative: {negative}")

    print(f"\n{'='*60}")

    # Print example
    print("\nExample sample (first entry):")
    print(json.dumps(samples[0], indent=2))


def main():
    """Main entry point."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    input_csv = project_root / 'scripts' / 'human_judging_analysis' / 'output' / 'golden_questions.csv'
    output_jsonl = project_root / 'data' / 'golden_questions.jsonl'

    # Convert
    convert_golden_questions_to_jsonl(input_csv, output_jsonl)


if __name__ == '__main__':
    main()
