"""
Extract golden questions from human ratings.

Golden questions are defined as observations where all available raters agree on the
direction (positive/negative sign), regardless of the exact magnitude.

Criteria:
- All raters must have ratings with the same sign (all positive OR all negative)
- Works for observations with 2, 3, or 4 raters
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any


def load_dataset_metadata(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata from the main dataset JSONL file.

    Args:
        dataset_path: Path to humane_bench.jsonl

    Returns:
        Dictionary mapping input_id to metadata
    """
    metadata_dict = {}

    with open(dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            input_id = item.get('id')

            if input_id:
                metadata_dict[input_id] = {
                    'principle': item.get('target', ''),
                    'domain': item.get('metadata', {}).get('domain', ''),
                    'vulnerable_population': item.get('metadata', {}).get('vulnerable-population', ''),
                    'input_text': item.get('input', '')
                }

    return metadata_dict


def check_sign_agreement(ratings: List[float]) -> tuple[bool, bool]:
    """
    Check if all ratings have the same sign (all positive or all negative).

    Args:
        ratings: List of numeric rating values

    Returns:
        Tuple of (is_golden, all_positive)
        - is_golden: True if all ratings have the same sign
        - all_positive: True if all ratings are positive/zero, False if all negative
    """
    if not ratings:
        return False, False

    # Check if all ratings are non-negative (positive or zero)
    all_positive = all(r >= 0 for r in ratings)

    # Check if all ratings are non-positive (negative or zero)
    all_negative = all(r <= 0 for r in ratings)

    # It's golden if all have same sign
    is_golden = all_positive or all_negative

    return is_golden, all_positive


def extract_golden_questions(
    consolidated_path: Path,
    dataset_path: Path,
    output_path: Path
):
    """
    Extract golden questions from consolidated ratings.

    Args:
        consolidated_path: Path to consolidated_human_ratings.csv
        dataset_path: Path to humane_bench.jsonl
        output_path: Path where golden questions CSV should be saved
    """
    # Load consolidated ratings
    ratings_df = pd.read_csv(consolidated_path)
    print(f"Loaded {len(ratings_df)} ratings from {consolidated_path}")

    # Load dataset metadata
    print(f"Loading dataset metadata from {dataset_path}...")
    metadata_dict = load_dataset_metadata(dataset_path)
    print(f"Loaded metadata for {len(metadata_dict)} samples")

    # Group by observation_id
    golden_questions = []

    for obs_id, group in ratings_df.groupby('observation_id'):
        # Get ratings for this observation
        ratings = group['rating_numeric'].tolist()
        rater_names = group['rater_name'].tolist()
        rating_strings = group['rating'].tolist()

        # Check if this is a golden question
        is_golden, all_positive = check_sign_agreement(ratings)

        if is_golden:
            # Get observation details
            first_row = group.iloc[0]
            input_id = first_row['input_id']

            # Get metadata from dataset
            metadata = metadata_dict.get(input_id, {})

            # Calculate consensus metrics
            consensus_score = np.mean(ratings)
            rating_std = np.std(ratings, ddof=1) if len(ratings) > 1 else 0.0
            rating_range = max(ratings) - min(ratings)

            # Categorize agreement strength
            if rating_std == 0:
                agreement_strength = "perfect"
            elif rating_range <= 0.5:
                agreement_strength = "strong"
            elif rating_range <= 1.0:
                agreement_strength = "moderate"
            else:
                agreement_strength = "low"

            # Create golden question entry
            golden_entry = {
                'observation_id': obs_id,
                'input_id': input_id,
                'ai_model': first_row['ai_model'],
                'ai_persona': first_row['ai_persona'],
                'num_raters': len(ratings),
                'rater_names': ', '.join(rater_names),
                'ratings': ', '.join(rating_strings),
                'rating_values': ', '.join([f"{r:+.1f}" for r in ratings]),
                'consensus_score': consensus_score,
                'rating_std': rating_std,
                'rating_range': rating_range,
                'agreement_strength': agreement_strength,
                'all_positive': all_positive,
                'principle': metadata.get('principle', first_row.get('input_id', '').rsplit('-', 1)[0]),
                'domain': metadata.get('domain', ''),
                'vulnerable_population': metadata.get('vulnerable_population', ''),
                'input_text': metadata.get('input_text', first_row.get('input_text', '')),
                'ai_output': first_row.get('ai_output', ''),
            }

            golden_questions.append(golden_entry)

    # Create DataFrame
    golden_df = pd.DataFrame(golden_questions)

    # Sort by principle, then by input_id
    golden_df = golden_df.sort_values(['principle', 'input_id', 'ai_model', 'ai_persona']).reset_index(drop=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    golden_df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("GOLDEN QUESTIONS SUMMARY")
    print(f"{'='*60}")
    print(f"Total observations evaluated: {ratings_df['observation_id'].nunique()}")
    print(f"Golden questions identified: {len(golden_df)}")
    print(f"Percentage golden: {len(golden_df) / ratings_df['observation_id'].nunique() * 100:.1f}%")

    print(f"\nGolden questions by number of raters:")
    for n_raters in sorted(golden_df['num_raters'].unique()):
        count = (golden_df['num_raters'] == n_raters).sum()
        print(f"  {n_raters} raters: {count} questions")

    print(f"\nGolden questions by direction:")
    positive_count = golden_df['all_positive'].sum()
    negative_count = len(golden_df) - positive_count
    print(f"  All positive: {positive_count} questions")
    print(f"  All negative: {negative_count} questions")

    print(f"\nGolden questions by persona:")
    for persona, count in golden_df['ai_persona'].value_counts().items():
        print(f"  {persona}: {count} questions")

    print(f"\nGolden questions by principle (top 10):")
    principle_counts = golden_df['principle'].value_counts().head(10)
    for principle, count in principle_counts.items():
        print(f"  {principle}: {count}")

    # Consensus metrics summary
    print(f"\nConsensus Score Distribution:")
    print(golden_df['consensus_score'].describe().to_string())

    print(f"\nAgreement Strength Breakdown:")
    for strength, count in golden_df['agreement_strength'].value_counts().items():
        print(f"  {strength}: {count} questions")

    print(f"\nMean Consensus Score by Direction:")
    positive_df = golden_df[golden_df['all_positive'] == True]
    negative_df = golden_df[golden_df['all_positive'] == False]
    if len(positive_df) > 0:
        print(f"  Positive: {positive_df['consensus_score'].mean():.3f} (median: {positive_df['consensus_score'].median():.3f})")
    if len(negative_df) > 0:
        print(f"  Negative: {negative_df['consensus_score'].mean():.3f} (median: {negative_df['consensus_score'].median():.3f})")

    print(f"\nOutput saved to: {output_path}")
    print(f"{'='*60}")

    # Also print a few examples
    print("\nExample golden questions:")
    print("-" * 60)
    for idx, row in golden_df.head(3).iterrows():
        print(f"\nObservation {idx + 1}:")
        print(f"  Input ID: {row['input_id']}")
        print(f"  Model: {row['ai_model']}, Persona: {row['ai_persona']}")
        print(f"  Raters ({row['num_raters']}): {row['rater_names']}")
        print(f"  Ratings: {row['ratings']}")
        print(f"  Direction: {'Positive' if row['all_positive'] else 'Negative'}")
        print(f"  Input: {row['input_text'][:100]}...")


def main():
    """Main entry point."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    consolidated_path = project_root / 'scripts' / 'human_judging_analysis' / 'output' / 'consolidated_human_ratings.csv'
    dataset_path = project_root / 'data' / 'humane_bench.jsonl'
    output_path = project_root / 'scripts' / 'human_judging_analysis' / 'output' / 'golden_questions.csv'

    # Extract golden questions
    extract_golden_questions(consolidated_path, dataset_path, output_path)


if __name__ == '__main__':
    main()
