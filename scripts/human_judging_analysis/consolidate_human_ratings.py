"""
Consolidate human ratings from individual CSV files into a single dataset.

This script reads ratings from all human raters in data/human_ratings/ and combines them
into a single consolidated CSV file with observation IDs based on unique combinations
of input_id, ai_model, and ai_persona.
"""

import pandas as pd
import hashlib
from pathlib import Path

# Rating conversion map (same as existing consolidate_ratings.py)
RATING_MAP = {
    'HELL YES': 1.0,
    'Soft yes': 0.5,
    'Soft no': -0.5,
    'HELL NO': -1.0
}


def generate_observation_id(input_id: str, ai_model: str, ai_persona: str) -> str:
    """
    Generate a unique observation ID based on input_id, ai_model, and ai_persona.

    Args:
        input_id: The sample/input identifier
        ai_model: The AI model name
        ai_persona: The persona type (good, baseline, bad)

    Returns:
        MD5 hash (16 characters) of the combined fields
    """
    combined = f"{input_id}|{ai_model}|{ai_persona}"
    hash_obj = hashlib.md5(combined.encode())
    return hash_obj.hexdigest()[:16]


def convert_rating(rating_value) -> float:
    """
    Convert string rating to numeric value.

    Args:
        rating_value: String rating like "HELL YES" or "Soft no"

    Returns:
        Numeric rating value, or None if invalid
    """
    if isinstance(rating_value, str):
        rating_str = rating_value.strip()
        if rating_str in RATING_MAP:
            return RATING_MAP[rating_str]
    return None


def load_rater_file(file_path: Path) -> pd.DataFrame:
    """
    Load a single rater's CSV file and process it.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with processed ratings
    """
    # Extract rater name from filename (e.g., "rater-1" from "rater-1_ratings.csv")
    rater_name = file_path.stem.replace('_ratings', '').replace('_partial', '')

    # Read CSV
    df = pd.read_csv(file_path)

    # Normalize column names (rater-3's file has "rating (rubric)" instead of "rating")
    if 'rating (rubric)' in df.columns:
        df = df.rename(columns={'rating (rubric)': 'rating'})

    # Add rater name column
    df['rater_name'] = rater_name

    # Convert ratings
    df['rating_numeric'] = df['rating'].apply(convert_rating)

    # Generate observation IDs
    df['observation_id'] = df.apply(
        lambda row: generate_observation_id(
            str(row['input_id']),
            str(row['ai_model']),
            str(row['ai_persona'])
        ),
        axis=1
    )

    # Filter out invalid ratings
    df = df[df['rating_numeric'].notna()].copy()

    return df


def consolidate_ratings(data_dir: Path, output_path: Path):
    """
    Consolidate all human ratings into a single CSV file.

    Args:
        data_dir: Directory containing individual rater CSV files
        output_path: Path where consolidated CSV should be saved
    """
    # Find all rating CSV files
    rating_files = list(data_dir.glob('*_ratings*.csv'))

    if not rating_files:
        raise ValueError(f"No rating files found in {data_dir}")

    print(f"Found {len(rating_files)} rating files:")
    for f in rating_files:
        print(f"  - {f.name}")

    # Load and combine all files
    all_ratings = []
    for file_path in rating_files:
        print(f"\nProcessing {file_path.name}...")
        df = load_rater_file(file_path)
        print(f"  Loaded {len(df)} valid ratings")
        all_ratings.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_ratings, ignore_index=True)

    # Select and order columns
    output_columns = [
        'observation_id',
        'input_id',
        'ai_model',
        'ai_persona',
        'rater_name',
        'rating',
        'rating_numeric',
        'reasoning',
        'misc_comments',
        'input_text',
        'ai_output'
    ]

    # Ensure all columns exist (some might be missing)
    for col in output_columns:
        if col not in combined_df.columns:
            combined_df[col] = None

    combined_df = combined_df[output_columns]

    # Sort by observation_id and rater_name for easy comparison
    combined_df = combined_df.sort_values(['observation_id', 'rater_name']).reset_index(drop=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    # Print summary statistics
    print(f"\n{'='*60}")
    print("CONSOLIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total ratings: {len(combined_df)}")
    print(f"Unique observations: {combined_df['observation_id'].nunique()}")
    print(f"Unique raters: {combined_df['rater_name'].nunique()}")
    print(f"\nRatings per rater:")
    for rater, count in combined_df['rater_name'].value_counts().items():
        print(f"  {rater}: {count}")

    print(f"\nObservations by number of raters:")
    obs_counts = combined_df.groupby('observation_id').size()
    for n_raters in sorted(obs_counts.unique()):
        count = (obs_counts == n_raters).sum()
        print(f"  {n_raters} raters: {count} observations")

    print(f"\nRating distribution:")
    for rating, count in combined_df['rating'].value_counts().sort_index().items():
        numeric = combined_df[combined_df['rating'] == rating]['rating_numeric'].iloc[0]
        print(f"  {rating} ({numeric:+.1f}): {count}")

    print(f"\nOutput saved to: {output_path}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'human_ratings'
    output_dir = project_root / 'scripts' / 'human_judging_analysis' / 'output'
    output_path = output_dir / 'consolidated_human_ratings.csv'

    # Run consolidation
    consolidate_ratings(data_dir, output_path)


if __name__ == '__main__':
    main()
