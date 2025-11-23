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
from typing import List, Dict, Any, Optional
from scipy import stats
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


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


def krippendorff_alpha(df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict[str, Any]:
    """
    Calculate Krippendorff's Alpha for ordinal data with bootstrap confidence intervals.

    Args:
        df: DataFrame with consolidated ratings
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with alpha value and confidence intervals
    """
    # Create pivot table: rows=raters, columns=observations
    pivot = df.pivot_table(
        index='rater_name',
        columns='observation_id',
        values='rating_numeric',
        aggfunc='first'
    )

    # Only consider observations rated by at least 2 raters
    multi_rated_cols = pivot.columns[pivot.notna().sum() >= 2]

    if len(multi_rated_cols) < 10:  # Need sufficient data
        return {'alpha': None, 'ci_lower': None, 'ci_upper': None, 'n_observations': len(multi_rated_cols)}

    pivot_filtered = pivot[multi_rated_cols]

    # Convert to format needed for Krippendorff's alpha
    # Each row is a rater, each column is an observation
    reliability_matrix = pivot_filtered.values

    # Calculate observed disagreement
    def calculate_alpha(matrix):
        n_raters, n_items = matrix.shape

        # Get all valid values (non-NaN)
        all_values = []
        for i in range(n_items):
            col = matrix[:, i]
            valid = col[~np.isnan(col)]
            all_values.extend(valid)

        if len(all_values) == 0:
            return np.nan

        all_values = np.array(all_values)
        unique_values = np.unique(all_values)

        # Calculate coincidence matrix
        n_values = len(unique_values)
        coincidence_matrix = np.zeros((n_values, n_values))

        for i in range(n_items):
            col = matrix[:, i]
            valid = col[~np.isnan(col)]
            n_valid = len(valid)

            if n_valid >= 2:
                for v1 in valid:
                    for v2 in valid:
                        idx1 = np.where(unique_values == v1)[0][0]
                        idx2 = np.where(unique_values == v2)[0][0]
                        coincidence_matrix[idx1, idx2] += 1.0 / (n_valid - 1) if v1 != v2 else 1.0 / n_valid

        # Calculate observed disagreement
        total_pairs = coincidence_matrix.sum()
        if total_pairs == 0:
            return np.nan

        observed_disagreement = 0
        for i in range(n_values):
            for j in range(n_values):
                if i != j:
                    # For ordinal data, use squared difference
                    metric = (i - j) ** 2
                    observed_disagreement += coincidence_matrix[i, j] * metric

        observed_disagreement /= total_pairs

        # Calculate expected disagreement
        marginals = coincidence_matrix.sum(axis=1)
        expected_disagreement = 0
        total_marginal = marginals.sum()

        for i in range(n_values):
            for j in range(n_values):
                if i != j:
                    metric = (i - j) ** 2
                    expected_disagreement += (marginals[i] * marginals[j] * metric)

        expected_disagreement /= (total_marginal * (total_marginal - 1))

        if expected_disagreement == 0:
            return 1.0 if observed_disagreement == 0 else 0.0

        alpha = 1 - (observed_disagreement / expected_disagreement)
        return alpha

    # Calculate main alpha
    alpha_value = calculate_alpha(reliability_matrix)

    # Bootstrap for confidence intervals
    if not np.isnan(alpha_value):
        bootstrap_alphas = []
        n_raters, n_items = reliability_matrix.shape

        for _ in range(n_bootstrap):
            # Resample observations (columns)
            resample_idx = np.random.choice(n_items, size=n_items, replace=True)
            resampled_matrix = reliability_matrix[:, resample_idx]
            bootstrap_alpha = calculate_alpha(resampled_matrix)

            if not np.isnan(bootstrap_alpha):
                bootstrap_alphas.append(bootstrap_alpha)

        if len(bootstrap_alphas) > 0:
            ci_lower = np.percentile(bootstrap_alphas, 2.5)
            ci_upper = np.percentile(bootstrap_alphas, 97.5)
        else:
            ci_lower = None
            ci_upper = None
    else:
        ci_lower = None
        ci_upper = None

    return {
        'alpha': alpha_value if not np.isnan(alpha_value) else None,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_observations': len(multi_rated_cols)
    }


def calculate_icc(df: pd.DataFrame) -> Optional[float]:
    """
    Calculate ICC(2,k) - Two-way random effects, consistency, average measures.

    Args:
        df: DataFrame with consolidated ratings

    Returns:
        ICC value, or None if insufficient data
    """
    # Create pivot table: rows=observations, columns=raters
    pivot = df.pivot_table(
        index='observation_id',
        columns='rater_name',
        values='rating_numeric',
        aggfunc='first'
    )

    # Only use observations rated by all raters
    complete_obs = pivot.dropna()

    if len(complete_obs) < 10:  # Need at least 10 complete observations
        return None

    # Calculate ICC(2,k)
    ratings_matrix = complete_obs.values
    n_subjects, n_raters = ratings_matrix.shape

    # Grand mean
    grand_mean = np.mean(ratings_matrix)

    # Sum of squares
    ss_total = np.sum((ratings_matrix - grand_mean) ** 2)
    ss_rows = n_raters * np.sum((np.mean(ratings_matrix, axis=1) - grand_mean) ** 2)
    ss_cols = n_subjects * np.sum((np.mean(ratings_matrix, axis=0) - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Degrees of freedom
    df_rows = n_subjects - 1
    df_cols = n_raters - 1
    df_error = df_rows * df_cols

    # Mean squares
    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error if df_error > 0 else 0

    # ICC(2,k) formula
    if (ms_rows + ms_cols - ms_error) == 0:
        return None

    icc = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n_raters)

    return icc


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

    # Calculate IRR metrics for golden questions only
    print("\nCalculating IRR metrics for golden questions...")
    golden_obs_ids = golden_df['observation_id'].unique()
    golden_ratings_df = ratings_df[ratings_df['observation_id'].isin(golden_obs_ids)]

    # Krippendorff's Alpha
    print("  - Krippendorff's Alpha (with bootstrap CI)...")
    alpha_results = krippendorff_alpha(golden_ratings_df, n_bootstrap=1000)

    # ICC
    print("  - ICC(2,k)...")
    icc_value = calculate_icc(golden_ratings_df)

    # Combine IRR metrics
    irr_metrics = {
        'krippendorff_alpha': alpha_results['alpha'],
        'krippendorff_alpha_ci_lower': alpha_results['ci_lower'],
        'krippendorff_alpha_ci_upper': alpha_results['ci_upper'],
        'krippendorff_n_observations': alpha_results['n_observations'],
        'icc_2k': icc_value,
        'n_golden_questions': len(golden_df),
        'n_golden_ratings': len(golden_ratings_df),
    }

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    golden_df.to_csv(output_path, index=False)

    # Save IRR metrics
    irr_output_path = output_path.parent / 'golden_questions_irr.csv'
    irr_df = pd.DataFrame([irr_metrics])
    irr_df.to_csv(irr_output_path, index=False)
    print(f"IRR metrics saved to: {irr_output_path}")

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

    # Print IRR metrics
    print(f"\n{'='*60}")
    print("INTER-RATER RELIABILITY (Golden Questions Only)")
    print(f"{'='*60}")
    if alpha_results['alpha'] is not None:
        print(f"Krippendorff's Alpha: {alpha_results['alpha']:.3f}")
        if alpha_results['ci_lower'] is not None:
            print(f"  95% CI: [{alpha_results['ci_lower']:.3f}, {alpha_results['ci_upper']:.3f}]")
        print(f"  (based on {alpha_results['n_observations']} multi-rated observations)")
    else:
        print(f"Krippendorff's Alpha: Not enough data (need 10+ multi-rated observations)")

    if icc_value is not None:
        print(f"ICC(2,k): {icc_value:.3f}")
    else:
        print(f"ICC(2,k): Not enough data (need 10+ complete observations)")

    print(f"\nNote: These metrics measure inter-rater reliability specifically")
    print(f"for golden questions where all raters agreed on direction.")
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
