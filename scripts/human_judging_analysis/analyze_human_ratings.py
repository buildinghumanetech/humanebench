"""
Analyze human ratings data and calculate inter-rater reliability (IRR) metrics.

This script calculates:
- Coverage analysis (how many raters per observation)
- Descriptive statistics (rating distributions)
- IRR metrics (percentage agreement, Krippendorff's Alpha, ICC)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from scipy import stats
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_coverage_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate coverage statistics - how many raters rated each observation.

    Args:
        df: DataFrame with consolidated ratings

    Returns:
        Dictionary with coverage statistics
    """
    observation_counts = df.groupby('observation_id').size()

    stats_dict = {
        'total_observations': len(observation_counts),
        'total_ratings': len(df),
        'observations_2_raters': (observation_counts >= 2).sum(),
        'observations_3_raters': (observation_counts >= 3).sum(),
        'observations_4_raters': (observation_counts == 4).sum(),
        'avg_ratings_per_observation': observation_counts.mean(),
        'max_ratings_per_observation': observation_counts.max(),
        'min_ratings_per_observation': observation_counts.min(),
        'unique_raters': df['rater_name'].nunique(),
    }

    # Count observations by exact number of raters
    for n in range(1, int(observation_counts.max()) + 1):
        stats_dict[f'observations_exactly_{n}_raters'] = (observation_counts == n).sum()

    return stats_dict


def calculate_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for ratings.

    Args:
        df: DataFrame with consolidated ratings

    Returns:
        Dictionary with descriptive statistics
    """
    ratings = df['rating_numeric']
    rating_counts = ratings.value_counts()

    # Calculate percentages
    total = len(ratings)
    pct_positive = (ratings > 0).sum() / total * 100
    pct_negative = (ratings < 0).sum() / total * 100

    stats_dict = {
        'n_ratings': len(ratings),
        'mean': ratings.mean(),
        'median': ratings.median(),
        'std': ratings.std(),
        'min': ratings.min(),
        'max': ratings.max(),
        'q25': ratings.quantile(0.25),
        'q75': ratings.quantile(0.75),
        'iqr': ratings.quantile(0.75) - ratings.quantile(0.25),
        'count_minus1': rating_counts.get(-1.0, 0),
        'count_minus0.5': rating_counts.get(-0.5, 0),
        'count_plus0.5': rating_counts.get(0.5, 0),
        'count_plus1': rating_counts.get(1.0, 0),
        'pct_positive': pct_positive,
        'pct_negative': pct_negative,
    }

    return stats_dict


def calculate_percentage_agreement(df: pd.DataFrame) -> Optional[float]:
    """
    Calculate pairwise percentage agreement between raters.

    Args:
        df: DataFrame with consolidated ratings

    Returns:
        Percentage agreement (0-100), or None if insufficient data
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

    if len(multi_rated_cols) == 0:
        return None

    pivot_filtered = pivot[multi_rated_cols]

    # Calculate pairwise agreement
    raters = pivot_filtered.index.tolist()
    agreements = []

    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            rater1 = pivot_filtered.loc[raters[i]]
            rater2 = pivot_filtered.loc[raters[j]]

            # Find observations both raters rated
            both_rated = rater1.notna() & rater2.notna()

            if both_rated.sum() > 0:
                agreed = (rater1[both_rated] == rater2[both_rated]).sum()
                total = both_rated.sum()
                agreements.append(agreed / total)

    if not agreements:
        return None

    return np.mean(agreements) * 100


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
        return {'alpha': None, 'ci_lower': None, 'ci_upper': None}

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


def analyze_ratings(input_path: Path, output_dir: Path):
    """
    Analyze consolidated human ratings and calculate IRR metrics.

    Args:
        input_path: Path to consolidated ratings CSV
        output_dir: Directory where output files should be saved
    """
    # Load data
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} ratings from {input_path}")
    print(f"Unique observations: {df['observation_id'].nunique()}")
    print(f"Unique raters: {df['rater_name'].nunique()}")

    # Calculate coverage statistics
    print("\nCalculating coverage statistics...")
    coverage_stats = calculate_coverage_stats(df)

    # Calculate descriptive statistics
    print("Calculating descriptive statistics...")
    descriptive_stats = calculate_descriptive_stats(df)

    # Calculate IRR metrics
    print("Calculating IRR metrics...")

    # Percentage agreement
    print("  - Percentage agreement...")
    pct_agreement = calculate_percentage_agreement(df)

    # Krippendorff's Alpha
    print("  - Krippendorff's Alpha (with bootstrap CI)...")
    alpha_results = krippendorff_alpha(df, n_bootstrap=1000)

    # ICC
    print("  - ICC(2,k)...")
    icc_value = calculate_icc(df)

    # Combine IRR metrics
    irr_metrics = {
        'percentage_agreement': pct_agreement,
        'krippendorff_alpha': alpha_results['alpha'],
        'krippendorff_alpha_ci_lower': alpha_results['ci_lower'],
        'krippendorff_alpha_ci_upper': alpha_results['ci_upper'],
        'krippendorff_n_observations': alpha_results['n_observations'],
        'icc_2k': icc_value,
    }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coverage analysis
    coverage_df = pd.DataFrame([coverage_stats])
    coverage_path = output_dir / 'coverage_analysis.csv'
    coverage_df.to_csv(coverage_path, index=False)
    print(f"\nCoverage analysis saved to: {coverage_path}")

    # Descriptive statistics
    descriptive_df = pd.DataFrame([descriptive_stats])
    descriptive_path = output_dir / 'descriptive_stats.csv'
    descriptive_df.to_csv(descriptive_path, index=False)
    print(f"Descriptive statistics saved to: {descriptive_path}")

    # IRR metrics
    irr_df = pd.DataFrame([irr_metrics])
    irr_path = output_dir / 'irr_metrics.csv'
    irr_df.to_csv(irr_path, index=False)
    print(f"IRR metrics saved to: {irr_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print("\nCOVERAGE:")
    print(f"  Total observations: {coverage_stats['total_observations']}")
    print(f"  Total ratings: {coverage_stats['total_ratings']}")
    print(f"  Observations with 4 raters: {coverage_stats['observations_4_raters']}")
    print(f"  Observations with 3 raters: {coverage_stats.get('observations_exactly_3_raters', 0)}")
    print(f"  Observations with 2 raters: {coverage_stats.get('observations_exactly_2_raters', 0)}")
    print(f"  Average ratings per observation: {coverage_stats['avg_ratings_per_observation']:.2f}")

    print("\nDESCRIPTIVE STATISTICS:")
    print(f"  Mean rating: {descriptive_stats['mean']:.3f}")
    print(f"  Median rating: {descriptive_stats['median']:.3f}")
    print(f"  Std dev: {descriptive_stats['std']:.3f}")
    print(f"  Positive ratings: {descriptive_stats['pct_positive']:.1f}%")
    print(f"  Negative ratings: {descriptive_stats['pct_negative']:.1f}%")

    print("\nINTER-RATER RELIABILITY:")
    if pct_agreement is not None:
        print(f"  Percentage Agreement: {pct_agreement:.1f}%")
    if alpha_results['alpha'] is not None:
        print(f"  Krippendorff's Alpha: {alpha_results['alpha']:.3f}")
        if alpha_results['ci_lower'] is not None:
            print(f"    95% CI: [{alpha_results['ci_lower']:.3f}, {alpha_results['ci_upper']:.3f}]")
        print(f"    (based on {alpha_results['n_observations']} multi-rated observations)")
    if icc_value is not None:
        print(f"  ICC(2,k): {icc_value:.3f}")

    print(f"{'='*60}")


def main():
    """Main entry point."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / 'scripts' / 'human_judging_analysis' / 'output' / 'consolidated_human_ratings.csv'
    output_dir = project_root / 'scripts' / 'human_judging_analysis' / 'output'

    # Run analysis
    analyze_ratings(input_path, output_dir)


if __name__ == '__main__':
    main()
