"""
HumaneBench Analysis Script
Calculates inter-rater reliability and descriptive statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports - script will gracefully handle if not installed
try:
    import krippendorff
    KRIPPENDORFF_AVAILABLE = True
except ImportError:
    KRIPPENDORFF_AVAILABLE = False
    print("⚠ Warning: krippendorff package not installed. IRR calculation will be skipped.")
    print("  Install with: pip install krippendorff")

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("⚠ Warning: pingouin package not installed. ICC calculation will be skipped.")
    print("  Install with: pip install pingouin")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠ Warning: scipy not installed. Bootstrap CIs will be skipped.")
    print("  Install with: pip install scipy")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "./output/consolidated_ratings.csv"
OUTPUT_DIR = "./output/analysis"

# Minimum samples with ≥2 raters needed for IRR calculation
MIN_SAMPLES_FOR_IRR = 20

# ============================================================================
# Analysis Functions
# ============================================================================

def load_data(filepath):
    """Load consolidated ratings data"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"\n✓ Loaded {len(df)} ratings")
    print(f"  Raters: {df['rater_name'].nunique()}")
    print(f"  Samples (prompts): {df['sample_id'].nunique() if 'sample_id' in df.columns else 'N/A'}")
    print(f"  Observations (responses): {df['observation_id'].nunique() if 'observation_id' in df.columns else 'N/A'}")
    print(f"  Principles: {df['principle'].nunique()}")

    return df


def calculate_coverage(df):
    """Calculate coverage statistics per principle"""
    print("\n" + "="*80)
    print("COVERAGE ANALYSIS")
    print("="*80)
    
    coverage_stats = []
    
    for principle in sorted(df['principle'].unique()):
        principle_df = df[df['principle'] == principle]
        
        # Count ratings per observation (each observation is a unique model response)
        observation_counts = principle_df.groupby('observation_id').size()
        
        stats = {
            'principle': principle,
            'total_observations': len(observation_counts),
            'total_ratings': len(principle_df),
            'observations_2plus_raters': (observation_counts >= 2).sum(),
            'observations_3plus_raters': (observation_counts >= 3).sum(),
            'observations_5plus_raters': (observation_counts >= 5).sum(),
            'avg_ratings_per_observation': observation_counts.mean(),
            'max_ratings_per_observation': observation_counts.max(),
            'unique_raters': principle_df['rater_name'].nunique()
        }
        
        coverage_stats.append(stats)
    
    coverage_df = pd.DataFrame(coverage_stats)
    
    print("\nCoverage by Principle:")
    print("-"*80)
    for _, row in coverage_df.iterrows():
        print(f"\n{row['principle']}")
        print(f"  Total observations: {row['total_observations']}")
        print(f"  Total ratings: {row['total_ratings']}")
        print(f"  Observations with ≥2 raters: {row['observations_2plus_raters']}")
        print(f"  Observations with ≥3 raters: {row['observations_3plus_raters']}")
        print(f"  Avg ratings/observation: {row['avg_ratings_per_observation']:.1f}")
        print(f"  Unique raters: {row['unique_raters']}")

        # Determine if sufficient for IRR
        if row['observations_2plus_raters'] >= MIN_SAMPLES_FOR_IRR:
            print(f"  ✓ SUFFICIENT for IRR (≥{MIN_SAMPLES_FOR_IRR} observations with 2+ raters)")
        else:
            print(f"  ✗ Insufficient for IRR (need {MIN_SAMPLES_FOR_IRR}, have {row['observations_2plus_raters']})")
    
    return coverage_df


def calculate_descriptive_stats(df):
    """Calculate Tier 1 descriptive statistics (always calculated)"""
    print("\n" + "="*80)
    print("TIER 1: DESCRIPTIVE STATISTICS")
    print("="*80)
    
    desc_stats = []
    
    for principle in sorted(df['principle'].unique()):
        principle_df = df[df['principle'] == principle]
        
        # Rating distribution
        rating_counts = principle_df['rating'].value_counts()
        
        # Calculate percentages
        total = len(principle_df)
        pct_positive = ((principle_df['rating'] >= 0.5).sum() / total * 100) if total > 0 else 0
        pct_negative = ((principle_df['rating'] <= -0.5).sum() / total * 100) if total > 0 else 0
        
        stats = {
            'principle': principle,
            'n_ratings': len(principle_df),
            'median': principle_df['rating'].median(),
            'mean': principle_df['rating'].mean(),
            'iqr': principle_df['rating'].quantile(0.75) - principle_df['rating'].quantile(0.25),
            'count_minus1': rating_counts.get(-1.0, 0),
            'count_minus0.5': rating_counts.get(-0.5, 0),
            'count_plus0.5': rating_counts.get(0.5, 0),
            'count_plus1': rating_counts.get(1.0, 0),
            'pct_positive': pct_positive,
            'pct_negative': pct_negative
        }
        
        desc_stats.append(stats)
    
    desc_df = pd.DataFrame(desc_stats)
    
    print("\nDescriptive Statistics by Principle:")
    print("-"*80)
    print(desc_df.to_string(index=False))
    
    return desc_df


def prepare_irr_matrix(principle_df):
    """Prepare data matrix for IRR calculation (raters × items)"""
    # Create pivot table: rows = raters, columns = observations
    # This creates a matrix suitable for Krippendorff's alpha
    pivot = principle_df.pivot_table(
        index='rater_name',
        columns='observation_id',
        values='rating',
        aggfunc='first'  # Use first rating if duplicates (shouldn't happen)
    )

    # Convert to numpy array for krippendorff package
    # It expects shape (raters, items) with NaN for missing
    return pivot.values


def calculate_percentage_agreement(principle_df):
    """Calculate simple percentage agreement for observations with multiple raters"""
    # Get observations with 2+ raters
    observation_counts = principle_df.groupby('observation_id').size()
    multi_rated_observations = observation_counts[observation_counts >= 2].index

    if len(multi_rated_observations) == 0:
        return None, 0

    agreements = []
    total_pairs = 0

    for observation_id in multi_rated_observations:
        observation_ratings = principle_df[principle_df['observation_id'] == observation_id]['rating'].values
        
        # Count exact agreements among all pairs
        n_raters = len(observation_ratings)
        for i in range(n_raters):
            for j in range(i+1, n_raters):
                total_pairs += 1
                if observation_ratings[i] == observation_ratings[j]:
                    agreements.append(1)
                else:
                    agreements.append(0)
    
    if total_pairs == 0:
        return None, 0

    pct_agreement = (sum(agreements) / total_pairs) * 100
    return pct_agreement, len(multi_rated_observations)


def calculate_krippendorff_alpha(principle_df, principle_name):
    """Calculate Krippendorff's alpha with bootstrap CI"""
    if not KRIPPENDORFF_AVAILABLE:
        return None, None, None
    
    # Prepare data matrix
    reliability_data = prepare_irr_matrix(principle_df)
    
    # Calculate alpha
    try:
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement='ordinal'
        )
        
        # Bootstrap confidence intervals if scipy available
        if SCIPY_AVAILABLE:
            n_bootstrap = 1000
            bootstrap_alphas = []
            
            n_raters, n_items = reliability_data.shape
            
            for _ in range(n_bootstrap):
                # Resample items (columns) with replacement
                indices = np.random.choice(n_items, size=n_items, replace=True)
                resampled_data = reliability_data[:, indices]
                
                try:
                    boot_alpha = krippendorff.alpha(
                        reliability_data=resampled_data,
                        level_of_measurement='ordinal'
                    )
                    if not np.isnan(boot_alpha):
                        bootstrap_alphas.append(boot_alpha)
                except:
                    continue
            
            if len(bootstrap_alphas) > 0:
                ci_lower = np.percentile(bootstrap_alphas, 2.5)
                ci_upper = np.percentile(bootstrap_alphas, 97.5)
            else:
                ci_lower, ci_upper = None, None
        else:
            ci_lower, ci_upper = None, None
        
        return alpha, ci_lower, ci_upper
    
    except Exception as e:
        print(f"    Error calculating alpha for {principle_name}: {e}")
        return None, None, None


def calculate_icc(principle_df, principle_name):
    """Calculate ICC(2,k) if sufficient data"""
    if not PINGOUIN_AVAILABLE:
        return None

    # Need at least 30 observations for ICC reliability
    observation_counts = principle_df.groupby('observation_id').size()
    multi_rated_observations = observation_counts[observation_counts >= 2].index

    if len(multi_rated_observations) < 30:
        return None

    # Filter to only multi-rated observations
    icc_df = principle_df[principle_df['observation_id'].isin(multi_rated_observations)].copy()

    try:
        # Calculate ICC
        icc_result = pg.intraclass_corr(
            data=icc_df,
            targets='observation_id',
            raters='rater_name',
            ratings='rating'
        )
        
        # Get ICC(2,k) - "ICC2k" in pingouin
        icc2k = icc_result[icc_result['Type'] == 'ICC2k']['ICC'].values[0]
        
        return icc2k
    
    except Exception as e:
        print(f"    Error calculating ICC for {principle_name}: {e}")
        return None


def calculate_irr_metrics(df, coverage_df):
    """Calculate Tier 2 IRR metrics where sufficient data exists"""
    print("\n" + "="*80)
    print("TIER 2: INTER-RATER RELIABILITY")
    print("="*80)
    
    irr_results = []
    
    for _, row in coverage_df.iterrows():
        principle = row['principle']
        
        print(f"\n{principle}")
        print("-"*80)

        # Check if sufficient data
        if row['observations_2plus_raters'] < MIN_SAMPLES_FOR_IRR:
            print(f"  ✗ Skipping: only {row['observations_2plus_raters']} observations with 2+ raters")
            print(f"    (need {MIN_SAMPLES_FOR_IRR})")

            irr_results.append({
                'principle': principle,
                'sufficient_data': False,
                'n_samples_for_irr': row['observations_2plus_raters'],
                'krippendorff_alpha': None,
                'alpha_ci_lower': None,
                'alpha_ci_upper': None,
                'pct_agreement': None,
                'icc_2k': None
            })
            continue

        print(f"  ✓ Calculating IRR ({row['observations_2plus_raters']} observations with 2+ raters)")

        principle_df = df[df['principle'] == principle]
        
        # Calculate percentage agreement
        pct_agreement, n_samples = calculate_percentage_agreement(principle_df)
        if pct_agreement is not None:
            print(f"    Percentage agreement: {pct_agreement:.1f}%")
        
        # Calculate Krippendorff's alpha
        alpha, ci_lower, ci_upper = calculate_krippendorff_alpha(principle_df, principle)
        if alpha is not None:
            if ci_lower is not None:
                print(f"    Krippendorff's α: {alpha:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
            else:
                print(f"    Krippendorff's α: {alpha:.3f}")
        
        # Calculate ICC if sufficient data
        icc = calculate_icc(principle_df, principle)
        if icc is not None:
            print(f"    ICC(2,k): {icc:.3f}")
        
        irr_results.append({
            'principle': principle,
            'sufficient_data': True,
            'n_samples_for_irr': row['observations_2plus_raters'],
            'krippendorff_alpha': alpha,
            'alpha_ci_lower': ci_lower,
            'alpha_ci_upper': ci_upper,
            'pct_agreement': pct_agreement,
            'icc_2k': icc
        })
    
    irr_df = pd.DataFrame(irr_results)
    
    # Print summary
    print("\n" + "="*80)
    print("IRR SUMMARY")
    print("="*80)
    
    sufficient = irr_df[irr_df['sufficient_data'] == True]
    print(f"\nPrinciples with sufficient data for IRR: {len(sufficient)}/{len(irr_df)}")
    
    if len(sufficient) > 0:
        print("\nIRR Metrics (where calculable):")
        print("-"*80)
        display_cols = ['principle', 'n_samples_for_irr', 'krippendorff_alpha', 
                       'pct_agreement', 'icc_2k']
        print(sufficient[display_cols].to_string(index=False))
    
    return irr_df


def generate_summary_report(df, coverage_df, desc_df, irr_df):
    """Generate executive summary for MIT presentation"""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY FOR MIT PRESENTATION")
    print("="*80)
    
    print("\n" + "─"*80)
    print("STUDY OVERVIEW")
    print("─"*80)
    print(f"Total ratings collected: {len(df)}")
    print(f"Number of raters: {df['rater_name'].nunique()}")
    print(f"Unique prompts evaluated: {df['sample_id'].nunique() if 'sample_id' in df.columns else 'N/A'}")
    print(f"Unique observations evaluated: {df['observation_id'].nunique() if 'observation_id' in df.columns else 'N/A'}")
    print(f"Principles evaluated: {df['principle'].nunique()}")
    print(f"Data sources: {', '.join(df['source'].unique())}")

    print("\n" + "─"*80)
    print("DATA QUALITY")
    print("─"*80)

    # Coverage summary
    total_principles = len(coverage_df)
    principles_with_irr = (coverage_df['observations_2plus_raters'] >= MIN_SAMPLES_FOR_IRR).sum()

    print(f"Principles with sufficient overlap for IRR: {principles_with_irr}/{total_principles}")

    avg_overlap = coverage_df['observations_2plus_raters'].mean()
    print(f"Average samples with 2+ raters per principle: {avg_overlap:.1f}")
    
    print("\n" + "─"*80)
    print("KEY FINDINGS")
    print("─"*80)
    
    # Overall rating distribution
    overall_positive_pct = (df['rating'] >= 0.5).sum() / len(df) * 100
    overall_negative_pct = (df['rating'] <= -0.5).sum() / len(df) * 100
    
    print(f"Overall positive ratings (≥0.5): {overall_positive_pct:.1f}%")
    print(f"Overall negative ratings (≤-0.5): {overall_negative_pct:.1f}%")
    
    # Best/worst performing principles
    print(f"\nHighest-rated principle: {desc_df.loc[desc_df['median'].idxmax(), 'principle']}")
    print(f"  Median rating: {desc_df['median'].max():.2f}")
    
    print(f"\nLowest-rated principle: {desc_df.loc[desc_df['median'].idxmin(), 'principle']}")
    print(f"  Median rating: {desc_df['median'].min():.2f}")
    
    # IRR results if available
    if principles_with_irr > 0:
        valid_alphas = irr_df[irr_df['krippendorff_alpha'].notna()]
        if len(valid_alphas) > 0:
            avg_alpha = valid_alphas['krippendorff_alpha'].mean()
            print(f"\nAverage Krippendorff's α (where calculable): {avg_alpha:.3f}")
            
            if avg_alpha >= 0.80:
                print("  → Strong reliability")
            elif avg_alpha >= 0.67:
                print("  → Acceptable reliability")
            else:
                print("  → Below acceptable threshold (0.67)")


def save_results(coverage_df, desc_df, irr_df, output_dir):
    """Save all results to CSV files"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save each results table
    coverage_path = Path(output_dir) / "coverage_analysis.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"✓ Coverage analysis: {coverage_path}")
    
    desc_path = Path(output_dir) / "descriptive_statistics.csv"
    desc_df.to_csv(desc_path, index=False)
    print(f"✓ Descriptive statistics: {desc_path}")
    
    irr_path = Path(output_dir) / "irr_metrics.csv"
    irr_df.to_csv(irr_path, index=False)
    print(f"✓ IRR metrics: {irr_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*80)
    print("HUMANEBENCH PILOT ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_data(INPUT_FILE)
    
    # Calculate coverage
    coverage_df = calculate_coverage(df)
    
    # Tier 1: Descriptive statistics (always calculated)
    desc_df = calculate_descriptive_stats(df)
    
    # Tier 2: IRR metrics (where sufficient data)
    irr_df = calculate_irr_metrics(df, coverage_df)
    
    # Generate summary
    generate_summary_report(df, coverage_df, desc_df, irr_df)
    
    # Save results
    save_results(coverage_df, desc_df, irr_df, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
