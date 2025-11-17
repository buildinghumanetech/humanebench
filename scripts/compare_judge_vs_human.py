"""
Compare AI judge scores against human ratings for golden questions.

This script loads Inspect AI evaluation results and compares them to human
ratings to validate the AI judging pipeline.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys
from scipy import stats as scipy_stats
from sklearn.metrics import cohen_kappa_score


def load_inspect_results(log_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load Inspect AI evaluation results from a log file.

    Args:
        log_path: Path to the .eval log file

    Returns:
        Dictionary mapping sample_id to pattern scores
    """
    import zipfile

    # Inspect logs are ZIP files
    results = {}

    with zipfile.ZipFile(log_path, 'r') as zip_file:
        # List all sample files
        sample_files = [f for f in zip_file.namelist() if f.startswith('samples/') and f.endswith('.json')]

        # Read each sample file
        for sample_file in sample_files:
            with zip_file.open(sample_file) as f:
                sample_data = json.load(f)

            sample_id = sample_data.get('id')
            target = sample_data.get('target')
            scores = sample_data.get('scores', {})

            # Extract score from the overseer scorer
            if 'overseer' in scores:
                score_value = scores['overseer'].get('value')
                score_answer = scores['overseer'].get('answer')

                # Store as a dict with the principle name as key
                results[sample_id] = {
                    'principle': target or score_answer,
                    'score': score_value
                }

    return results


def load_human_ratings(consolidated_csv: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load human ratings from consolidated CSV.

    Args:
        consolidated_csv: Path to consolidated_human_ratings.csv

    Returns:
        Dictionary mapping observation_id to human rating info
    """
    df = pd.read_csv(consolidated_csv)

    # Group by observation_id and calculate consensus
    results = {}

    for obs_id, group in df.groupby('observation_id'):
        ratings = group['rating_numeric'].tolist()
        rater_names = group['rater_name'].tolist()

        # Calculate consensus (mean rating)
        consensus = np.mean(ratings)

        # Check if all same sign (golden question criterion)
        all_positive = all(r >= 0 for r in ratings)
        all_negative = all(r <= 0 for r in ratings)
        is_golden = all_positive or all_negative

        # Get principle from first row
        principle = group.iloc[0]['input_id']
        if isinstance(principle, str) and '-' in principle:
            # Extract principle name from input_id (e.g., "be-transparent-and-honest-082")
            principle = '-'.join(principle.split('-')[:-1])
        else:
            principle = "unknown"

        results[obs_id] = {
            'principle': principle,
            'num_raters': len(ratings),
            'ratings': ratings,
            'rater_names': rater_names,
            'consensus': consensus,
            'all_positive': all_positive,
            'all_negative': all_negative,
            'is_golden': is_golden,
            'ai_model': group.iloc[0].get('ai_model', 'unknown'),
            'ai_persona': group.iloc[0].get('ai_persona', 'unknown'),
        }

    return results


def compare_scores(
    inspect_results: Dict[str, Dict[str, Any]],
    human_ratings: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare AI judge scores to human ratings.

    Args:
        inspect_results: Dict mapping sample_id to score info
        human_ratings: Dict mapping observation_id to human rating info

    Returns:
        DataFrame with comparison results
    """
    comparisons = []

    for obs_id, human_data in human_ratings.items():
        if obs_id not in inspect_results:
            continue

        ai_data = inspect_results[obs_id]
        principle = human_data['principle']

        # Get AI judge score
        ai_score = ai_data.get('score')

        if ai_score is None:
            continue

        # Determine agreement
        human_consensus = human_data['consensus']
        human_direction = 'positive' if human_consensus > 0 else 'negative' if human_consensus < 0 else 'neutral'
        ai_direction = 'positive' if ai_score > 0 else 'negative' if ai_score < 0 else 'neutral'

        direction_match = human_direction == ai_direction

        # Calculate absolute difference
        abs_diff = abs(ai_score - human_consensus)

        comparisons.append({
            'observation_id': obs_id,
            'principle': principle,
            'ai_model': human_data['ai_model'],
            'ai_persona': human_data['ai_persona'],
            'num_human_raters': human_data['num_raters'],
            'human_consensus': human_consensus,
            'human_direction': human_direction,
            'ai_judge_score': ai_score,
            'ai_direction': ai_direction,
            'direction_match': direction_match,
            'abs_difference': abs_diff,
            'is_golden': human_data['is_golden'],
        })

    return pd.DataFrame(comparisons)


def discretize_score(score: float) -> float:
    """
    Round continuous score to nearest ordinal category.

    Args:
        score: Continuous score value

    Returns:
        Nearest ordinal category from {-1.0, -0.5, 0.5, 1.0}
    """
    categories = [-1.0, -0.5, 0.5, 1.0]
    return min(categories, key=lambda x: abs(x - score))


def analyze_agreement(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze agreement between AI judges and human raters.

    Analyzes agreement using both ordinal-appropriate metrics (Spearman, direction
    agreement, weighted kappa) and interval-based metrics (Pearson, MAE, RMSE).
    The latter assume equal intervals between rating categories, justified by the
    continuous underlying construct of humaneness and averaging across multiple raters.

    Args:
        comparison_df: DataFrame with comparison results

    Returns:
        Dictionary with agreement metrics
    """
    total = len(comparison_df)

    if total == 0:
        return {'error': 'No comparisons found'}

    # PRIMARY METRICS (ordinal-safe)

    # 1. Direction agreement
    direction_agreement = comparison_df['direction_match'].mean() * 100

    # 2. Spearman's rho (rank-based correlation - appropriate for ordinal data)
    spearman_result = scipy_stats.spearmanr(
        comparison_df['human_consensus'],
        comparison_df['ai_judge_score']
    )
    spearman_rho = spearman_result.correlation if not np.isnan(spearman_result.correlation) else None
    spearman_p = spearman_result.pvalue if spearman_rho is not None else None

    # 3. Weighted Cohen's kappa (ordinal metric with partial credit for near-misses)
    discretized_human = comparison_df['human_consensus'].apply(discretize_score)
    discretized_ai = comparison_df['ai_judge_score'].apply(discretize_score)

    # Convert to categorical labels (sklearn requires non-continuous data for kappa)
    category_map = {-1.0: 0, -0.5: 1, 0.5: 2, 1.0: 3}
    discretized_human_cat = discretized_human.map(category_map)
    discretized_ai_cat = discretized_ai.map(category_map)

    try:
        weighted_kappa = cohen_kappa_score(
            discretized_human_cat,
            discretized_ai_cat,
            weights='quadratic',
            labels=[0, 1, 2, 3]
        )
    except Exception as e:
        print(f"Warning: Could not calculate weighted kappa: {e}")
        weighted_kappa = None

    # SUPPLEMENTARY METRICS (interval assumption)

    # 4. Pearson correlation (assumes interval data - justified by continuous underlying construct)
    pearson_correlation = comparison_df['human_consensus'].corr(comparison_df['ai_judge_score'])

    # 5. Mean Absolute Error (assumes interval data)
    mae = comparison_df['abs_difference'].mean()

    # 6. Root Mean Square Error (assumes interval data, penalizes large errors more)
    rmse = np.sqrt((comparison_df['abs_difference'] ** 2).mean())

    # BREAKDOWNS

    # Agreement by persona
    persona_agreement = comparison_df.groupby('ai_persona')['direction_match'].mean() * 100

    # Agreement by principle (top 5)
    principle_agreement = comparison_df.groupby('principle')['direction_match'].agg([
        ('count', 'count'),
        ('agreement', lambda x: x.mean() * 100)
    ]).sort_values('count', ascending=False).head(5)

    return {
        # Primary metrics (ordinal-safe)
        'direction_agreement_pct': direction_agreement,
        'spearman_correlation': spearman_rho,
        'spearman_p_value': spearman_p,
        'weighted_kappa': weighted_kappa,

        # Supplementary metrics (interval assumption)
        'pearson_correlation': pearson_correlation,
        'mae': mae,
        'rmse': rmse,

        # Breakdowns
        'total_comparisons': total,
        'persona_agreement': persona_agreement.to_dict(),
        'principle_agreement': principle_agreement.to_dict('index'),
    }


def main(log_path: str = None):
    """
    Main entry point.

    Args:
        log_path: Path to Inspect AI log file (optional - will find latest if not provided)
    """
    # Set up paths
    project_root = Path(__file__).parent.parent
    consolidated_csv = project_root / 'scripts' / 'human_judging_analysis' / 'output' / 'consolidated_human_ratings.csv'
    output_dir = project_root / 'scripts' / 'human_judging_analysis' / 'output'

    # Find log file
    if log_path:
        log_path = Path(log_path)
    else:
        # Find the latest golden-questions-eval log
        logs_dir = project_root / 'logs'
        log_files = list(logs_dir.glob('*_golden-questions-eval_*.eval'))

        if not log_files:
            print("ERROR: No golden-questions-eval log files found in logs/")
            print("Please run: inspect eval src/golden_questions_task.py")
            sys.exit(1)

        # Sort by modification time, get latest
        log_path = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest log file: {log_path.name}")

    # Load data
    print(f"\nLoading Inspect AI results from {log_path}...")
    inspect_results = load_inspect_results(log_path)
    print(f"Loaded {len(inspect_results)} sample scores")

    print(f"\nLoading human ratings from {consolidated_csv}...")
    human_ratings = load_human_ratings(consolidated_csv)
    print(f"Loaded {len(human_ratings)} observations with human ratings")

    # Compare
    print("\nComparing AI judge scores to human ratings...")
    comparison_df = compare_scores(inspect_results, human_ratings)

    if comparison_df.empty:
        print("ERROR: No matching samples found between Inspect results and human ratings")
        sys.exit(1)

    print(f"Found {len(comparison_df)} matched samples")

    # Analyze agreement
    agreement_metrics = analyze_agreement(comparison_df)

    # Save comparison CSV
    comparison_path = output_dir / 'judge_vs_human_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison results saved to: {comparison_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("AGREEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"\nTotal comparisons: {agreement_metrics['total_comparisons']}")

    print(f"\nPRIMARY METRICS (Ordinal-appropriate):")
    print(f"  Direction agreement: {agreement_metrics['direction_agreement_pct']:.1f}%")
    if agreement_metrics['spearman_correlation'] is not None:
        print(f"  Spearman's rho: {agreement_metrics['spearman_correlation']:.3f}", end="")
        if agreement_metrics['spearman_p_value'] is not None:
            print(f" (p={agreement_metrics['spearman_p_value']:.4f})")
        else:
            print()
    if agreement_metrics['weighted_kappa'] is not None:
        print(f"  Weighted Cohen's kappa: {agreement_metrics['weighted_kappa']:.3f}")
        # Add interpretation
        kappa = agreement_metrics['weighted_kappa']
        if kappa < 0:
            interp = "no agreement"
        elif kappa < 0.20:
            interp = "slight"
        elif kappa < 0.40:
            interp = "fair"
        elif kappa < 0.60:
            interp = "moderate"
        elif kappa < 0.80:
            interp = "substantial"
        else:
            interp = "almost perfect"
        print(f"    (interpretation: {interp})")

    print(f"\nSUPPLEMENTARY METRICS (Interval assumption):")
    print(f"  Pearson correlation: {agreement_metrics['pearson_correlation']:.3f}")
    print(f"  Mean Absolute Error: {agreement_metrics['mae']:.3f}")
    print(f"  Root Mean Square Error: {agreement_metrics['rmse']:.3f}")

    print(f"\nAgreement by persona:")
    for persona, agreement in agreement_metrics['persona_agreement'].items():
        print(f"  {persona}: {agreement:.1f}%")

    print(f"\nAgreement by principle (top 5):")
    for principle, stats in agreement_metrics['principle_agreement'].items():
        print(f"  {principle}: {stats['agreement']:.1f}% ({int(stats['count'])} samples)")

    # Print disagreements
    disagreements = comparison_df[~comparison_df['direction_match']]

    if len(disagreements) > 0:
        print(f"\n{'='*60}")
        print(f"DISAGREEMENTS ({len(disagreements)} cases)")
        print(f"{'='*60}")

        for idx, row in disagreements.iterrows():
            print(f"\nObservation: {row['observation_id']}")
            print(f"  Principle: {row['principle']}")
            print(f"  Model: {row['ai_model']}, Persona: {row['ai_persona']}")
            print(f"  Human: {row['human_consensus']:+.2f} ({row['human_direction']})")
            print(f"  AI Judge: {row['ai_judge_score']:+.2f} ({row['ai_direction']})")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
