#!/usr/bin/env python3
"""
Longitudinal analysis of frontier labs' models over time.
"""

import csv
import pandas as pd

# Define model progressions for each lab
LAB_PROGRESSIONS = {
    "Anthropic": ["claude-sonnet-4", "claude-sonnet-4.5", "claude-opus-4.1"],
    "OpenAI": ["gpt-4o-2024-11-20", "gpt-4.1", "gpt-5", "gpt-5.1"],
    "Google": ["gemini-2.0-flash-001", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"],
    "Meta": ["llama-3.1-405b-instruct", "llama-4-maverick"]
}

def main():
    # Load data
    baseline = pd.read_csv('baseline_scores.csv')
    good_persona = pd.read_csv('good_persona_scores.csv')
    bad_persona = pd.read_csv('bad_persona_scores.csv')
    steerability = pd.read_csv('steerability_comparison.csv')

    # Prepare longitudinal data
    results = []

    for lab, models in LAB_PROGRESSIONS.items():
        for i, model in enumerate(models):
            baseline_row = baseline[baseline['model'] == model]
            good_row = good_persona[good_persona['model'] == model]
            bad_row = bad_persona[bad_persona['model'] == model]
            steer_row = steerability[steerability['model'] == model]

            if not baseline_row.empty or not bad_row.empty:
                def _val(row_df, col):
                    if row_df.empty or col not in row_df.columns:
                        return None
                    v = row_df[col].values[0]
                    return v if pd.notna(v) else None

                result = {
                    'lab': lab,
                    'model': model,
                    'generation': i + 1,
                    'baseline_score': _val(baseline_row, 'overall'),
                    'baseline_score_ci_lower': _val(baseline_row, 'overall_ci_lower'),
                    'baseline_score_ci_upper': _val(baseline_row, 'overall_ci_upper'),
                    'good_persona_score': _val(good_row, 'overall'),
                    'good_persona_score_ci_lower': _val(good_row, 'overall_ci_lower'),
                    'good_persona_score_ci_upper': _val(good_row, 'overall_ci_upper'),
                    'bad_persona_score': _val(bad_row, 'overall'),
                    'bad_persona_score_ci_lower': _val(bad_row, 'overall_ci_lower'),
                    'bad_persona_score_ci_upper': _val(bad_row, 'overall_ci_upper'),
                    'good_delta': _val(steer_row, 'good_delta'),
                    'good_delta_ci_lower': _val(steer_row, 'good_delta_ci_lower'),
                    'good_delta_ci_upper': _val(steer_row, 'good_delta_ci_upper'),
                    'bad_delta': _val(steer_row, 'bad_delta'),
                    'bad_delta_ci_lower': _val(steer_row, 'bad_delta_ci_lower'),
                    'bad_delta_ci_upper': _val(steer_row, 'bad_delta_ci_upper'),
                    'robustness_status': steer_row['robustness_status'].values[0] if not steer_row.empty else None,
                    'baseline_negative_rate': _val(baseline_row, 'negative_rate'),
                    'bad_negative_rate': _val(bad_row, 'negative_rate'),
                }
                results.append(result)

    # Write longitudinal comparison
    fieldnames = ['lab', 'model', 'generation',
                  'baseline_score', 'baseline_score_ci_lower', 'baseline_score_ci_upper',
                  'good_persona_score', 'good_persona_score_ci_lower', 'good_persona_score_ci_upper',
                  'bad_persona_score', 'bad_persona_score_ci_lower', 'bad_persona_score_ci_upper',
                  'good_delta', 'good_delta_ci_lower', 'good_delta_ci_upper',
                  'bad_delta', 'bad_delta_ci_lower', 'bad_delta_ci_upper',
                  'robustness_status', 'baseline_negative_rate', 'bad_negative_rate']

    with open('longitudinal_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Generated longitudinal_comparison.csv with {len(results)} model generations across 4 labs")

    # Print summary
    print("\nLongitudinal Trends Summary:\n")
    for lab, models in LAB_PROGRESSIONS.items():
        print(f"{lab}:")
        lab_results = [r for r in results if r['lab'] == lab]
        for r in lab_results:
            baseline = f"{r['baseline_score']:.3f}" if r['baseline_score'] else "N/A"
            bad = f"{r['bad_persona_score']:.3f}" if r['bad_persona_score'] else "N/A"
            status = r['robustness_status'] if r['robustness_status'] else "N/A"
            print(f"  Gen {r['generation']} ({r['model']}): Baseline={baseline}, Bad={bad}, Status={status}")
        print()

if __name__ == "__main__":
    main()
