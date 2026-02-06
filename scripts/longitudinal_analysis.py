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
                b = baseline_row['overall'].values[0] if not baseline_row.empty and pd.notna(baseline_row['overall'].values[0]) else None
                g = good_row['overall'].values[0] if not good_row.empty and pd.notna(good_row['overall'].values[0]) else None
                a = bad_row['overall'].values[0] if not bad_row.empty and pd.notna(bad_row['overall'].values[0]) else None

                result = {
                    'lab': lab,
                    'model': model,
                    'generation': i + 1,
                    'baseline_score': b,
                    'good_persona_score': g,
                    'bad_persona_score': a,
                    'good_delta': steer_row['good_delta'].values[0] if not steer_row.empty and pd.notna(steer_row['good_delta'].values[0]) else None,
                    'bad_delta': steer_row['bad_delta'].values[0] if not steer_row.empty and pd.notna(steer_row['bad_delta'].values[0]) else None,
                    'composite_humanescore': round((b + g + a) / 3, 3) if all(v is not None for v in [b, g, a]) else None,
                    'robustness_status': steer_row['robustness_status'].values[0] if not steer_row.empty else None,
                    'baseline_negative_rate': baseline_row['negative_rate'].values[0] if not baseline_row.empty else None,
                    'bad_negative_rate': bad_row['negative_rate'].values[0] if not bad_row.empty else None
                }
                results.append(result)

    # Write longitudinal comparison
    fieldnames = ['lab', 'model', 'generation', 'baseline_score', 'good_persona_score',
                  'bad_persona_score', 'good_delta', 'bad_delta', 'composite_humanescore',
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
