#!/usr/bin/env python3
"""
Generate comprehensive tables for HumaneBench results writeup.
Creates both CSV and Markdown formatted tables.
"""

import pandas as pd
import numpy as np

PRINCIPLES = [
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion"
]

PRINCIPLE_LABELS = {
    "respect-user-attention": "Respect Attention",
    "enable-meaningful-choices": "Enable Choices",
    "enhance-human-capabilities": "Enhance Capabilities",
    "protect-dignity-and-safety": "Protect Safety",
    "foster-healthy-relationships": "Foster Relationships",
    "prioritize-long-term-wellbeing": "Long-term Wellbeing",
    "be-transparent-and-honest": "Be Transparent",
    "design-for-equity-and-inclusion": "Equity & Inclusion"
}

# Standard model ordering (alphabetical)
MODEL_ORDER = [
    "claude-opus-4.1",
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "deepseek-v3.1-terminus",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gpt-4.1",
    "gpt-4o-2024-11-20",
    "gpt-5",
    "gpt-5.1",
    "grok-4",
    "llama-3.1-405b-instruct",
    "llama-4-maverick"
]

def format_score(val):
    """Format score for display."""
    if pd.isna(val):
        return "N/A"
    return f"{val:.3f}"

def format_delta(val):
    """Format delta for display with +/- sign."""
    if pd.isna(val):
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.3f}"

def create_table1_steerability_summary():
    """Table 1: Comprehensive Steerability Summary."""
    print("Creating Table 1: Steerability Summary...")

    steer = pd.read_csv('steerability_comparison.csv')

    # Apply standard model ordering
    steer['model'] = pd.Categorical(steer['model'], categories=MODEL_ORDER, ordered=True)
    steer = steer.sort_values('model')

    # Create formatted table
    table_data = []
    for _, row in steer.iterrows():
        table_data.append({
            'Model': row['model'],
            'Baseline HumaneScore': format_score(row['baseline_score']),
            'Good Persona HumaneScore': format_score(row['good_persona_score']),
            'Good Δ': format_delta(row['good_delta']),
            'Bad Persona HumaneScore': format_score(row['bad_persona_score']),
            'Bad Δ': format_delta(row['bad_delta']),
            'Composite HumaneScore': format_score(row['composite_humanescore']),
            'Robustness Status': row['robustness_status'] if pd.notna(row['robustness_status']) else 'N/A'
        })

    df = pd.DataFrame(table_data)

    # Save as CSV
    df.to_csv('tables/table1_steerability_summary.csv', index=False)

    # Save as Markdown
    with open('tables/table1_steerability_summary.md', 'w') as f:
        f.write("# Table 1: Comprehensive Steerability Summary\n\n")
        f.write("Shows baseline performance, response to humane-aligned prompting (Good Persona), ")
        f.write("adversarial robustness (Bad Persona), and Composite HumaneScore (mean of all 3 personas) ")
        f.write("across all 13 models.\n\n")
        f.write(df.to_markdown(index=False))

    print(f"  Saved table1_steerability_summary (13 models)")
    return df

def create_table2_baseline_heatmap():
    """Table 2: Baseline Performance Heatmap."""
    print("Creating Table 2: Baseline Performance Heatmap...")

    baseline = pd.read_csv('baseline_scores.csv')

    # Apply standard model ordering
    baseline['model'] = pd.Categorical(baseline['model'], categories=MODEL_ORDER, ordered=True)
    baseline = baseline.sort_values('model')

    # Create heatmap data with HumaneScore first
    table_data = []
    for _, row in baseline.iterrows():
        data = {
            'Model': row['model'],
            'HumaneScore': format_score(row['overall'])
        }
        for principle in PRINCIPLES:
            data[PRINCIPLE_LABELS[principle]] = format_score(row[principle])
        table_data.append(data)

    df = pd.DataFrame(table_data)

    # Save as CSV
    df.to_csv('tables/table2_baseline_heatmap.csv', index=False)

    # Save as Markdown
    with open('tables/table2_baseline_heatmap.md', 'w') as f:
        f.write("# Table 2: Baseline Performance Heatmap\n\n")
        f.write("Per-principle and overall humaneness scores for all 13 models in baseline condition.\n\n")
        f.write(df.to_markdown(index=False))

    print(f"  Saved table2_baseline_heatmap (13 models × 9 columns)")
    return df

def create_table3_good_persona_heatmap():
    """Table 3: Good Persona Performance Heatmap."""
    print("Creating Table 3: Good Persona Performance Heatmap...")

    good = pd.read_csv('good_persona_scores.csv')

    # Apply standard model ordering
    good['model'] = pd.Categorical(good['model'], categories=MODEL_ORDER, ordered=True)
    good = good.sort_values('model')

    # Create heatmap data with HumaneScore first
    table_data = []
    for _, row in good.iterrows():
        data = {
            'Model': row['model'],
            'HumaneScore': format_score(row['overall'])
        }
        for principle in PRINCIPLES:
            data[PRINCIPLE_LABELS[principle]] = format_score(row[principle])
        table_data.append(data)

    df = pd.DataFrame(table_data)

    # Save as CSV
    df.to_csv('tables/table3_good_persona_heatmap.csv', index=False)

    # Save as Markdown
    with open('tables/table3_good_persona_heatmap.md', 'w') as f:
        f.write("# Table 3: Good Persona Performance Heatmap\n\n")
        f.write("Per-principle and overall humaneness scores with humane-aligned system prompts.\n\n")
        f.write(df.to_markdown(index=False))

    print(f"  Saved table3_good_persona_heatmap (13 models × 9 columns)")
    return df

def create_table4_bad_persona_heatmap():
    """Table 4: Bad Persona (Adversarial) Performance Heatmap."""
    print("Creating Table 4: Bad Persona Performance Heatmap...")

    bad = pd.read_csv('bad_persona_scores.csv')

    # Apply standard model ordering
    bad['model'] = pd.Categorical(bad['model'], categories=MODEL_ORDER, ordered=True)
    bad = bad.sort_values('model')

    # Create heatmap data with HumaneScore first
    table_data = []
    for _, row in bad.iterrows():
        data = {
            'Model': row['model'],
            'HumaneScore': format_score(row['overall'])
        }
        for principle in PRINCIPLES:
            data[PRINCIPLE_LABELS[principle]] = format_score(row[principle])
        table_data.append(data)

    df = pd.DataFrame(table_data)

    # Save as CSV
    df.to_csv('tables/table4_bad_persona_heatmap.csv', index=False)

    # Save as Markdown
    with open('tables/table4_bad_persona_heatmap.md', 'w') as f:
        f.write("# Table 4: Bad Persona (Adversarial) Performance Heatmap\n\n")
        f.write("Per-principle and overall humaneness scores under adversarial system prompts.\n\n")
        f.write(df.to_markdown(index=False))

    print(f"  Saved table4_bad_persona_heatmap (13 models × 9 columns)")
    return df

def create_table5_longitudinal_comparison():
    """Table 5: Longitudinal Comparison Across Labs."""
    print("Creating Table 5: Longitudinal Comparison...")

    long = pd.read_csv('longitudinal_comparison.csv')

    # Create formatted table
    table_data = []
    for _, row in long.iterrows():
        # Calculate composite from the three persona scores
        b = row['baseline_score']
        g = row['good_persona_score']
        a = row['bad_persona_score']
        if all(pd.notna(v) for v in [b, g, a]):
            composite = round((b + g + a) / 3, 3)
        else:
            composite = None

        table_data.append({
            'Lab': row['lab'],
            'Model': row['model'],
            'Generation': row['generation'],
            'Baseline HumaneScore': format_score(row['baseline_score']),
            'Good Persona HumaneScore': format_score(row['good_persona_score']),
            'Bad Persona HumaneScore': format_score(row['bad_persona_score']),
            'Good Δ': format_delta(row['good_delta']),
            'Bad Δ': format_delta(row['bad_delta']),
            'Composite HumaneScore': format_score(composite),
            'Robustness Status': row['robustness_status'] if pd.notna(row['robustness_status']) else 'N/A'
        })

    df = pd.DataFrame(table_data)

    # Save as CSV
    df.to_csv('tables/table5_longitudinal_comparison.csv', index=False)

    # Save as Markdown
    with open('tables/table5_longitudinal_comparison.md', 'w') as f:
        f.write("# Table 5: Longitudinal Comparison Across Labs\n\n")
        f.write("Tracking humaneness and adversarial robustness across model generations ")
        f.write("from Anthropic, OpenAI, Google, and Meta.\n\n")
        f.write(df.to_markdown(index=False))

    print(f"  Saved table5_longitudinal_comparison ({len(df)} model generations)")
    return df

def main():
    # Create tables directory if it doesn't exist
    import os
    os.makedirs('tables', exist_ok=True)

    print("\nGenerating all tables for HumaneBench writeup...\n")

    # Generate all tables
    table1 = create_table1_steerability_summary()
    table2 = create_table2_baseline_heatmap()
    table3 = create_table3_good_persona_heatmap()
    table4 = create_table4_bad_persona_heatmap()
    table5 = create_table5_longitudinal_comparison()

    print("\n✅ All tables generated successfully in tables/ directory")
    print("   - CSV format for data analysis")
    print("   - Markdown format for writeup inclusion")

if __name__ == "__main__":
    main()
