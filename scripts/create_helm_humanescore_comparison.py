#!/usr/bin/env python3
"""
Create comprehensive comparisons between HELM capability scores and HumaneScore
across all three personas (baseline, good, bad).

Generates:
- Combined table (HELM + all 3 personas)
- Three separate tables (one per persona)
- Small multiples scatter chart
- Heatmap matrix chart

All outputs in both CSV and Markdown for web/README use.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from matplotlib.colors import TwoSlopeNorm, Normalize

# Configuration
TABLES_DIR = 'tables'
FIGURES_DIR = 'figures'
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# HELM model name -> eval model name mapping
HELM_TO_EVAL_MAP = {
    'OpenAI GPT 5 (2025-08-07)': 'gpt-5',
    'OpenAI GPT 4 1 (2025-04-14)': 'gpt-4.1',
    'OpenAI GPT 4O (2024-11-20)': 'gpt-4o-2024-11-20',
    'Anthropic CLAUDE 4 5 SONNET (2025-09-29)': 'claude-sonnet-4.5',
    'Anthropic CLAUDE 4 SONNET (2025-05-14)': 'claude-sonnet-4',
    'Google Gemini 3 Pro (Preview)': 'gemini-3-pro-preview',
    'Google Gemini 2 5 Pro': 'gemini-2.5-pro',
    'Google Gemini 2.0 Flash 001': 'gemini-2.0-flash-001',
    'Meta Llama 4 Maverick (17Bx128E) Instruct FP8': 'llama-4-maverick',
    'Grok 4 (0709)': 'grok-4',
}

# Model family colors (colorblind-friendly)
MODEL_FAMILIES = {
    'gpt': ('OpenAI', '#3B82F6'),      # Blue
    'claude': ('Anthropic', '#F97316'), # Orange
    'gemini': ('Google', '#10B981'),    # Green
    'llama': ('Meta', '#8B5CF6'),       # Purple
    'grok': ('xAI', '#EF4444'),         # Red
    'deepseek': ('DeepSeek', '#14B8A6') # Teal
}

# Color scheme
COLORS = {
    'grid': '#E5E7EB',
    'background': '#FFFFFF'
}

def get_model_family(model_name):
    """Determine model family from model name."""
    model_lower = model_name.lower()
    for prefix, (family_name, color) in MODEL_FAMILIES.items():
        if prefix in model_lower:
            return family_name, color
    return 'Other', '#6B7280'  # Gray for unknown

def load_and_merge_data():
    """Load and merge HELM and HumaneScore data."""
    print("Loading data...")

    # Load HumaneScore data
    humanescore_df = pd.read_csv(f'{TABLES_DIR}/table1_steerability_summary.csv')

    # Load HELM data
    with open('helm_integration/data/helm_aggregate_scores.json', 'r') as f:
        helm_data = json.load(f)

    helm_df = pd.DataFrame(helm_data['models'])

    # Add eval model name mapping
    helm_df['eval_model'] = helm_df['model_name'].map(HELM_TO_EVAL_MAP)

    # Filter to only models we can map
    helm_df = helm_df[helm_df['eval_model'].notna()].copy()

    # Merge datasets
    merged_df = helm_df.merge(
        humanescore_df,
        left_on='eval_model',
        right_on='Model',
        how='inner'
    )

    # Add family information
    merged_df['family'], merged_df['color'] = zip(*merged_df['eval_model'].map(get_model_family))

    # Sort by HELM score descending
    merged_df = merged_df.sort_values('mean_score', ascending=False)

    print(f"  ✓ Loaded {len(merged_df)} models with both HELM and HumaneScore data")

    # Determine which models are missing
    all_models = set(humanescore_df['Model'])
    matched_models = set(merged_df['eval_model'])
    missing_models = all_models - matched_models

    return merged_df, missing_models

def generate_combined_table(df, missing_models):
    """Generate combined table with all personas."""
    print("\nGenerating combined table...")

    # Select and rename columns
    table_df = df[[
        'model_name',
        'mean_score',
        'Baseline HumaneScore',
        'Good Persona HumaneScore',
        'Bad Persona HumaneScore',
    ]].copy()

    table_df.columns = [
        'Model',
        'HELM Score',
        'Baseline HS',
        'Good HS',
        'Bad HS',
    ]

    # Round scores to 3 decimal places
    for col in ['HELM Score', 'Baseline HS', 'Good HS', 'Bad HS']:
        table_df[col] = table_df[col].round(3)

    # Save CSV
    csv_path = f'{TABLES_DIR}/helm_humanescore_combined.csv'
    table_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved {csv_path}")

    # Save Markdown
    md_path = f'{TABLES_DIR}/helm_humanescore_combined.md'
    with open(md_path, 'w') as f:
        f.write("# HELM Capability vs HumaneScore (All Personas)\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write(f"**Note:** Models without HELM capability scores (excluded): {', '.join(sorted(missing_models))}\n")
    print(f"  ✓ Saved {md_path}")

    return table_df

def generate_persona_tables(df, missing_models):
    """Generate three separate tables (one per persona)."""
    print("\nGenerating persona-specific tables...")

    personas = [
        ('baseline', 'Baseline HumaneScore', 'Baseline HS'),
        ('good_persona', 'Good Persona HumaneScore', 'Good HS'),
        ('bad_persona', 'Bad Persona HumaneScore', 'Bad HS')
    ]

    for persona_name, col_name, short_name in personas:
        # Select columns
        table_df = df[['model_name', 'mean_score', col_name]].copy()
        table_df.columns = ['Model', 'HELM Score', short_name]

        # Round scores
        table_df['HELM Score'] = table_df['HELM Score'].round(3)
        table_df[short_name] = table_df[short_name].round(3)

        # Save CSV
        csv_path = f'{TABLES_DIR}/helm_vs_{persona_name}.csv'
        table_df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {csv_path}")

        # Save Markdown
        md_path = f'{TABLES_DIR}/helm_vs_{persona_name}.md'
        with open(md_path, 'w') as f:
            persona_title = persona_name.replace('_', ' ').title()
            f.write(f"# HELM Capability vs {persona_title}\n\n")
            f.write(table_df.to_markdown(index=False))
            f.write("\n\n---\n\n")
            f.write(f"**Note:** Models without HELM capability scores (excluded): {', '.join(sorted(missing_models))}\n")
        print(f"  ✓ Saved {md_path}")

def create_scatter_chart(df):
    """Create small multiples scatter chart (3 panels)."""
    print("\nCreating small multiples scatter chart...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.patch.set_facecolor(COLORS['background'])

    personas = [
        ('Baseline HumaneScore', 'Baseline', axes[0]),
        ('Good Persona HumaneScore', 'Good Persona', axes[1]),
        ('Bad Persona HumaneScore', 'Bad Persona', axes[2])
    ]

    # Track families for legend (only add once)
    families_plotted = set()

    for col_name, title, ax in personas:
        ax.set_facecolor(COLORS['background'])

        # Plot points by family
        for idx, row in df.iterrows():
            family = row['family']
            color = row['color']

            # Only add to legend on first panel
            label = family if (family not in families_plotted and ax == axes[0]) else None
            if label:
                families_plotted.add(family)

            ax.scatter(
                row[col_name],
                row['mean_score'],
                color=color,
                s=120,
                alpha=0.8,
                edgecolors='white',
                linewidth=1.5,
                label=label,
                zorder=3
            )

        # Configure axes
        ax.set_xlabel('HumaneScore', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Y-axis label only on first panel
    axes[0].set_ylabel('HELM Capability Score', fontsize=11, fontweight='bold')

    # Legend on first panel
    axes[0].legend(loc='lower right', fontsize=9, frameon=True, fancybox=False, shadow=False)

    # Main title
    fig.suptitle('HELM Capability vs HumaneScore Across Personas',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save in multiple formats
    for ext in ['png', 'svg', 'pdf']:
        path = f'{FIGURES_DIR}/helm_vs_humanescore_scatter.{ext}'
        dpi = 300 if ext == 'png' else None
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved {path}")

    plt.close(fig)

def create_heatmap_chart(df):
    """Create heatmap matrix chart."""
    print("\nCreating heatmap matrix chart...")

    # Prepare data for heatmap
    heatmap_data = df[[
        'model_name',
        'mean_score',
        'Baseline HumaneScore',
        'Good Persona HumaneScore',
        'Bad Persona HumaneScore',
    ]].copy()

    heatmap_data.columns = ['Model', 'HELM Capability', 'Baseline HS', 'Good Persona HS', 'Bad Persona HS']
    heatmap_data = heatmap_data.set_index('Model')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])

    # Create color matrix manually (RGBA)
    n_rows, n_cols = heatmap_data.shape
    colors = np.zeros((n_rows, n_cols, 4))

    # Use RdYlGn for all columns
    cmap = plt.cm.RdYlGn

    # Fixed scales per column type
    helm_norm = Normalize(vmin=0, vmax=1)
    humanescore_norm = Normalize(vmin=-1, vmax=1)

    for i in range(n_rows):
        # Column 0: HELM (0-1 scale)
        colors[i, 0] = cmap(helm_norm(heatmap_data.iloc[i, 0]))

        # Columns 1-3: HumaneScore (-1 to 1 scale)
        for j in range(1, n_cols):
            colors[i, j] = cmap(humanescore_norm(heatmap_data.iloc[i, j]))

    # Display
    im = ax.imshow(colors, aspect='auto')

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}',
                   ha="center", va="center", color="black", fontsize=9)

    # Add grid lines
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)

    # Configure axes
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(heatmap_data.columns, fontsize=10, fontweight='bold')
    ax.set_yticklabels(heatmap_data.index, fontsize=9, rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Title
    ax.set_title('Smarter Doesn\'t Mean More Humane',
                fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()

    # Save in multiple formats
    for ext in ['png', 'svg', 'pdf']:
        path = f'{FIGURES_DIR}/helm_vs_humanescore_heatmap.{ext}'
        dpi = 300 if ext == 'png' else None
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved {path}")

    # Also save as helm_humanebench_comparison.png (referenced by website)
    fig.savefig(f'{FIGURES_DIR}/helm_humanebench_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved {FIGURES_DIR}/helm_humanebench_comparison.png")

    plt.close(fig)

def main():
    print("\n" + "="*60)
    print("HELM Capability vs HumaneScore Comparison")
    print("="*60)

    # Load and merge data
    df, missing_models = load_and_merge_data()

    # Generate tables
    combined_table = generate_combined_table(df, missing_models)
    generate_persona_tables(df, missing_models)

    # Generate charts
    create_scatter_chart(df)
    create_heatmap_chart(df)

    print("\n" + "="*60)
    print("✅ All visualizations created successfully!")
    print("="*60)
    print(f"\nGenerated {len(df)} model comparisons")
    print(f"\nTables in {TABLES_DIR}/:")
    print("  - helm_humanescore_combined.{csv,md}")
    print("  - helm_vs_baseline.{csv,md}")
    print("  - helm_vs_good_persona.{csv,md}")
    print("  - helm_vs_bad_persona.{csv,md}")
    print(f"\nCharts in {FIGURES_DIR}/:")
    print("  - helm_vs_humanescore_scatter.{png,svg,pdf}")
    print("  - helm_vs_humanescore_heatmap.{png,svg,pdf}")
    print()

if __name__ == "__main__":
    main()
