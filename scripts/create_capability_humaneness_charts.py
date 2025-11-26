#!/usr/bin/env python3
"""
Create scatter plots comparing model capability (HELM) with humaneness metrics.
Shows relationship between general capability and both baseline humaneness
and adversarial robustness.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

# Configuration
FIGURE_DIR = 'figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

# Model name mapping: HELM format -> eval format
MODEL_NAME_MAP = {
    'GPT-5.1': 'gpt-5.1',
    'GPT-5': 'gpt-5',
    'GPT-4.1': 'gpt-4.1',
    'GPT-4o': 'gpt-4o-2024-11-20',
    'Claude Sonnet 4.5': 'claude-sonnet-4.5',
    'Claude Sonnet 4': 'claude-sonnet-4',
    'Claude Opus 4.1': 'claude-opus-4.1',
    'Gemini 3 Pro Preview': 'gemini-3-pro-preview',
    'Gemini 2.5 Pro': 'gemini-2.5-pro',
    'Gemini 2.5 Flash': 'gemini-2.5-flash',
    'Gemini 2.0 Flash': 'gemini-2.0-flash-001',
    'Llama 3.1 405B': 'llama-3.1-405b-instruct',
    'Llama 4': 'llama-4-maverick',
    'DeepSeek V3': 'deepseek-v3.1-terminus',
    'Grok-4': 'grok-4',
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

# Color scheme (shared)
COLORS = {
    'grid': '#E5E7EB',     # Light gray for grid
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
    """Load and merge data from all sources."""
    # Load HELM capability data
    helm_df = pd.read_csv('helm_integration/output/capability_percentiles.csv')

    # Load baseline humaneness data
    baseline_df = pd.read_csv('baseline_scores.csv')

    # Load steerability data
    steerability_df = pd.read_csv('steerability_comparison.csv')

    # Add normalized model names to HELM data
    helm_df['model_normalized'] = helm_df['model_name'].map(MODEL_NAME_MAP)

    # Filter to only models with HELM scores
    helm_df = helm_df[helm_df['in_helm'] == True].copy()

    # Merge HELM with baseline
    merged_df = helm_df.merge(
        baseline_df[['model', 'overall']],
        left_on='model_normalized',
        right_on='model',
        how='inner'
    )

    # Merge with steerability
    merged_df = merged_df.merge(
        steerability_df[['model', 'bad_delta']],
        on='model',
        how='inner'
    )

    # Add family information
    merged_df['family'], merged_df['color'] = zip(*merged_df['model_name'].map(get_model_family))

    return merged_df

def create_capability_chart(df, y_column, y_label, title, subtitle, invert_y=False):
    """Create a capability vs humaneness scatter plot."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Set background color
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # Plot points by family
    families_plotted = set()
    for idx, row in df.iterrows():
        family = row['family']
        color = row['color']

        # Only add to legend once per family
        label = family if family not in families_plotted else None
        if family not in families_plotted:
            families_plotted.add(family)

        ax.scatter(
            row['helm_aggregate_score'],
            row[y_column],
            color=color,
            s=150,
            alpha=0.8,
            edgecolors='white',
            linewidth=1.5,
            label=label,
            zorder=3
        )

    # Add labels for all points
    for idx, row in df.iterrows():
        # Smart label positioning to reduce overlap
        # Offset alternates based on index for simple spread
        offset_x = 0.005 if idx % 2 == 0 else -0.005
        offset_y = 0.01 if idx % 3 == 0 else -0.01

        ax.annotate(
            row['model_name'],
            (row['helm_aggregate_score'], row[y_column]),
            xytext=(offset_x, offset_y),
            textcoords='offset fontsize',
            fontsize=9,
            ha='left' if idx % 2 == 0 else 'right',
            va='bottom' if idx % 3 == 0 else 'top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='none', alpha=0.7),
            zorder=4
        )

    # Configure axes
    ax.set_xlabel('HELM Aggregate Score', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Invert y-axis if requested (for bad_delta where lower is better)
    if invert_y:
        ax.invert_yaxis()
        # Set explicit limits with 0 at top (after inversion)
        if y_column == 'bad_delta':
            y_min = df[y_column].min()
            ax.set_ylim(y_min, 0)  # After inversion: most_negative at bottom, 0 at top

    # Grid
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title and subtitle
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    ax.set_title(subtitle, fontsize=11, pad=20, color='#374151')

    # Legend
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False, shadow=False)

    # Add annotation about missing models
    missing_models = "GPT-5.1, Claude Opus 4.1, Gemini 2.5 Flash, Llama 3.1 405B, DeepSeek V3"
    fig.text(0.5, 0.02, f'Models without HELM capability scores (not plotted): {missing_models}',
             ha='center', fontsize=8, color='#6B7280', style='italic')

    # Adjust layout
    plt.tight_layout()

    return fig, ax

def save_chart(fig, filename_base):
    """Save chart in multiple formats."""
    print(f"Saving {filename_base}...")

    # PNG (high resolution)
    fig.savefig(f'{FIGURE_DIR}/{filename_base}.png',
               dpi=300, bbox_inches='tight', facecolor='white')

    # SVG (vector)
    fig.savefig(f'{FIGURE_DIR}/{filename_base}.svg',
               bbox_inches='tight', facecolor='white')

    # PDF (publication)
    fig.savefig(f'{FIGURE_DIR}/{filename_base}.pdf',
               bbox_inches='tight', facecolor='white')

    print(f"  ✓ Saved PNG (300 DPI)")
    print(f"  ✓ Saved SVG (vector)")
    print(f"  ✓ Saved PDF (publication)")

def main():
    print("\n" + "="*60)
    print("Creating Capability vs Humaneness Charts")
    print("="*60 + "\n")

    # Load and merge data
    print("Loading and merging data...")
    df = load_and_merge_data()
    print(f"  ✓ Loaded {len(df)} models with HELM capability scores\n")

    # Chart 1: Capability vs Baseline HumaneScore
    print("Creating Chart 1: Capability vs Baseline Humaneness...")
    fig1, ax1 = create_capability_chart(
        df,
        y_column='overall',
        y_label='Baseline HumaneScore',
        title='Model Capability vs Baseline Humaneness',
        subtitle='Higher capability models tend to exhibit higher baseline humaneness',
        invert_y=False
    )
    save_chart(fig1, 'capability_vs_baseline_humaneness')
    plt.close(fig1)

    # Chart 2: Capability vs Adversarial Robustness
    print("\nCreating Chart 2: Capability vs Adversarial Robustness...")
    fig2, ax2 = create_capability_chart(
        df,
        y_column='bad_delta',
        y_label='Adversarial Robustness (bad_delta, lower is better)',
        title='Model Capability vs Adversarial Robustness',
        subtitle='Higher capability models show better resistance to adversarial prompts (less negative degradation)',
        invert_y=True
    )
    save_chart(fig2, 'capability_vs_adversarial_robustness')
    plt.close(fig2)

    print("\n" + "="*60)
    print("✅ All visualizations created successfully!")
    print("="*60)
    print(f"\nOutputs in {FIGURE_DIR}/:")
    print("  - capability_vs_baseline_humaneness.png (300 DPI)")
    print("  - capability_vs_baseline_humaneness.svg (vector)")
    print("  - capability_vs_baseline_humaneness.pdf (publication)")
    print("  - capability_vs_adversarial_robustness.png (300 DPI)")
    print("  - capability_vs_adversarial_robustness.svg (vector)")
    print("  - capability_vs_adversarial_robustness.pdf (publication)")
    print()

if __name__ == "__main__":
    main()
