#!/usr/bin/env python3
"""
Create candlestick/range chart showing bidirectional model drift of LLMs.
Shows how far each model drifts toward humane (+) vs harmful (-) behavior.
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Configuration
FIGURE_DIR = 'figures'
MODEL_MAP_PATH = os.path.join(FIGURE_DIR, 'model_display_names.json')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Color scheme (colorblind-friendly)
COLORS = {
    'red': '#DC2626',      # Red for negative drift
    'green': '#16A34A',    # Green for positive drift
    'baseline': '#000000', # Black for baseline dot
    'point_five_line': '#9CA3AF', # Gray for acceptable humaneness threshold
    'zero_line': '#000000', # Black for anti-humane threshold
    'grid': '#E5E7EB',     # Light gray for grid
    'background': '#FFFFFF'
}

def load_model_map(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model display name map not found at {path}")
    with open(path, 'r') as f:
        return json.load(f)


def create_model_drift_chart(compact=False, model_map=None):
    """Create the model drift candlestick chart."""

    # Load data
    df = pd.read_csv('steerability_comparison.csv')

    if model_map is None:
        model_map = load_model_map(MODEL_MAP_PATH)
    df['model'] = df['model'].apply(lambda m: model_map.get(m, m))

    # Sort by bad_persona_score (descending) - robust models at top
    df = df.sort_values('bad_persona_score', ascending=False, na_position='last')

    # Reset index so it matches enumeration positions (0, 1, 2, 3...)
    # This ensures label positioning works correctly
    df = df.reset_index(drop=True)

    # For compact version, take top 10 models
    if compact:
        df = df.head(10)

    n_models = len(df)

    # Create figure with appropriate size
    fig_height = max(8, n_models * 0.6) if not compact else 8
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Set background color
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # Track category boundaries for separators
    robust_end = -1
    moderate_end = -1

    # Plot each model
    for i, (idx, row) in enumerate(df.iterrows()):
        y_pos = n_models - i - 1  # Invert so top is first

        baseline = row['baseline_score']
        good = row['good_persona_score']
        bad = row['bad_persona_score']
        status = row['robustness_status']

        # Track category boundaries
        if status == 'Robust':
            robust_end = y_pos
        elif status == 'Moderate':
            moderate_end = y_pos

        # Draw green bar (baseline → good persona)
        if pd.notna(baseline) and pd.notna(good):
            ax.plot([baseline, good], [y_pos, y_pos],
                   color=COLORS['green'], linewidth=4, solid_capstyle='butt', zorder=2)
            # Add cap at end
            ax.plot([good], [y_pos], marker='|', markersize=10,
                   color=COLORS['green'], markeredgewidth=2, zorder=2)

        # Draw red bar (baseline → bad persona)
        if pd.notna(baseline) and pd.notna(bad):
            ax.plot([bad, baseline], [y_pos, y_pos],
                   color=COLORS['red'], linewidth=4, solid_capstyle='butt', zorder=2)
            # Add cap at end
            ax.plot([bad], [y_pos], marker='|', markersize=10,
                   color=COLORS['red'], markeredgewidth=2, zorder=2)

        # Draw baseline dot (on top)
        if pd.notna(baseline):
            ax.plot([baseline], [y_pos], marker='o', markersize=10,
                   color=COLORS['baseline'], markeredgecolor='white',
                   markeredgewidth=1.5, zorder=3)

    # Add vertical line at zero (harmful threshold)
    ax.axvline(x=0, color=COLORS['zero_line'], linewidth=2,
              linestyle='--', alpha=0.7, zorder=1, label='Harmful Threshold')

    # Add vertical line at +0.5 (acceptable humaneness threshold)
    ax.axvline(x=0.5, color=COLORS['point_five_line'], linewidth=2,
              linestyle='--', alpha=0.7, zorder=1, label='Harmful Threshold')

    # Add category separators
    if robust_end >= 0 and robust_end < n_models - 1:
        ax.axhline(y=robust_end - 0.5, color='#6B7280', linewidth=1,
                  linestyle='-', alpha=0.3, zorder=1)
    if moderate_end >= 0 and moderate_end < n_models - 1:
        ax.axhline(y=moderate_end - 0.5, color='#6B7280', linewidth=1,
                  linestyle='-', alpha=0.3, zorder=1)

    # Configure axes
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.5, n_models - 0.5)

    # X-axis
    ax.set_xlabel('HumaneScore', fontsize=12, fontweight='bold')
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(['-1.0\n(Harmful)', '-0.5', '0.0', '+0.5', '+1.0\n(Humane)'])
    ax.tick_params(axis='x', labelsize=10)

    # Y-axis - model names
    model_labels = df['model'].tolist()
    model_labels.reverse()  # Reverse to match y_pos inversion
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.tick_params(axis='y', length=0)  # Remove tick marks

    # Add category labels on right side
    robust_models = df[df['robustness_status'] == 'Robust']
    moderate_models = df[df['robustness_status'] == 'Moderate']
    failed_models = df[df['robustness_status'] == 'Failed']

    label_x = 1.05  # Position outside plot area
    if len(robust_models) > 0:
        # Position label at middle of category group
        robust_indices = df[df['robustness_status'] == 'Robust'].index.tolist()
        robust_middle = (robust_indices[0] + robust_indices[-1]) / 2.0
        robust_y = n_models - robust_middle - 1
        ax.text(label_x, robust_y, f'✓ Robust ({len(robust_models)})',
               transform=ax.get_yaxis_transform(), fontsize=9,
               color='#059669', fontweight='bold', va='center')

    if len(moderate_models) > 0:
        # Position label at middle of category group
        moderate_indices = df[df['robustness_status'] == 'Moderate'].index.tolist()
        moderate_middle = (moderate_indices[0] + moderate_indices[-1]) / 2.0
        moderate_y = n_models - moderate_middle - 1
        ax.text(label_x, moderate_y, f'⚠ Moderate ({len(moderate_models)})',
               transform=ax.get_yaxis_transform(), fontsize=9,
               color='#D97706', fontweight='bold', va='center')

    if len(failed_models) > 0:
        # Position label at middle of category group
        failed_indices = df[df['robustness_status'] == 'Failed'].index.tolist()
        failed_middle = (failed_indices[0] + failed_indices[-1]) / 2.0
        failed_y = n_models - failed_middle - 1
        ax.text(label_x, failed_y, f'✗ Failed ({len(failed_models)})',
               transform=ax.get_yaxis_transform(), fontsize=9,
               color='#DC2626', fontweight='bold', va='center')

    # Grid
    ax.grid(True, axis='x', alpha=0.2, color=COLORS['grid'], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Generate model drift-specific insights
    improved_count = len(df[df['good_delta'] > 0])
    flip_to_negative_count = len(df[df['bad_persona_score'] < 0])
    avg_good_delta = df['good_delta'].mean()

    # Title and subtitle
    title_text = 'The Model Drift Problem'
    subtitle_text = (
        f'{improved_count} models improve with humane prompts (avg +{avg_good_delta:.2f}), '
        f'but {flip_to_negative_count}/{n_models} flip to harmful behavior under adversarial prompts'
    )

    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    ax.set_title(subtitle_text, fontsize=11, pad=20, color='#374151')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['green'], label='→ Good Persona (humane-aligned prompt)'),
        mpatches.Patch(facecolor=COLORS['baseline'], label='● Baseline (default behavior)'),
        mpatches.Patch(facecolor=COLORS['red'], label='← Bad Persona (adversarial prompt)'),
        mpatches.Patch(facecolor='none', edgecolor=COLORS['zero_line'],
                      linestyle='--', label='| Harmful Threshold (HumaneScore = 0)'),
        mpatches.Patch(facecolor='none', edgecolor=COLORS['point_five_line'],
                      linestyle='--', label='| Acceptable Threshold (HumaneScore = 0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             frameon=True, fancybox=False, shadow=False)

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

def create_alt_text():
    """Generate accessibility alt-text description."""
    df = pd.read_csv('steerability_comparison.csv')
    df = df.sort_values('bad_persona_score', ascending=False, na_position='last')

    robust = df[df['robustness_status'] == 'Robust']
    moderate = df[df['robustness_status'] == 'Moderate']
    failed = df[df['robustness_status'] == 'Failed']

    alt_text = f"""Model drift range chart showing {len(df)} AI models and their response to humane-aligned and adversarial prompts.

The chart displays each model as a horizontal line showing:
- Baseline HumaneScore (black dot): Default behavior without prompting
- Good Persona (green bar extending right): Improvement with humane-aligned prompts
- Bad Persona (red bar extending left): Degradation with adversarial prompts

Key findings:
- All 14 models improve with humane prompts (average +12%)
- 71% of models (10/14) flip to harmful behavior (negative scores) under adversarial prompts
- Only 21% (3/14) maintain positive scores under adversarial pressure: {', '.join(robust['model'].tolist())}

Models are sorted by adversarial robustness:
- Robust models ({len(robust)}): Maintain positive scores under adversarial prompts
- Moderate models ({len(moderate)}): Degrade significantly but remain positive
- Failed models ({len(failed)}): Flip to harmful behavior (negative scores)

The vertical dashed line at HumaneScore=0 marks the threshold between humane and harmful behavior.
"""

    with open(f'{FIGURE_DIR}/model_drift_candlestick_alttext.txt', 'w') as f:
        f.write(alt_text)

    print(f"  ✓ Saved alt-text description")

def main():
    print("\n" + "="*60)
    print("Creating Model Drift Candlestick Charts")
    print("="*60 + "\n")

    # Create full version
    print("Creating full version (14 models)...")
    fig_full, ax_full = create_model_drift_chart(compact=False)
    save_chart(fig_full, 'model_drift_candlestick')
    plt.close(fig_full)

    # Create compact version
    print("\nCreating compact version (10 models)...")
    fig_compact, ax_compact = create_model_drift_chart(compact=True)
    save_chart(fig_compact, 'model_drift_candlestick_compact')
    plt.close(fig_compact)

    # Create alt-text
    print("\nCreating accessibility alt-text...")
    create_alt_text()

    print("\n" + "="*60)
    print("✅ All visualizations created successfully!")
    print("="*60)
    print(f"\nOutputs in {FIGURE_DIR}/:")
    print("  - model_drift_candlestick.png (300 DPI)")
    print("  - model_drift_candlestick.svg (vector)")
    print("  - model_drift_candlestick.pdf (publication)")
    print("  - model_drift_candlestick_compact.png (10 models)")
    print("  - model_drift_candlestick_compact.svg")
    print("  - model_drift_candlestick_compact.pdf")
    print("  - model_drift_candlestick_alttext.txt")
    print()

if __name__ == "__main__":
    main()
