#!/usr/bin/env python3
"""
Render the HELM × Δ_bad scatter for §4.6 of the AAAI paper.

Single-panel chart: HELM aggregate capability (x) vs adversarial
degradation Δ_bad = S_bad − S_baseline (y), for the n=10 HELM-matched
models. The absence of a clean positive slope is §4.6's visual claim
("Intelligence ≠ Humaneness").

Output: paper_notes/latex/figs/fig_helm_scatter_1col.pdf

Run from repo root:
    python scripts/create_aaai_helm_scatter.py
"""

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 7

OUT_DIR = 'paper_notes/latex/figs'
OUT_FILE = 'fig_helm_scatter_1col.pdf'

HELM_TO_EVAL = {
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

FAMILY_COLORS = {
    'gpt': ('OpenAI', '#3B82F6'),
    'claude': ('Anthropic', '#F97316'),
    'gemini': ('Google', '#10B981'),
    'llama': ('Meta', '#8B5CF6'),
    'grok': ('xAI', '#EF4444'),
    'deepseek': ('DeepSeek', '#14B8A6'),
}

# Standardized label placement: every label sits either directly ABOVE
# or directly BELOW its marker, with a thin gray leader line connecting
# the two. ABOVE is the default; the entries below override to BELOW for
# close-y pairs that would otherwise share a row.
ABOVE = ('center', 'bottom',  0,  10)
BELOW = ('center', 'top',     0, -10)

LABEL_DIRECTION = {
    # Claude Sonnet 4.5 sits near the top-left where the r/p annotation
    # box lives; flip BELOW to clear it.
    'claude-sonnet-4.5':     BELOW,
    # Bottom cluster (Δ_bad ≲ -1): Gemini 2.0 Flash / GPT-4.1 / Gemini
    # 2.5 Pro / Grok 4 all sit near y ≈ -1.3 to -1.5. Alternate the two
    # Gemini models BELOW so their labels don't collide with the
    # ABOVE-labelled GPT-4.1 / Grok 4 / Gemini 3 Pro Preview.
    'gemini-2.0-flash-001':  BELOW,
    'gemini-2.5-pro':        BELOW,
}


def get_family(slug):
    for prefix, (fam, color) in FAMILY_COLORS.items():
        if prefix in slug.lower():
            return fam, color
    return 'Other', '#6B7280'


def significance(p):
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def load_merged():
    with open('helm_integration/data/helm_aggregate_scores.json') as f:
        helm = pd.DataFrame(json.load(f)['models'])
    helm['eval_model'] = helm['model_name'].map(HELM_TO_EVAL)
    helm = helm[helm['eval_model'].notna()].copy()

    summary = pd.read_csv('tables/table1_steerability_summary.csv')
    summary = summary.rename(columns={
        'Model': 'eval_model',
        'Baseline HumaneScore': 'baseline',
        'Good Persona HumaneScore': 'good',
        'Bad Persona HumaneScore': 'bad',
    })

    df = helm.merge(summary[['eval_model', 'baseline', 'good', 'bad']],
                    on='eval_model', how='inner')
    df['delta_bad'] = df['bad'] - df['baseline']
    df['family'], df['color'] = zip(*df['eval_model'].map(get_family))

    with open('figures/model_display_names.json') as f:
        names = json.load(f)
    label_overrides = {
        'gpt-4o-2024-11-20': 'GPT-4o',
        'gemini-2.0-flash-001': 'Gemini 2.0 Flash',
        'llama-4-maverick': 'Llama 4 Maverick',
    }
    df['display'] = df['eval_model'].map(
        lambda s: label_overrides.get(s, names.get(s, s)))
    return df


def render(df, out_path):
    fig, ax = plt.subplots(figsize=(3.3, 2.8))

    families_seen = set()
    for _, row in df.iterrows():
        fam = row['family']
        label = fam if fam not in families_seen else None
        families_seen.add(fam)
        ax.scatter(row['mean_score'], row['delta_bad'],
                   c=row['color'], s=28, zorder=3,
                   edgecolors='white', linewidths=0.5, label=label)
        ha, va, dx_pts, dy_pts = LABEL_DIRECTION.get(row['eval_model'], ABOVE)
        ax.annotate(row['display'],
                    xy=(row['mean_score'], row['delta_bad']),
                    xytext=(dx_pts, dy_pts), textcoords='offset points',
                    fontsize=5.5, ha=ha, va=va,
                    color='#1f2937', zorder=4,
                    arrowprops=dict(arrowstyle='-', color='#6B7280',
                                    lw=0.5, alpha=0.85,
                                    shrinkA=2.5, shrinkB=1.0))

    x = df['mean_score'].values
    y = df['delta_bad'].values
    slope, intercept, r, p, _ = sp_stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, 'k--', lw=0.8, alpha=0.5, zorder=2)

    # Reference line at Δ_bad = 0 ("no degradation").
    ax.axhline(0, color='#9CA3AF', lw=0.5, ls=':', alpha=0.7, zorder=1)

    sig = significance(p)
    ax.text(0.04, 0.96,
            f'$r = {r:+.2f}$, $p = {p:.3f}$ ({sig})',
            transform=ax.transAxes, fontsize=6.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#9CA3AF', linewidth=0.5))

    ax.set_xlabel('HELM Aggregate Score')
    ax.set_ylabel(r'$\Delta_{\mathrm{bad}} = S^{\mathrm{(bad)}} - S^{\mathrm{(base)}}$')
    ax.grid(True, alpha=0.18, lw=0.5)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    span_x = df['mean_score'].max() - df['mean_score'].min()
    ax.set_xlim(df['mean_score'].min() - span_x * 0.12,
                df['mean_score'].max() + span_x * 0.18)
    span_y = df['delta_bad'].max() - df['delta_bad'].min()
    ax.set_ylim(df['delta_bad'].min() - span_y * 0.18,
                df['delta_bad'].max() + span_y * 0.20)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=len(labels),
               frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return r, p


def main():
    if not os.path.exists('helm_integration/data/helm_aggregate_scores.json'):
        print('ERROR: run from the humanebench repo root.', file=sys.stderr)
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_merged()
    print(f'matched n = {len(df)} models')

    out_path = os.path.join(OUT_DIR, OUT_FILE)
    r, p = render(df, out_path)
    print(f'wrote {out_path}')
    print(f'HELM × Δ_bad: r = {r:+.3f}, p = {p:.3f} ({significance(p)})')


if __name__ == '__main__':
    main()
