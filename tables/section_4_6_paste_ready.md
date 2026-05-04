# §4.6 (Vulnerable-Population Breakdown) — Paste-Ready Numbers

Single source of truth for the §4.6 paragraph. All numbers come from `tables/vp_breakdown.{csv,md}`, `tables/vp_robustness_gap.{csv,md}`, and `tables/vp_breakdown_full.csv` — produced by `scripts/compute_vp_stats.py`. Cluster-bootstrap CIs at the scenario level (cluster: `sample_id`, seed 20260407, n_bootstrap=1000, percentile method). Default-excludes 12 confabulation-flagged items via `humanebench.excluded.load_excluded_ids()` for a working n of **788** scenarios.

---

## Headline DiD

**General-audience HumaneScore drops **more** than VP under adversarial prompting (DiD = +0.0482, 95% CI [+0.0121, +0.0872]).**

Decomposition (item-pooled HumaneScore on `{-1, -0.5, +0.5, +1}`):

| component | value | 95% CI |
| --- | ---: | --- |
| Δ general (bad − baseline)      | -0.8648 | [-0.8887, -0.8405] |
| Δ VP pooled (bad − baseline)    | -0.8165 | [-0.8433, -0.7861] |
| **DiD** (Δ_VP − Δ_general)      | **+0.0482** | **[+0.0121, +0.0872]** |

Sample sizes: general n_scenarios = 520 (n_items = 15,588); VP_pooled n_scenarios = 268 (n_items = 8,030).

---

## Per-stratum HumaneScore by persona

Item-pooled across 15 models. Full CIs in `tables/vp_breakdown.md`.

| stratum | n_scen | baseline | good_persona | bad_persona |
| --- | ---: | ---: | ---: | ---: |
| general | 520 | +0.671 | +0.814 | -0.194 |
| teenagers | 68 | +0.824 | +0.894 | -0.029 |
| elderly | 46 | +0.891 | +0.904 | -0.056 |
| children | 40 | +0.750 | +0.850 | -0.008 |
| people-with-disabilities | 24 | +0.917 | +0.936 | +0.120 |
| other-VP | 90 | +0.772 | +0.843 | +0.019 |

---

## Per-stratum baseline→bad erosion (top 3 non-general)

- **elderly** (n=46): ordinal gap +0.947 [+0.898, +1.001]; prosocial-rate gap +0.578 [+0.544, +0.611]
- **teenagers** (n=68): ordinal gap +0.853 [+0.794, +0.909]; prosocial-rate gap +0.502 [+0.460, +0.536]
- **people-with-disabilities** (n=24): ordinal gap +0.797 [+0.728, +0.865]; prosocial-rate gap +0.503 [+0.453, +0.555]

Full per-stratum gaps (with prosocial-rate gap CIs and gap ratios) in `tables/vp_robustness_gap.md`.

---

## Caveat: `other-VP` is not a coherent subpopulation

The `other-VP` stratum pools 13 distinct groups (non-native-speakers, low-tech-literacy, women, low-income-communities, neurodivergent-people, gender-diverse-people, low-literacy-users, marginalized-groups, low-connectivity-users, religious-minorities, transgender-people, refugees, shift-workers) for statistical power on the DiD, **not** because those groups share behavior profiles. For per-group detail see the appendix table `tables/vp_breakdown_full.csv`.

## Figure references for §4.6 LaTeX

- Combined two-up bad-persona heatmap (Children + Teenagers): `figures/vp_heatmap_combined_children_teenagers_bad_persona.png`
- Single-VP heatmaps (children/teenagers/elderly × baseline/good/bad): `figures/vp_heatmap_<vp>_<persona>.png` (9 figures, restyled)
- VP comparison dot chart: `figures/vp_dot_chart_comparison.png`
- VP grouped bar chart: `figures/vp_grouped_bar_comparison.png`
