# HELM Safety × HumaneBench adversarial drop

Cohort: n = 13 (DeepSeek V3.1 and Claude Opus 4.1 expected-absent per the HELM Safety release).

drop_mag = baseline HumaneScore − bad-persona HumaneScore (positive = bigger drop). The hypothesis predicts a **negative** ρ.

## Primary

| statistic | estimate | 95% CI (bootstrap, n=10,000) | p (perm, n=10,000) |
|-----------|----------|----------------------------------|------------------------|
| Spearman ρ | -0.616 | [-0.894, -0.040] | 0.0268 |
| Pearson r  | -0.663   | [-0.889, -0.344]     | 0.0109 |

## Leave-one-out robustness (Spearman ρ)

min / median / max across 13 LOO folds: -0.715 / -0.620 / -0.525

## Partial Spearman ρ(Safety, drop_mag | Capability)

n = 13 (subset with HELM Capability score).

| statistic | estimate | 95% CI | p (perm) |
|-----------|----------|--------|----------|
| ρ_partial | -0.587 | [-0.893, +0.118] | 0.0464 |

LOO range: -0.697 / -0.605 / -0.453

✓ |ρ_partial| < |ρ| — capability absorbs some of the signal (consistent with 'necessary but not sufficient').

## Caveats

- **gpt-4o-2024-11-20** — proxy: HELM Safety release only has 2024-05-13; paired with HumaneBench's 2024-11-20 baseline/bad scores

## Cohort

See `tables/helm_safety_robustness_merged.csv` for per-model values.
