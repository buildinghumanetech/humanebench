# HELM correlation: observed effect and minimum detectable effect

Matched cohort: **n = 13** models. HELM aggregate score vs HumaneBench outcomes. Delta_bad regenerated from the logs via `compute_loo_sensitivity.py` (`ensemble3`), not read from the stale `table1_steerability_summary.csv`.

## Observed correlations

| outcome | Pearson r | 95% CI | p | Spearman rho | p |
| --- | ---: | :---: | ---: | ---: | ---: |
| Delta_bad | +0.248 | [-0.351, +0.703] | 0.415 | +0.231 | 0.448 |
| S_bad | +0.281 | [-0.319, +0.721] | 0.352 | +0.236 | 0.437 |
| S_baseline | +0.351 | [-0.248, +0.756] | 0.239 | +0.434 | 0.138 |

## Minimum detectable effect

Two-tailed, alpha = 0.05, power = 0.8, n = 13.

| method | MDE (|r|) |
| --- | ---: |
| Fisher z approximation | 0.709 |
| Exact t-test, Monte-Carlo (200,000 sims) | 0.690 |

With 13 models this design could only have detected a correlation of about **|r| >= 0.69** -- a very large effect. Its power to detect the correlation actually observed (|r| = 0.248) is **0.13**.

This bounds what the null can be read to mean. The result rules out a *strong* capability-robustness relationship; it does not establish the absence of a moderate one. Section 4.6's phrasing should say so explicitly rather than leaving "no significant correlation" to carry the weight, and should print the observed r and p, which the current text does not.
