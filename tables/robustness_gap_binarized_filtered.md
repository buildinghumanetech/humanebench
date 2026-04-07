# Robustness-gap ratio sensitivity to small-gap models

The unfiltered mean ratio of `binarized_gap / ordinal_gap` across all 15 models is **+0.589**, which tripped the < 0.7 warning threshold. This number is mathematically unstable for models with very small ordinal gaps (e.g. gpt-5.1 has ord_gap = 0.091 and ratio = 0.700 because the denominator is near zero). This file recomputes under filters and reports both mean and median.

## Per-model ratios

Sorted by ordinal gap (descending). Ratio = `binarized_gap / ordinal_gap`.

| model | ord_gap | bin_gap | ratio |
| --- | ---: | ---: | ---: |
| gemini-2.5-pro | +1.626 | +0.947 | +0.583 |
| grok-4 | +1.621 | +0.961 | +0.593 |
| gemini-2.0-flash-001 | +1.524 | +0.940 | +0.617 |
| gemini-2.5-flash | +1.447 | +0.921 | +0.636 |
| gpt-4.1 | +1.425 | +0.906 | +0.636 |
| gpt-4o-2024-11-20 | +1.383 | +0.911 | +0.659 |
| gemini-3-pro-preview | +1.374 | +0.711 | +0.518 |
| deepseek-v3.1-terminus | +1.193 | +0.766 | +0.642 |
| llama-3.1-405b-instruct | +1.161 | +0.781 | +0.673 |
| llama-4-maverick | +0.798 | +0.533 | +0.668 |
| claude-sonnet-4 | +0.342 | +0.189 | +0.552 |
| claude-opus-4.1 | +0.185 | +0.096 | +0.521 |
| claude-sonnet-4.5 | +0.119 | +0.037 | +0.315 |
| gpt-5 | +0.111 | +0.058 | +0.518 |
| gpt-5.1 | +0.091 | +0.064 | +0.700 |

## Sensitivity summary

| filter | n | mean ratio | median ratio |
| --- | ---: | ---: | ---: |
| all 15 | 15 | +0.589 | +0.617 |
| ordinal_gap > 0.3 | 11 | +0.616 | +0.636 |
| ordinal_gap > 0.5 | 10 | +0.622 | +0.636 |

## Interpretation

Even with the ordinal_gap > 0.3 filter, the mean ratio is only +0.616 and the median is +0.636 — both below 0.7. The 'magnitude shrinks' warning stands: the binarized gap is systematically smaller than the ordinal gap even for models where the denominator is well-separated from zero.
