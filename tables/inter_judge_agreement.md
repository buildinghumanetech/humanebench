# Inter-judge agreement

Computed across **35,956 scored items** (samples scanned: 36,000; excluded due to missing individual_scores: 44; excluded due to off-scale severities: 0).

Eval files scanned: 45.

## Pooled (paste these into the LaTeX placeholders)

- **Krippendorff's α (ordinal, 4-point):** 0.705 [95% CI: 0.700, 0.711]
- **Krippendorff's α (binary, sev ≥ 0):** 0.755 [95% CI: 0.748, 0.760]
- **Sign-disagreement rate (≥1 judge disagrees in sign):** 14.476%

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 35,956 | 0.508 | 0.811 | 72.363% |
| claude-4.5-sonnet × gpt-5.1 | 35,956 | 0.484 | 0.773 | 66.937% |
| gemini-2.5-pro × gpt-5.1 | 35,956 | 0.365 | 0.738 | 62.212% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.108 | 0.189 | 0.155 | 0.548 |
| gemini-2.5-pro | 0.159 | 0.083 | 0.026 | 0.733 |
| gpt-5.1 | 0.056 | 0.213 | 0.223 | 0.508 |
