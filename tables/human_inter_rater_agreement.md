# Human inter-rater agreement

Computed across **173 ratings** from **4 human raters** on **48 observations**.

Reliability matrix is `(raters × observations)` with NaN for missing rater-observation pairs. Bootstrap CIs use the same seed and resample-count as `compute_inter_judge_agreement.py` so the human and AI numbers are directly comparable.

## Pooled (paste these next to the AI judge numbers)

- **Krippendorff's α (ordinal, 4-point):** 0.712 [95% CI: 0.587, 0.802]
- **Krippendorff's α (binary, sign):** 0.574 [95% CI: 0.403, 0.718]

## Comparison to AI judge ensemble

| metric | humans | AI judge ensemble |
| --- | --- | --- |
| α (ordinal) | 0.712 [95% CI: 0.587, 0.802] | 0.705 [95% CI: 0.700, 0.711] |
| α (binary)  | 0.574 [95% CI: 0.403, 0.718] | 0.755 [95% CI: 0.748, 0.760] |

## Notes

- The existing hand-rolled implementation in `scripts/human_judging_analysis/analyze_human_ratings.py` reports α = 0.726 [0.609, 0.808]. It uses index-based ordinal distances (treats the 4 categories as evenly spaced) and a different bootstrap seed. The number above is the canonical value because (a) it uses the standard `krippendorff` package, and (b) it shares its bootstrap RNG state with the AI inter-judge analysis.