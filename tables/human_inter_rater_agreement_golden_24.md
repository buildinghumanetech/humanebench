# Human inter-rater agreement

_Subset mode: `golden_24` — filtered before pooling._

Computed across **85 ratings** from **4 human raters** on **24 observations**.

Reliability matrix is `(raters × observations)` with NaN for missing rater-observation pairs. Bootstrap CIs use the same seed and resample-count as `compute_inter_judge_agreement.py` so the human and AI numbers are directly comparable.

## Pooled α (multi-level bootstrap)

Three bootstrap CIs are reported per metric, differing only in the 
resampling unit. `cluster: input_id` is the primary honest CI: a 
single scenario drives every (ai_model, ai_persona) cell it appears 
in, so those cells are not independent draws. `cluster: input_id × 
eval_model` is a sensitivity check (finer; clusters only across 
personas). The **design effect** column is spec variance ÷ naive 
variance — values > 1 mean the naive CI understates uncertainty.

**Caveat: small n.** On this human-rated set there are **8 clusters** at the `input_id` level (and 15 at `input_id × eval_model`), which is below the Cameron–Gelbach–Miller (2008) well-calibrated regime for cluster bootstrap. Interpret the cluster CIs as a lower bound on honest uncertainty, not a precise interval.

| metric | spec | n clusters | avg cluster size | α | 95% CI | design effect |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| α ordinal (4-point) | item-level (naive) | 24 | 1.00 | 0.891 | [0.679, 0.960] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 8 | 3.00 | 0.891 | [0.714, 0.957] | 0.89 |
| α ordinal (4-point) | cluster: input_id × eval_model | 15 | 1.60 | 0.891 | [0.751, 0.954] | 0.74 |
| α binary (sign) | item-level (naive) | 24 | 1.00 | 1.000 | [1.000, 1.000] | 1.00 |
| α binary (sign) | cluster: input_id | 8 | 3.00 | 1.000 | [1.000, 1.000] | 1.00 |
| α binary (sign) | cluster: input_id × eval_model | 15 | 1.60 | 1.000 | [1.000, 1.000] | 1.00 |

### Paste-ready single-line numbers (cluster: input_id)

- **Krippendorff's α (ordinal, 4-point):** 0.891 [95% CI: 0.714, 0.957]
- **Krippendorff's α (binary, sign):** 1.000 [95% CI: 1.000, 1.000]

## Notes

- The existing hand-rolled implementation in `scripts/human_judging_analysis/analyze_human_ratings.py` reports α = 0.726 [0.609, 0.808]. It uses index-based ordinal distances (treats the 4 categories as evenly spaced) and a different bootstrap seed. The number above is the canonical value because (a) it uses the standard `krippendorff` package, and (b) it shares its bootstrap RNG state and cluster-level conventions with the AI inter-judge analysis.
- The naive row is retained only as a regression check — item-level resampling ignores the fact that each scenario's response text drives multiple correlated (ai_model × ai_persona) cells.