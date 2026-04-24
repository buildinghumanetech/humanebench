# Inter-judge agreement

_Subset mode: `golden_24` — filtered before pooling._

Computed across **24 scored items** (samples scanned: 24; excluded (no individual_scores): 0; excluded (off-scale): 0).

Eval files scanned: 1.

## Pooled α (multi-level bootstrap)

Three bootstrap CIs are reported per metric, differing only in the 
resampling unit:

- **item-level (naive)** — resamples individual judge × sample rows. 
  Ignores within-scenario correlation and is the *narrow* CI reported 
  in earlier drafts. Retained here as a regression check and reference.
- **cluster: input_id** — resamples scenarios; every (model, persona) 
  cell belonging to a selected scenario is carried along. This is the 
  honest cluster unit: the response text for a given prompt drives 
  every judge's score for that prompt across the (model, persona) grid.
- **cluster: input_id × eval_model** — resamples scenario × eval-model 
  cells (finer; clusters only across personas). A sensitivity check 
  for the primary `cluster: input_id` CI.

The **design effect** column = spec variance ÷ naive variance. 
Values > 1 mean the naive CI understates uncertainty. On the 48-item 
human-rated slice, n_clusters = 8 for `input_id`, which is below the 
Cameron–Gelbach–Miller (2008) well-calibrated regime; interpret that 
CI as a lower bound on honest uncertainty rather than a precise number.

| metric | spec | n clusters | avg cluster size | α | 95% CI | design effect |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| α ordinal (4-point) | item-level (naive) | 24 | 1.00 | 0.784 | [0.570, 0.876] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 8 | 3.00 | 0.784 | [0.521, 0.876] | 1.25 |
| α ordinal (4-point) | cluster: input_id × eval_model | 15 | 1.60 | 0.784 | [0.611, 0.883] | 0.81 |
| α binary (sev ≥ 0) | item-level (naive) | 24 | 1.00 | 0.877 | [0.669, 1.000] | 1.00 |
| α binary (sev ≥ 0) | cluster: input_id | 8 | 3.00 | 0.877 | [0.636, 1.000] | 1.21 |
| α binary (sev ≥ 0) | cluster: input_id × eval_model | 15 | 1.60 | 0.877 | [0.671, 1.000] | 1.07 |

**Sign-disagreement rate (≥1 judge disagrees in sign):** 8.333%

### Paste-ready single-line numbers

- **α (ordinal, 4-point), cluster: input_id:** 0.784 [95% CI: 0.521, 0.876]
- **α (binary, sev ≥ 0), cluster: input_id:** 0.877 [95% CI: 0.636, 1.000]

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 24 | 0.455 | 0.839 | 62.500% |
| claude-4.5-sonnet × gpt-5.1 | 24 | 0.362 | 0.808 | 54.167% |
| gemini-2.5-pro × gpt-5.1 | 24 | 0.426 | 0.865 | 62.500% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.208 | 0.125 | 0.292 | 0.375 |
| gemini-2.5-pro | 0.250 | 0.083 | 0.000 | 0.667 |
| gpt-5.1 | 0.083 | 0.250 | 0.208 | 0.458 |
