# Inter-judge agreement

Computed across **2,199 scored items** (samples scanned: 2,200; excluded (no individual_scores): 1; excluded (off-scale): 0).

Eval files scanned: 11.

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
| α ordinal (4-point) | item-level (naive) | 2,199 | 1.00 | 0.495 | [0.465, 0.526] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 200 | 10.99 | 0.495 | [0.428, 0.557] | 4.31 |
| α ordinal (4-point) | cluster: input_id × eval_model | 2,199 | 1.00 | 0.495 | [0.465, 0.526] | 0.92 |
| α binary (sev ≥ 0) | item-level (naive) | 2,199 | 1.00 | 0.600 | [0.567, 0.630] | 1.00 |
| α binary (sev ≥ 0) | cluster: input_id | 200 | 10.99 | 0.600 | [0.535, 0.656] | 3.35 |
| α binary (sev ≥ 0) | cluster: input_id × eval_model | 2,199 | 1.00 | 0.600 | [0.567, 0.631] | 1.02 |

**Sign-disagreement rate (≥1 judge disagrees in sign):** 20.782%

### Paste-ready single-line numbers

- **α (ordinal, 4-point), cluster: input_id:** 0.495 [95% CI: 0.428, 0.557]
- **α (binary, sev ≥ 0), cluster: input_id:** 0.600 [95% CI: 0.535, 0.656]

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 2,199 | 0.325 | 0.615 | 62.756% |
| claude-4.5-sonnet × gpt-5.1 | 2,199 | 0.462 | 0.691 | 64.575% |
| gemini-2.5-pro × gpt-5.1 | 2,199 | 0.242 | 0.519 | 52.979% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.049 | 0.183 | 0.252 | 0.517 |
| gemini-2.5-pro | 0.067 | 0.087 | 0.030 | 0.815 |
| gpt-5.1 | 0.026 | 0.255 | 0.293 | 0.425 |
