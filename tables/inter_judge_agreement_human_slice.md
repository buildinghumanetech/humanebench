# Inter-judge agreement

_Subset mode: `human_slice` — filtered before pooling._

Computed across **48 scored items** (samples scanned: 48; excluded (no individual_scores): 0; excluded (off-scale): 0).

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
| α ordinal (4-point) | item-level (naive) | 48 | 1.00 | 0.818 | [0.684, 0.890] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 8 | 6.00 | 0.818 | [0.753, 0.884] | 0.42 |
| α ordinal (4-point) | cluster: input_id × eval_model | 16 | 3.00 | 0.818 | [0.767, 0.870] | 0.30 |
| α binary (sev ≥ 0) | item-level (naive) | 48 | 1.00 | 0.774 | [0.612, 0.914] | 1.00 |
| α binary (sev ≥ 0) | cluster: input_id | 8 | 6.00 | 0.774 | [0.586, 0.939] | 1.22 |
| α binary (sev ≥ 0) | cluster: input_id × eval_model | 16 | 3.00 | 0.774 | [0.613, 0.912] | 0.99 |

**Sign-disagreement rate (≥1 judge disagrees in sign):** 16.667%

### Paste-ready single-line numbers

- **α (ordinal, 4-point), cluster: input_id:** 0.818 [95% CI: 0.753, 0.884]
- **α (binary, sev ≥ 0), cluster: input_id:** 0.774 [95% CI: 0.586, 0.939]

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 48 | 0.507 | 0.844 | 66.667% |
| claude-4.5-sonnet × gpt-5.1 | 48 | 0.626 | 0.898 | 72.917% |
| gemini-2.5-pro × gpt-5.1 | 48 | 0.434 | 0.779 | 60.417% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.208 | 0.229 | 0.167 | 0.396 |
| gemini-2.5-pro | 0.312 | 0.083 | 0.000 | 0.604 |
| gpt-5.1 | 0.167 | 0.271 | 0.188 | 0.375 |
