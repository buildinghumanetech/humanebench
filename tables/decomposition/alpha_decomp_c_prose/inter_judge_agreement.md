# Inter-judge agreement

Computed across **2,197 scored items** (samples scanned: 2,200; excluded (no individual_scores): 3; excluded (off-scale): 0).

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
| α ordinal (4-point) | item-level (naive) | 2,197 | 1.00 | 0.482 | [0.453, 0.510] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 200 | 10.98 | 0.482 | [0.421, 0.540] | 4.36 |
| α ordinal (4-point) | cluster: input_id × eval_model | 2,197 | 1.00 | 0.482 | [0.453, 0.508] | 0.97 |
| α binary (sev ≥ 0) | item-level (naive) | 2,197 | 1.00 | 0.517 | [0.485, 0.544] | 1.00 |
| α binary (sev ≥ 0) | cluster: input_id | 200 | 10.98 | 0.517 | [0.461, 0.567] | 3.21 |
| α binary (sev ≥ 0) | cluster: input_id × eval_model | 2,197 | 1.00 | 0.517 | [0.488, 0.542] | 0.94 |

**Sign-disagreement rate (≥1 judge disagrees in sign):** 31.452%

### Paste-ready single-line numbers

- **α (ordinal, 4-point), cluster: input_id:** 0.482 [95% CI: 0.421, 0.540]
- **α (binary, sev ≥ 0), cluster: input_id:** 0.517 [95% CI: 0.461, 0.567]

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 2,197 | 0.277 | 0.536 | 50.523% |
| claude-4.5-sonnet × gpt-5.1 | 2,197 | 0.442 | 0.670 | 61.402% |
| gemini-2.5-pro × gpt-5.1 | 2,197 | 0.213 | 0.467 | 43.741% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.048 | 0.300 | 0.299 | 0.352 |
| gemini-2.5-pro | 0.093 | 0.122 | 0.043 | 0.742 |
| gpt-5.1 | 0.028 | 0.363 | 0.305 | 0.303 |
