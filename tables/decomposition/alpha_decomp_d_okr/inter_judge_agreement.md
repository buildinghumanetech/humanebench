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
| α ordinal (4-point) | item-level (naive) | 2,197 | 1.00 | 0.506 | [0.478, 0.536] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 200 | 10.98 | 0.506 | [0.451, 0.557] | 3.76 |
| α ordinal (4-point) | cluster: input_id × eval_model | 2,197 | 1.00 | 0.506 | [0.478, 0.534] | 0.97 |
| α binary (sev ≥ 0) | item-level (naive) | 2,197 | 1.00 | 0.559 | [0.530, 0.586] | 1.00 |
| α binary (sev ≥ 0) | cluster: input_id | 200 | 10.98 | 0.559 | [0.508, 0.606] | 3.09 |
| α binary (sev ≥ 0) | cluster: input_id × eval_model | 2,197 | 1.00 | 0.559 | [0.529, 0.587] | 1.00 |

**Sign-disagreement rate (≥1 judge disagrees in sign):** 29.085%

### Paste-ready single-line numbers

- **α (ordinal, 4-point), cluster: input_id:** 0.506 [95% CI: 0.451, 0.557]
- **α (binary, sev ≥ 0), cluster: input_id:** 0.559 [95% CI: 0.508, 0.606]

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 2,197 | 0.289 | 0.566 | 51.934% |
| claude-4.5-sonnet × gpt-5.1 | 2,197 | 0.471 | 0.683 | 63.541% |
| gemini-2.5-pro × gpt-5.1 | 2,197 | 0.223 | 0.471 | 43.924% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.043 | 0.301 | 0.293 | 0.363 |
| gemini-2.5-pro | 0.084 | 0.145 | 0.043 | 0.728 |
| gpt-5.1 | 0.023 | 0.381 | 0.311 | 0.286 |
