# Inter-judge agreement

Computed across **8,655 scored items** (samples scanned: 8,800; excluded (no individual_scores): 13; excluded (off-scale): 0).

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
| α ordinal (4-point) | item-level (naive) | 8,655 | 1.00 | 0.514 | [0.500, 0.527] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 788 | 10.98 | 0.514 | [0.489, 0.538] | 3.30 |
| α ordinal (4-point) | cluster: input_id × eval_model | 8,655 | 1.00 | 0.514 | [0.500, 0.527] | 1.06 |
| α binary (sev ≥ 0) | item-level (naive) | 8,655 | 1.00 | 0.556 | [0.542, 0.570] | 1.00 |
| α binary (sev ≥ 0) | cluster: input_id | 788 | 10.98 | 0.556 | [0.532, 0.578] | 2.43 |
| α binary (sev ≥ 0) | cluster: input_id × eval_model | 8,655 | 1.00 | 0.556 | [0.542, 0.570] | 1.01 |

**Sign-disagreement rate (≥1 judge disagrees in sign):** 28.458%

### Paste-ready single-line numbers

- **α (ordinal, 4-point), cluster: input_id:** 0.514 [95% CI: 0.489, 0.538]
- **α (binary, sev ≥ 0), cluster: input_id:** 0.556 [95% CI: 0.532, 0.578]

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 8,655 | 0.298 | 0.563 | 53.206% |
| claude-4.5-sonnet × gpt-5.1 | 8,655 | 0.468 | 0.702 | 63.512% |
| gemini-2.5-pro × gpt-5.1 | 8,655 | 0.245 | 0.493 | 48.018% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.041 | 0.294 | 0.292 | 0.373 |
| gemini-2.5-pro | 0.075 | 0.130 | 0.052 | 0.743 |
| gpt-5.1 | 0.023 | 0.365 | 0.280 | 0.333 |
