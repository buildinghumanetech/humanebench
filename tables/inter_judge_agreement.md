# Inter-judge agreement

Computed across **35,416 scored items** (samples scanned: 36,000; excluded (no individual_scores): 44; excluded (off-scale): 0).

Eval files scanned: 45.

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
| α ordinal (4-point) | item-level (naive) | 35,416 | 1.00 | 0.706 | [0.700, 0.711] | 1.00 |
| α ordinal (4-point) | cluster: input_id | 788 | 44.94 | 0.706 | [0.696, 0.715] | 2.77 |
| α ordinal (4-point) | cluster: input_id × eval_model | 11,820 | 3.00 | 0.706 | [0.700, 0.711] | 0.94 |
| α binary (sev ≥ 0) | item-level (naive) | 35,416 | 1.00 | 0.757 | [0.751, 0.762] | 1.00 |
| α binary (sev ≥ 0) | cluster: input_id | 788 | 44.94 | 0.757 | [0.744, 0.769] | 4.11 |
| α binary (sev ≥ 0) | cluster: input_id × eval_model | 11,820 | 3.00 | 0.757 | [0.750, 0.763] | 0.99 |

**Sign-disagreement rate (≥1 judge disagrees in sign):** 14.369%

### Paste-ready single-line numbers

- **α (ordinal, 4-point), cluster: input_id:** 0.706 [95% CI: 0.696, 0.715]
- **α (binary, sev ≥ 0), cluster: input_id:** 0.757 [95% CI: 0.744, 0.769]

## Pairwise Cohen's κ (Appendix app:judges)

| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 35,416 | 0.506 | 0.811 | 72.233% |
| claude-4.5-sonnet × gpt-5.1 | 35,416 | 0.485 | 0.775 | 66.975% |
| gemini-2.5-pro × gpt-5.1 | 35,416 | 0.366 | 0.739 | 62.220% |

## Per-judge marginal distributions (sanity check)

| judge | -1.0 | -0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.108 | 0.190 | 0.155 | 0.547 |
| gemini-2.5-pro | 0.159 | 0.082 | 0.025 | 0.734 |
| gpt-5.1 | 0.056 | 0.212 | 0.224 | 0.507 |
