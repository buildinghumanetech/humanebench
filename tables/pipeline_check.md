# Pipeline check

Rebuilt from the 45 `.eval` files with no intermediate artifact in the way. Every count below is read off the logs; nothing is carried forward from the `.tex` or from a derived table.

## The accounting chain

| step | count | expected | verdict |
| --- | ---: | ---: | --- |
| `.eval` files scanned | 45 | 45 | MATCH |
| samples on disk (800 x 45) | 36,000 | 36,000 | MATCH |
| less: excluded by dataset flag (12 x 45) | -540 | 540 | MATCH |
| **items in analysis scope (788 x 45)** | **35,460** | 35,460 | MATCH |
| less: judge-failure cascades | -44 | 44 | MATCH |
| less: off-scale severities | -0 | 0 | MATCH |
| **successfully scored items** | **35,416** | 35,416 | MATCH |
| per-judge rows (scored x 3) | 106,248 | 106,248 | MATCH |

Judge-failure rate: **44 / 35,460 = 0.124%** of the analysis scope.

### Resolving the reported discrepancy

`tables/inter_judge_agreement.md` reports "35,416 scored items (samples scanned: 36,000; excluded (no individual_scores): 44)", which reads as though 36,000 - 44 = 35,956 should be the scored total. It omits the 540-item dataset-exclusion line (12 excluded prompts x 45 runs). With that line restored the chain closes exactly. The 35,956 figure quoted in `scripts/compute_binarized_robustness_gap.py`'s docstring is the pre-exclusion count and is not the analysis denominator.

**The paper's denominator for the judge-failure rate should be 35,460 (the 788-scenario analysis scope), not 36,000.**

## Committed artifact comparison

`tables/inter_judge_raw.csv` vs this fresh scan: **IDENTICAL** -- 106,248 rows match on ['persona', 'model', 'sample_id', 'judge_name']

## Cell census

45 cells; **23 are short of 788 scenarios**. Cells are ragged, so `n = 788` is not uniformly true and any analysis that assumes it will silently mis-weight.

| persona | model | n scenarios | shortfall |
| --- | --- | ---: | ---: |
| bad_persona | claude-sonnet-4 | 787 | 1 |
| bad_persona | deepseek-v3.1-terminus | 787 | 1 |
| bad_persona | gemini-2.5-pro | 787 | 1 |
| bad_persona | gpt-4.1 | 787 | 1 |
| bad_persona | gpt-5 | 786 | 2 |
| bad_persona | gpt-5.1 | 787 | 1 |
| bad_persona | grok-4 | 785 | 3 |
| bad_persona | llama-3.1-405b-instruct | 787 | 1 |
| bad_persona | llama-4-maverick | 787 | 1 |
| baseline | claude-opus-4.1 | 787 | 1 |
| baseline | gpt-4o-2024-11-20 | 786 | 2 |
| baseline | llama-3.1-405b-instruct | 787 | 1 |
| baseline | llama-4-maverick | 786 | 2 |
| good_persona | claude-opus-4.1 | 785 | 3 |
| good_persona | claude-sonnet-4 | 787 | 1 |
| good_persona | deepseek-v3.1-terminus | 785 | 3 |
| good_persona | gemini-2.0-flash-001 | 784 | 4 |
| good_persona | gemini-2.5-flash | 782 | 6 |
| good_persona | gemini-2.5-pro | 787 | 1 |
| good_persona | gpt-5.1 | 787 | 1 |
| good_persona | grok-4 | 787 | 1 |
| good_persona | llama-3.1-405b-instruct | 784 | 4 |
| good_persona | llama-4-maverick | 786 | 2 |

Full census: `cell_census.csv`.

## Complete-case scenario set

Scenarios present in **all 45 cells**: **746** of 788 (42 lost, 5.33%).

This is the resampling frame for every cohort-level statistic (flip count, robust-set size) computed with the shared-scenario cluster bootstrap. Per-cell marginals still use the full per-cell scenario set; the two frames differ and the difference is reported wherever it matters. Ids: `complete_case_scenarios.txt`.
