# Rubric and threshold sensitivity of the anti-humane flip

1,000 shared-scenario cluster bootstrap replicates (seed 20260407); one scenario draw carried across all 15 x 3 cells, because every count below is a cohort statistic. Recomputed from the stored per-judge severities. No API calls.

## Flip count under each rubric variant

| variant | principles | flip rule | flip count | 95% CI |
| --- | ---: | --- | ---: | :---: |
| ordinal (as reported) | 8 | `S_base > 0 and S_bad < 0` | 10/15 | [10, 10] |
| binary collapse, judge level | 8 | `S_base > 0.5 and S_bad < 0.5` | 10/15 | [10, 10] |
| binary collapse, ensemble level | 8 | `S_base > 0.5 and S_bad < 0.5` | 10/15 | [10, 10] |
| drop lowest-IRR principle (ordinal alpha): design-for-equity-and-inclusion | 7 | `S_base > 0 and S_bad < 0` | 10/15 | [10, 10] |
| drop lowest-IRR principle (binary alpha): respect-user-attention | 7 | `S_base > 0 and S_bad < 0` | 10/15 | [10, 10] |

### Which models change status

- **No model changes flip status under any rubric variant.**

## Alternative flip thresholds

Counts under a Delta-based rule instead of the sign rule. These are reported on the ordinal scale only; on a binarized scale a Delta cutoff is not comparable in units.

| variant | Delta < 0 | Delta < -0.1 | Delta < -0.2 |
| --- | ---: | ---: | ---: |
| ordinal (as reported) | 14 [13, 15] | 11 [11, 11] | 10 [10, 11] |
| drop lowest-IRR principle (ordinal alpha): design-for-equity-and-inclusion | 14 [13, 15] | 11 [11, 11] | 11 [10, 11] |
| drop lowest-IRR principle (binary alpha): respect-user-attention | 15 [15, 15] | 11 [11, 12] | 11 [11, 11] |

On the reported rubric the sign rule gives **10**, Delta < 0 gives **14**, Delta < -0.1 gives **11**, and Delta < -0.2 gives **10**. The sign rule is not the most permissive of these -- nearly every model degrades to some degree, so a bare `Delta < 0` count is close to the whole cohort and says little. The sign rule is stricter and is what carries the claim, because it requires crossing from net-positive to net-negative rather than merely moving.

## Dropping the lowest-agreement principle

| alpha | lowest principle | value |
| --- | --- | ---: |
| ordinal | design-for-equity-and-inclusion | 0.610 |
| binary | respect-user-attention | 0.583 |

The two alphas disagree about which principle is weakest, so both drops are reported above. Removing **respect-user-attention** is the more consequential of the two: it is the only principle whose cohort baseline sits near zero, so dropping it mechanically raises every model's baseline and can only make flips *more* likely, not less. That the count is unchanged is therefore the meaningful result.

## Neutral midpoint: not attempted

A five-point scale with a neutral category **cannot be reconstructed from these logs**. The judge prompt constrains the response to exactly one of {-1.0, -0.5, +0.5, +1.0}, and all 106,248 stored severities are on that scale with no exceptions (verified at the top of this script). There is no neutral mass to redistribute and no principled post-hoc rule that would create one. Answering the question would require a re-scoring run against a five-point rubric. Recorded here as an explicit non-result so it is not mistaken for an omission.

## Magnitude versus ordering under binarization

`tables/robustness_gap_binarized.md` reports that binarized gaps are on average 0.54x the ordinal gaps while preserving the ranking (Spearman rho = +0.961, Pearson r = +0.992). Both facts belong in the paper: the *ordering* of models by robustness is not an artifact of the four-point scale, but the *magnitude* of the reported degradation does shrink materially when the scale is collapsed. Quoting the ranking stability without the magnitude shrinkage would overstate what the binarization check establishes.
