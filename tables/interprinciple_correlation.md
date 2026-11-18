# Model-level inter-principle correlation

**Read the caveat before the numbers.** These correlations are computed across the **15 evaluated models**, so n = 15. That is the entire sample. A single correlation at n = 15 carries a 95% CI roughly 1.0 wide, and the 15 models are a coverage-driven selection of frontier systems, not a random sample from any population. This cannot establish factor structure, cannot refute construct overlap, and should not be described as doing either.

The analysis that *would* answer the construct-overlap question is a multi-label rescoring -- score a scenario subsample against all eight principles and factor the resulting item x dimension matrix. HumaneBench assigns exactly one principle per scenario, so no such matrix exists in the stored data, and building one needs new judge calls. It is not computed here.

## baseline

Median 95% CI width across the 28 pairs: **0.49** (on a scale that only spans 2.0). 23 of 28 pairs reach p < 0.05 uncorrected; at 28 tests roughly 1.4 would be expected by chance alone.

| | rua | emc | ehc | pds | fhr | pltw | bath | dei |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **rua** | -- | +0.38 | +0.32 | +0.19 | +0.50 | +0.33 | +0.66 | +0.59 |
| **emc** | +0.38 | -- | +0.81 | +0.96 | +0.66 | +0.95 | +0.84 | +0.85 |
| **ehc** | +0.32 | +0.81 | -- | +0.77 | +0.72 | +0.84 | +0.64 | +0.67 |
| **pds** | +0.19 | +0.96 | +0.77 | -- | +0.58 | +0.95 | +0.77 | +0.77 |
| **fhr** | +0.50 | +0.66 | +0.72 | +0.58 | -- | +0.64 | +0.70 | +0.62 |
| **pltw** | +0.33 | +0.95 | +0.84 | +0.95 | +0.64 | -- | +0.80 | +0.80 |
| **bath** | +0.66 | +0.84 | +0.64 | +0.77 | +0.70 | +0.80 | -- | +0.80 |
| **dei** | +0.59 | +0.85 | +0.67 | +0.77 | +0.62 | +0.80 | +0.80 | -- |

Strongest 5 pairs, with the CI that undercuts them:

| pair | r | 95% CI | p |
| --- | ---: | :---: | ---: |
| emc x pds | +0.96 | [+0.86, +0.99] | 0.000 |
| emc x pltw | +0.95 | [+0.83, +0.99] | 0.000 |
| pds x pltw | +0.95 | [+0.75, +0.99] | 0.000 |
| emc x dei | +0.85 | [+0.51, +0.97] | 0.000 |
| emc x bath | +0.84 | [+0.62, +0.94] | 0.000 |

## good_persona

Median 95% CI width across the 28 pairs: **0.28** (on a scale that only spans 2.0). 28 of 28 pairs reach p < 0.05 uncorrected; at 28 tests roughly 1.4 would be expected by chance alone.

| | rua | emc | ehc | pds | fhr | pltw | bath | dei |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **rua** | -- | +0.88 | +0.74 | +0.85 | +0.67 | +0.78 | +0.83 | +0.78 |
| **emc** | +0.88 | -- | +0.89 | +0.94 | +0.75 | +0.95 | +0.94 | +0.84 |
| **ehc** | +0.74 | +0.89 | -- | +0.85 | +0.63 | +0.90 | +0.85 | +0.83 |
| **pds** | +0.85 | +0.94 | +0.85 | -- | +0.76 | +0.89 | +0.86 | +0.91 |
| **fhr** | +0.67 | +0.75 | +0.63 | +0.76 | -- | +0.75 | +0.61 | +0.58 |
| **pltw** | +0.78 | +0.95 | +0.90 | +0.89 | +0.75 | -- | +0.90 | +0.80 |
| **bath** | +0.83 | +0.94 | +0.85 | +0.86 | +0.61 | +0.90 | -- | +0.81 |
| **dei** | +0.78 | +0.84 | +0.83 | +0.91 | +0.58 | +0.80 | +0.81 | -- |

Strongest 5 pairs, with the CI that undercuts them:

| pair | r | 95% CI | p |
| --- | ---: | :---: | ---: |
| emc x pltw | +0.95 | [+0.80, +0.99] | 0.000 |
| emc x pds | +0.94 | [+0.90, +0.99] | 0.000 |
| emc x bath | +0.94 | [+0.77, +0.98] | 0.000 |
| pds x dei | +0.91 | [+0.73, +0.98] | 0.000 |
| pltw x bath | +0.90 | [+0.64, +0.98] | 0.000 |

## bad_persona

Median 95% CI width across the 28 pairs: **0.06** (on a scale that only spans 2.0). 28 of 28 pairs reach p < 0.05 uncorrected; at 28 tests roughly 1.4 would be expected by chance alone.

| | rua | emc | ehc | pds | fhr | pltw | bath | dei |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **rua** | -- | +0.92 | +0.89 | +0.95 | +0.95 | +0.95 | +0.97 | +0.94 |
| **emc** | +0.92 | -- | +0.98 | +0.98 | +0.99 | +1.00 | +0.95 | +0.98 |
| **ehc** | +0.89 | +0.98 | -- | +0.94 | +0.95 | +0.98 | +0.89 | +0.97 |
| **pds** | +0.95 | +0.98 | +0.94 | -- | +0.99 | +0.98 | +0.98 | +0.98 |
| **fhr** | +0.95 | +0.99 | +0.95 | +0.99 | -- | +0.99 | +0.97 | +0.98 |
| **pltw** | +0.95 | +1.00 | +0.98 | +0.98 | +0.99 | -- | +0.95 | +0.99 |
| **bath** | +0.97 | +0.95 | +0.89 | +0.98 | +0.97 | +0.95 | -- | +0.95 |
| **dei** | +0.94 | +0.98 | +0.97 | +0.98 | +0.98 | +0.99 | +0.95 | -- |

Strongest 5 pairs, with the CI that undercuts them:

| pair | r | 95% CI | p |
| --- | ---: | :---: | ---: |
| emc x pltw | +1.00 | [+0.99, +1.00] | 0.000 |
| pds x fhr | +0.99 | [+0.98, +1.00] | 0.000 |
| emc x fhr | +0.99 | [+0.98, +1.00] | 0.000 |
| fhr x pltw | +0.99 | [+0.98, +1.00] | 0.000 |
| pltw x dei | +0.99 | [+0.96, +1.00] | 0.000 |

## Why high correlations here do *not* mean the principles overlap

The correlations are large and get larger under adversarial pressure. That pattern is expected under **either** hypothesis and so discriminates between neither.

The unit of observation is the model. Models differ enormously in overall quality, and that single dominant dimension enters every one of the eight per-principle means. A model that scores well on one principle scores well on all of them because it is a better model, not because the principles measure the same thing. Aggregating to the model level therefore confounds construct similarity with a general model-quality factor, and cannot separate them: eight genuinely distinct constructs measured on 15 models of widely varying quality would produce exactly this matrix.

The bad-persona column makes the artifact visible. Median |r| rises to 0.97 there, with CI widths collapsing to ~0.06 -- not because the principles became more alike under pressure, but because most models are driven toward the floor together, leaving one dimension of variance. Reading that as construct overlap would be a mistake.

## How to use this

As supporting material only, and only alongside the n = 15 caveat and the general-factor confound above, stated in the same breath. If the paper needs a defensible answer on construct distinctness, the multi-label rescoring is the analysis to run; this is not a substitute for it, and quoting these correlations without the confound would invite a fair reviewer objection.

Short codes: `rua` = respect-user-attention, `emc` = enable-meaningful-choices, `ehc` = enhance-human-capabilities, `pds` = protect-dignity-and-safety, `fhr` = foster-healthy-relationships, `pltw` = prioritize-long-term-wellbeing, `bath` = be-transparent-and-honest, `dei` = design-for-equity-and-inclusion.
