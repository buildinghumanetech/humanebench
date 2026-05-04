# Vulnerable-population breakdown (§4.6 headline)

HumaneScore (mean of judge-ensemble severities on `{-1, -0.5, +0.5, +1}`) and prosocial rate (`severity >= 0`) per stratum × persona, pooled across items × 15 models. CIs are scenario-level cluster-bootstrap (cluster: `sample_id`, seed 20260407, n_bootstrap=1000, percentile method).

## Headline DiD

`(Δ_VP_pooled) − (Δ_general)` where `Δ_group = mean(bad) − mean(baseline)`. Both Δs are typically negative (adversarial erosion); a more-negative DiD ⇒ adversarial prompting erodes VP HumaneScore *more* than general.

| metric | value | 95% CI |
| --- | ---: | --- |
| Δ general | -0.8648 | [-0.8887, -0.8405] |
| Δ VP pooled | -0.8165 | [-0.8433, -0.7861] |
| **DiD** (Δ_VP − Δ_general) | **+0.0482** | **[+0.0121, +0.0872]** |

**Verdict:** General-audience HumaneScore drops **more** than VP under adversarial prompting (DiD = +0.0482, 95% CI [+0.0121, +0.0872]).

## Per-stratum × persona

| stratum | persona | n_scenarios | n_items | HumaneScore | 95% CI | prosocial rate | 95% CI |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| general | baseline | 520 | 7796 | +0.671 | [+0.636, +0.707] | 0.888 | [0.869, 0.907] |
| general | good_persona | 520 | 7772 | +0.814 | [+0.795, +0.832] | 0.965 | [0.957, 0.973] |
| general | bad_persona | 520 | 7792 | -0.194 | [-0.212, -0.174] | 0.371 | [0.361, 0.383] |
| teenagers | baseline | 68 | 1020 | +0.824 | [+0.750, +0.883] | 0.954 | [0.914, 0.982] |
| teenagers | good_persona | 68 | 1017 | +0.894 | [+0.861, +0.920] | 0.991 | [0.983, 0.997] |
| teenagers | bad_persona | 68 | 1018 | -0.029 | [-0.077, +0.016] | 0.452 | [0.423, 0.482] |
| elderly | baseline | 46 | 688 | +0.891 | [+0.861, +0.917] | 0.996 | [0.990, 1.000] |
| elderly | good_persona | 46 | 689 | +0.904 | [+0.879, +0.928] | 0.994 | [0.987, 1.000] |
| elderly | bad_persona | 46 | 689 | -0.056 | [-0.109, -0.007] | 0.418 | [0.386, 0.450] |
| children | baseline | 40 | 599 | +0.750 | [+0.640, +0.835] | 0.927 | [0.865, 0.976] |
| children | good_persona | 40 | 599 | +0.850 | [+0.800, +0.892] | 0.982 | [0.962, 0.996] |
| children | bad_persona | 40 | 598 | -0.008 | [-0.081, +0.071] | 0.477 | [0.425, 0.533] |
| people-with-disabilities | baseline | 24 | 360 | +0.917 | [+0.868, +0.955] | 0.994 | [0.981, 1.000] |
| people-with-disabilities | good_persona | 24 | 359 | +0.936 | [+0.902, +0.965] | 1.000 | [1.000, 1.000] |
| people-with-disabilities | bad_persona | 24 | 360 | +0.120 | [+0.059, +0.184] | 0.492 | [0.441, 0.540] |
| other-VP | baseline | 90 | 1349 | +0.772 | [+0.711, +0.829] | 0.945 | [0.912, 0.971] |
| other-VP | good_persona | 90 | 1346 | +0.843 | [+0.814, +0.871] | 0.991 | [0.986, 0.996] |
| other-VP | bad_persona | 90 | 1349 | +0.019 | [-0.018, +0.054] | 0.479 | [0.456, 0.504] |
