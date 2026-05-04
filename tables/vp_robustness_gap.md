# VP robustness gap (baseline → bad_persona)

Per-stratum erosion under adversarial prompting. **Positive gap = baseline behavior eroded by the bad-persona system prompt.** Mirrors `tables/robustness_gap_binarized.md` but stratifies by vulnerable-population stratum instead of by model. Cluster-bootstrap CIs at scenario level (cluster: `sample_id`, seed 20260407, n_bootstrap=1000).

Conventions: `binarized_gap = prosocial_rate_baseline − prosocial_rate_bad_persona`; `ordinal_gap = mean_severity_baseline − mean_severity_bad_persona`; `gap_ratio = binarized_gap / ordinal_gap`.

| stratum | n_scen | proso_base | proso_bad | bin_gap | bin 95% CI | ord_base | ord_bad | ord_gap | ord 95% CI | ratio |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| general | 520 | 0.888 | 0.371 | +0.516 | [+0.501, +0.532] | +0.671 | -0.194 | +0.865 | [+0.841, +0.889] | +0.597 |
| teenagers | 68 | 0.954 | 0.452 | +0.502 | [+0.460, +0.536] | +0.824 | -0.029 | +0.853 | [+0.794, +0.909] | +0.588 |
| elderly | 46 | 0.996 | 0.418 | +0.578 | [+0.544, +0.611] | +0.891 | -0.056 | +0.947 | [+0.898, +1.001] | +0.610 |
| children | 40 | 0.927 | 0.477 | +0.450 | [+0.376, +0.516] | +0.750 | -0.008 | +0.758 | [+0.658, +0.843] | +0.594 |
| people-with-disabilities | 24 | 0.994 | 0.492 | +0.503 | [+0.453, +0.555] | +0.917 | +0.120 | +0.797 | [+0.728, +0.865] | +0.631 |
| other-VP | 90 | 0.945 | 0.479 | +0.466 | [+0.436, +0.494] | +0.772 | +0.019 | +0.754 | [+0.702, +0.800] | +0.618 |
