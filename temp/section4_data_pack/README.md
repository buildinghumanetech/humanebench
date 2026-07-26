# HumaneBench Section 4 Data Pack

Numbers and tables needed to draft the AIES paper's Section 4 (Results), plus
everything the website whitepaper and the internal results spreadsheet
reference. All CSVs are taken from the canonical `humanebench` repo and
reflect the **788-scenario analysis subset** (the 800-scenario corpus minus
12 items excluded for a known *Design for Equity and Inclusion* generation
artifact, where the prompts referenced a non-existent attached image or
document).

Snapshot date: 2026-05-04. Bootstrap CIs computed with seed `20260407`,
n=1,000 paired-scenario resamples, 95% percentile method. Regenerate via
`python scripts/compute_score_cis.py` from the humanebench repo root.

---

## Files

### Per-model headline tables (use these for almost everything in Section 4)

- **`steerability_comparison.csv`** — **the master table.** 15 rows × 20
  columns: per-model HumaneScore at baseline / good / bad with 95% CIs,
  good_delta and bad_delta with paired CIs, robustness_status (Robust /
  Moderate / Failed), and per-condition negative_rate. Every claim in
  Section 4.1, 4.2, and 4.3 traces to this file.
- **`baseline_scores.csv`**, **`good_persona_scores.csv`**, **`bad_persona_scores.csv`** —
  per-model × per-principle scores with 95% CIs. 15 rows × 31 columns
  each (the 8 principles × {value, ci_lower, ci_upper} + HumaneScore overall +
  metadata). Use these for Section 4.4's per-principle claims and for
  paper Tables 2–4 (the heatmaps).
- **`longitudinal_comparison.csv`** — 13 rows (Anthropic, OpenAI, Google,
  Meta — generations 1 through N each), with per-generation
  HumaneScore + CIs across all three conditions, plus deltas with CIs and
  robustness_status. Use this for Section 4.3's "Anthropic and Google
  generation regressions" call-out and any per-lab trend prose.

### Long-format CSVs (useful for pivots and per-principle work)

- **`tables/score_cis_long.csv`** — long-format `(model, persona,
  principle, point_estimate, ci_lower, ci_upper, n_eff)` for every cell.
  Equivalent to the three per-persona CSVs above, but tidy. Use for
  pandas pivots, especially the agency-cluster aggregation in 4.4.
- **`tables/persona_delta_cis_long.csv`** — per-`(model, contrast_persona,
  principle)` paired bootstrap deltas with CIs. Use for per-principle
  delta claims (e.g., the Foster Healthy Relationships standalone biggest
  drop, Δ = −1.12). The HumaneScore-level deltas in this file are the
  same as `good_delta` / `bad_delta` in `steerability_comparison.csv` —
  filter `principle == 'HumaneScore'` to extract them.

### Other Section 4 inputs

- **`tables/helm_humanescore_combined.csv`** — 10 rows × 5 columns: the 10
  HELM-matched models with their HELM aggregate score and HumaneScore at
  each of the three conditions. Use for Section 4.5 and the HELM scatter
  figure. Pearson correlations between HELM and HumaneScore are computed
  on the fly:
  - HELM ~ Baseline: r = +0.48, p = 0.16
  - HELM ~ Good: r = +0.67, **p = 0.035**
  - HELM ~ Bad: r = +0.43, p = 0.22
  - HELM ~ Δ_bad: r = +0.38, p = 0.27

### Optional (only needed for Section 3.5 / judge-validation prose)

- **`tables/ensemble_vs_human_curated_24_cis.csv`** — 1 row of
  ensemble-judge-vs-human agreement metrics on the curated 24-item
  validation set (weighted κ, Spearman ρ, Pearson r, direction-match
  rate), with bootstrap CIs. Source of the whitepaper's "23 out of 24
  test questions" framing (95.83% direction-match rate). Not needed for
  Section 4; only matters if Section 3.5 is also being drafted.

---

## Conventions and watch-outs

### CI methodology

All CIs in this pack are 95% percentile bootstrap from 1,000
paired-scenario resamples. Resampling unit: `sample_id` (scenario),
stratified by principle within each (model, persona) cell. The same
resampled scenario IDs are reused across personas within each model so
persona-delta CIs preserve within-prompt pairing — this is why
`good_delta` and `bad_delta` CIs in `steerability_comparison.csv` are
0.71×–0.94× tighter than naive marginal-difference CIs. Seed:
`20260407`. Conventions match the existing inter-judge α and
ensemble-vs-human CI scripts so all CIs in the paper share one
bootstrap design.

### The 788-subset

`baseline_scores.csv` reports `total_samples=788`, `scored_samples`
ranges 786–788 per cell (a handful of samples failed in one of the
three judges). The 12 excluded items are tagged in
`data/humane_bench.jsonl` via `metadata.excluded_from_analysis = true`
(8 of which fall on *Design for Equity and Inclusion*, which is why
that principle has only 90 baseline samples per model rather than
100).

### **Watch-out: +11% vs +16% — definitional, not a data error**

The whitepaper consistently says **"+16% average improvement"** under
good-persona prompting. That is the **per-model relative** improvement:

```
mean( good_delta_i / baseline_score_i )  =  +0.162  (+16.2%)
```

A naive `mean(good_delta)` returns **+0.114** (+11.4%) — the same
finding, expressed in absolute HumaneScore points on the [−1, +1]
scale. Both are computable from `steerability_comparison.csv` (using
the `baseline_score` and `good_delta` columns). The relative version
is what readers will internalize ("a 16% boost just from prompting");
the absolute version is invariant to scale choice.

**If you are paraphrasing the whitepaper, use +16%.** If you are
reporting on the [−1, +1] scale directly, +0.114 is honest. Don't mix
them up.

### Robust models: 4 (strict) vs 5 (loose)

- **5 models do not satisfy Eq. 4** (`S_baseline > 0 AND S_bad < 0`):
  GPT-5, GPT-5.1, Claude Sonnet 4.5, Claude Opus 4.1, *and* Claude
  Sonnet 4. The "67% (10/15) flip" headline is computed with this
  loose criterion.
- **4 models maintain HumaneScore ≥ 0.5 under bad persona with CI
  excluding 0.5**: GPT-5, GPT-5.1, Claude Sonnet 4.5, Claude Opus 4.1.
  This is the stricter robustness bar that matches the rubric's
  "acceptable" threshold and the whitepaper's Finding 2 list.
- Claude Sonnet 4 is borderline: bad-persona = +0.50 [+0.46, +0.54],
  CI straddles 0.5. Doesn't flip negative but doesn't clear the
  acceptable bar either.

### User agency cluster

The paper's Section 4.4 defines `user agency = {Respect User Attention,
Enable Meaningful Choices, Enhance Human Capabilities, Be Transparent
and Honest}`. The cluster claim is "user agency is the *structurally
weakest* dimension throughout":

- Baseline: agency cluster mean **+0.56** vs **+0.88** for the other 4
  principles
- Bad persona absolute: **−0.20** vs **−0.06**
- Drop magnitude (Δ): **−0.76** vs **−0.93** — agency drops *less*,
  but only because it had less ceiling to fall from. **Don't claim
  "agency takes the biggest hit"** — by drop magnitude that's actually
  Foster Healthy Relationships standalone (Δ = −1.12).

The whitepaper Finding 5 ("User Empowerment Takes the Biggest Hit")
predates this analysis and isn't quantitatively supported under the
strict 2-principle definition (Enable Choices + Enhance Capabilities)
either. Recommend the "weakest dimension throughout" framing for the
academic paper; flag back to the team whether the whitepaper headline
should be updated.

### Per-principle ranking under bad persona (mean Δ, all 15 models)

```
1. Foster Healthy Relationships    Δ = −1.12  (standalone worst)
2. Prioritize Long-term Wellbeing  Δ = −0.94
3. Protect Dignity and Safety      Δ = −0.92
4. Enable Meaningful Choices       Δ = −0.91  [agency]
5. Be Transparent and Honest       Δ = −0.89  [agency]
6. Enhance Human Capabilities      Δ = −0.79  [agency]
7. Design for Equity and Inclusion Δ = −0.75
8. Respect User Attention          Δ = −0.46  [agency]
                                    (smallest, but already near 0
                                    at baseline — ceiling effect)
```

---

## Quick reference: headline numbers as they should appear in the paper

| Claim | Value | Source |
|---|---|---|
| Models evaluated | 15 | `steerability_comparison.csv` row count |
| Scenarios in analysis subset | 788 | `baseline_scores.csv:total_samples` |
| Mean improvement under good persona (relative) | +16.2% | `mean(good_delta / baseline_score)` |
| Mean improvement under good persona (absolute) | +0.114 | `mean(good_delta)` |
| Models with strictly positive good_delta CI | 15 of 15 | `steerability_comparison.csv:good_delta_ci_lower > 0` |
| Models that flip negative under bad persona | 10 of 15 (66.7%) | `bad_persona_score_ci_upper < 0` count |
| Robust models (CI strictly above 0.5 under bad) | 4 of 15 | GPT-5, GPT-5.1, Sonnet 4.5, Opus 4.1 |
| Worst absolute swing | Gemini 2.5 Pro, Δ = −1.49 [−1.51, −1.46] | `steerability_comparison.csv:bad_delta` min |
| Largest standalone principle drop | Foster Healthy Relationships, Δ = −1.12 | `tables/persona_delta_cis_long.csv` |
| Respect User Attention below 0.5 at baseline | 15 of 15 (universal) | `tables/score_cis_long.csv` |
| Inter-judge agreement (curated 24) | 23 of 24 (95.83%) | `tables/ensemble_vs_human_curated_24_cis.csv` |
| HELM × Good-persona HumaneScore | r = +0.67, p = 0.035 | computed from `tables/helm_humanescore_combined.csv` |

---

## Regeneration

Everything in this pack is reproducible from a clean checkout of the
`humanebench` repo:

```
python scripts/compute_inter_judge_agreement.py    # produces tables/inter_judge_raw.csv
python scripts/compute_score_cis.py                # adds CIs and patches the per-persona / steerability CSVs
python scripts/longitudinal_analysis.py            # produces longitudinal_comparison.csv
python scripts/create_helm_humanescore_comparison.py  # produces tables/helm_humanescore_combined.csv
```

CSV contents are bit-for-bit reproducible given the same inter_judge_raw.csv
and the bootstrap seed.
