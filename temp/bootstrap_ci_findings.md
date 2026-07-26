# Bootstrap CI findings — paper writing notes

Generated 2026-05-04 from `tables/score_cis_long.csv`,
`tables/persona_delta_cis_long.csv`, and the augmented headline CSVs after
adding 95% paired-scenario bootstrap CIs (n=1000, seed 20260407,
percentile method) on the 788-scenario subset.

## Methodology one-liner (for paper §)

> Confidence intervals are 95% percentile bootstrap intervals from 1000
> resamples. Scenarios are resampled with replacement, stratified by
> principle within each (model, persona) cell. For persona-delta CIs the
> same resampled scenario IDs are reused across personas, preserving the
> within-prompt pairing. Per-sample scores are the mean across the 3-judge
> ensemble; HumaneScore is the mean of the 8 principle means. Seed and
> conventions match the existing inter-judge α and ensemble-vs-human CI
> tables, so every CI in the paper shares one bootstrap design.

---

## Headline findings — all hold up

### H1. "All 15 models improve under good persona, avg +0.11" ✅

- 15/15 models have `good_delta` CIs **strictly positive**
- Mean = +0.114, between-model SE = 0.012 (the +0.11 number is itself stable)
- Range: smallest improvement = +0.048 (gemini-3-pro-preview, already near
  ceiling), largest = +0.206 (gpt-4.1)

### H2. "10/15 flip to harmful under bad persona" ✅

- 10 models: `bad_persona_score` CI **strictly below 0** (clearly harmful)
- 5 models: CI **strictly above 0** (clearly humane)
- 0 ambiguous — the count is exactly 10, no judgment calls needed

### H3. "Respect User Attention is below 0.5 at baseline" ✅ — and it is the **only** principle for which this is true

|                                  | strict <0.5 | crosses 0.5 | strict >0.5 | median |
| -------------------------------- | :---------: | :---------: | :---------: | -----: |
| **respect-user-attention**       |  **15/15**  |      0      |      0      |  −0.04 |
| be-transparent-and-honest        |      1      |      9      |      5      |   0.54 |
| enable-meaningful-choices        |      0      |      1      |     14      |   0.82 |
| design-for-equity-and-inclusion  |      0      |      0      |     15      |   0.83 |
| protect-dignity-and-safety       |      0      |      1      |     14      |   0.89 |
| foster-healthy-relationships     |      0      |      0      |     15      |   0.91 |
| enhance-human-capabilities       |      0      |      0      |     15      |   0.94 |
| prioritize-long-term-wellbeing   |      0      |      0      |     15      |   0.96 |

Strongest defensible framing on Respect User Attention:

- **All 15 models** are below the acceptable threshold (0.5)
- 4/15 are strictly below the **harmful** threshold (grok-4, gpt-4.1,
  gemini-2.5-flash, llama-4-maverick) — these can be called actively
  anti-attentional
- 7/15 straddle 0 — neutral-leaning-negative
- 4/15 are strictly above 0 (gemini-3-pro-preview, claude-sonnet-4.5,
  gpt-5.1, gpt-5) — only frontier models clear the harmful line, none
  clear the acceptable line
- be-transparent-and-honest is the clear distant second-weakest principle

**Watch in prose:** the existing "below 0.5 acceptable threshold" framing
(matching the chart's vertical line) is bulletproof. Anywhere prose drifts
to "below 0," soften it — only true for 4/15 models.

### H4. Longitudinal: "newer models more robust" — mostly yes, with two statistically real regressions

| transition                                      | direction |   CIs    | comment                |
| ----------------------------------------------- | --------- | :------: | ---------------------- |
| claude-sonnet-4 → 4.5                           | ↑ +0.27   | disjoint | clear improvement      |
| **claude-sonnet-4.5 → claude-opus-4.1**         | **↓ −0.11** | **disjoint** | **real regression**    |
| gpt-4o → gpt-4.1                                | flat      | overlap  | indistinguishable      |
| gpt-4.1 → gpt-5                                 | ↑ +1.44   | disjoint | huge jump              |
| gpt-5 → gpt-5.1                                 | ↓ −0.004  | overlap  | indistinguishable      |
| gemini-2.0-flash → 2.5-flash                    | ↑ +0.025  | overlap  | indistinguishable      |
| **gemini-2.5-flash → 2.5-pro**                  | **↓ −0.04** | **disjoint** | real but tiny regression |
| gemini-2.5-pro → 3-pro-preview                  | ↑ +0.27   | disjoint | clear improvement      |
| llama-3.1-405b → llama-4-maverick               | ↑ +0.35   | disjoint | clear improvement      |

- 4 transitions: clean improvement (CIs disjoint, direction up)
- 2 transitions: **statistically real regressions** — claude-opus-4.1 and
  gemini-2.5-pro both lose ground vs their immediate predecessor on
  adversarial robustness
- 3 transitions: indistinguishable from noise (overlapping CIs)

CI lens cleanly separates real backslides from sampling noise.

### H5. Binarized robustness gap — pre-existing partial fail, not introduced by CIs

The script was already raising `**MAGNITUDE SHRINKS**` (mean ratio 0.542 <
0.7 threshold). After adding CIs:

- Every per-model CI on both ordinal and binarized gap is tight and
  excludes 0 — so the *direction* of the gap is solid for every model
- Magnitude really does shrink ~halve under binarization
- Treat as a pre-existing footnote-worthy caveat in §3.5 / §4

---

## Sub-findings worth a callout

### Two robustness claims that don't survive scrutiny

| model              | bad_delta | 95% CI                | what it really means                        |
| ------------------ | --------: | --------------------- | ------------------------------------------- |
| claude-opus-4.1    |    −0.019 | [−0.059, +0.019]      | indistinguishable from no drift             |
| **claude-sonnet-4.5** |  **+0.016** | **[−0.010, +0.045]** | **point estimate is positive — does not degrade at all under bad persona** |

Both are currently classified "Robust" — accurate — but the data support a
**stronger** claim for claude-sonnet-4.5: it doesn't show measurable drift.
Worth one sentence in the model-detail prose.

### Robustness classifications are clean

Zero models have a `bad_delta` CI that crosses the −0.1 (Robust|Moderate)
or −0.5 (Moderate|Failed) thresholds. Categories are stable.

---

## Methodological notes

### CI widths are well-behaved

- HumaneScore CI median width: 0.043 (range 0.020–0.107)
- HumaneScore CIs are **2.53× tighter** than per-principle CIs (close to
  the √(788/98) ≈ 2.83 expectation; the slight gap is from heavier-tailed
  principles)

### Per-principle CI widths split into two regimes

| principle                       | median CI width |
| ------------------------------- | --------------: |
| prioritize-long-term-wellbeing  |           0.050 |
| foster-healthy-relationships    |           0.070 |
| enhance-human-capabilities      |           0.075 |
| protect-dignity-and-safety      |           0.078 |
| enable-meaningful-choices       |           0.082 |
| design-for-equity-and-inclusion |           0.124 |
| respect-user-attention          |           0.170 |
| be-transparent-and-honest       |           0.175 |

The two wide-CI principles (be-transparent-and-honest and
respect-user-attention) are exactly the ones with the lowest judge α —
consistent finding across the paper.

### Pairing tightens delta CIs

Median paired-delta CI width is 0.87× the naive marginal-difference width;
goes as low as 0.71× for high-correlation models like gpt-5. Confirms the
within-prompt pairing is wired correctly and matters.

### No effective-n divergence

Paired-delta `n_eff` never drops more than ~3 below the marginal cell n.
The "788-scenario subset" framing is honest at every level.

---

## Pre-existing data-hygiene issue (not CI-related, but worth fixing)

`baseline_scores.csv` and `inter_judge_raw.csv` are slightly out of sync
for 3 models (4 cells). Differences are tiny but real:

| cell                                       | published `overall` | bootstrap PE | diff    |
| ------------------------------------------ | ------------------: | -----------: | ------: |
| gemini-2.5-pro · baseline                  |              0.7645 |       0.7625 | −0.0020 |
| gemini-3-pro-preview · baseline            |              0.7845 |       0.7827 | −0.0019 |
| gemini-3-pro-preview · good_persona        |              0.9305 |       0.9281 | −0.0024 |
| llama-3.1-405b-instruct · good_persona     |              0.6830 |       0.6721 | −0.0109 |
| (all other 41 cells)                       |                   — |            — | exact   |

All discrepancies are well inside the bootstrap CI widths (~0.04). No claim
changes. The bootstrap uses `inter_judge_raw.csv` as ground truth, where
the principle-weighted HumaneScore matches published values exactly for
13/15 models.

**Fix:** re-run `extract_all_scores.py` once the broken `botocore` import
is fixed in the venv to bring the two CSVs back into sync.

---

## Inter-model ranking (lower priority finding)

12/14 adjacent baseline-HumaneScore rank pairs have overlapping CIs. The
two clean rank gaps are gpt-5 vs gemini-3-pro-preview, and gpt-4.1 vs
llama-4-maverick. The whole middle band is one indistinguishable cluster
at the CI level.

Not a problem if the paper doesn't make strong "model A > model B" claims.
If anywhere the prose says one specific model beats another in the middle
of the pack, consider softening to tier-level language ("frontier tier",
"middle tier", "bottom tier").

---

## Definition of red flags used in this analysis

Statistical: CI excludes/includes 0 contradicting a directional claim; CI
crosses a classification threshold (0.5 acceptable, 0 harmful, ±0.1 / ±0.5
robustness). Methodological: CI widths >2.5× or <0.4× median; effective-n
divergence >10 between paired and marginal. Inter-model: overlapping CIs
supporting any "A > B" claim.

Scripts that produced these findings (in `/tmp/claude/`):

- `red_flag_scan.py` — first pass, all categories
- `headline_validate.py` — second pass, focused on actual paper claims
- `investigate_mismatch.py` — point-estimate diagnostic
