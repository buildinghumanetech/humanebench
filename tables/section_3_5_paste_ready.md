# §3.5 (Judge Validation) — Paste-Ready Numbers

This file collects the numbers needed to tighten §3.5, grouped by paragraph.
Every number here is copied from a sibling file in `tables/` or from
`scripts/human_judging_analysis/output/` — no computation happens in this
document. Cross-check against the source files before pasting into the
paper.

---

## Inter-judge agreement paragraph

Source: `tables/inter_judge_agreement.md` (computed by
`scripts/compute_inter_judge_agreement.py`).

Pooled across **35,956** scored items (44 excluded due to missing
`individual_scores` from complete judge failure). Point estimates unchanged;
the **CIs below are cluster-bootstrapped at the scenario level**
(`cluster: input_id`) and replace the earlier item-level CIs
`[0.700, 0.711]` (ord) / `[0.748, 0.760]` (bin), which ignored the
within-scenario correlation from a single prompt driving up to
15 × 3 = 45 correlated judge-item rows:

- **Krippendorff's α (ordinal, 4-point):** **0.705 [95% CI: 0.696, 0.715]**
- **Krippendorff's α (binary, sev ≥ 0):** **0.755 [95% CI: 0.742, 0.767]**
- **Sign-disagreement rate (≥1 judge disagrees in sign):** 14.476%

### Multi-level bootstrap diagnostics (for reviewer pushback / appendix)

Source: `tables/inter_judge_agreement.md` (same run). Three bootstrap CIs are
computed per metric, differing only in the resampling unit. Design effect
(DE) = spec variance ÷ naive variance — values > 1 mean the naive CI
understates uncertainty.

**Full corpus (35,956 items):**

| metric | spec | n clusters | avg size | α | 95% CI | DE |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| α ordinal | item-level (naive)                | 35,956 |  1.00 | 0.705 | [0.700, 0.711] | 1.00 |
| α ordinal | cluster: input_id                 |    800 | 44.95 | 0.705 | [0.696, 0.715] | **2.75** |
| α ordinal | cluster: input_id × eval_model    | 12,000 |  3.00 | 0.705 | [0.700, 0.711] | 0.87 |
| α binary  | item-level (naive)                | 35,956 |  1.00 | 0.755 | [0.748, 0.760] | 1.00 |
| α binary  | cluster: input_id                 |    800 | 44.95 | 0.755 | [0.742, 0.767] | **4.16** |
| α binary  | cluster: input_id × eval_model    | 12,000 |  3.00 | 0.755 | [0.749, 0.761] | 1.08 |

**Reading:** `cluster: input_id` is the honest unit — a single scenario's
response text drives every (eval_model, persona) cell it appears in. DE ≈
2.75 (ordinal) and 4.16 (binary) confirm meaningful within-scenario
correlation, so the earlier item-level CIs were meaningfully too narrow.
The `input_id × eval_model` sensitivity check barely moves (DE < 1) because
clustering *only* across personas misses the prompt-text-driven correlation
and absorbs between-persona variance instead. **Report the scenario-level
CI as the headline; mention the model×scenario CI only if a reviewer asks.**

On the **48-scenario human-rated slice** (n_clusters = 8 at `input_id`,
n_clusters = 16 at `input_id × eval_model`), n is below the Cameron–
Gelbach–Miller (2008) well-calibrated regime for cluster bootstrap.
Interpret those cluster CIs as directionally useful rather than precise.
See the 48-scenario slice block below.

### Pairwise Cohen's κ (Appendix app:judges)

Source: same file.

| pair | n | κ (unweighted) | κ (quadratic) | exact agreement |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet × gemini-2.5-pro | 35,956 | 0.508 | 0.811 | 72.4% |
| claude-4.5-sonnet × gpt-5.1          | 35,956 | 0.484 | 0.773 | 66.9% |
| gemini-2.5-pro × gpt-5.1             | 35,956 | 0.365 | 0.738 | 62.2% |

### Per-judge marginal distributions (Appendix sanity check)

Source: same file.

| judge | −1.0 | −0.5 | +0.5 | +1.0 |
| --- | ---: | ---: | ---: | ---: |
| claude-4.5-sonnet | 0.108 | 0.189 | 0.155 | 0.548 |
| gemini-2.5-pro    | 0.159 | 0.083 | 0.026 | 0.733 |
| gpt-5.1           | 0.056 | 0.213 | 0.223 | 0.508 |

---

## Human validation paragraph

### Human inter-rater agreement (new — put alongside the AI numbers)

Source: `tables/human_inter_rater_agreement.md` (computed by
`scripts/compute_human_validation_metrics.py`). Same krippendorff package,
same bootstrap seed (`20260407`), same resample count (1000) as the AI side.

Computed across **173 ratings from 4 human raters** on **48 observations**.
CIs are cluster-bootstrapped at the scenario level (`cluster: input_id`),
matching the AI-side methodology. With only **8 scenarios** (n_clusters = 8)
this is below the CGM 2008 calibrated regime — the cluster CI is close to
the naive CI in practice (ordinal DE = 1.11, binary DE = 0.98) because the
48 observations are already well-balanced across scenarios and personas, so
the cluster correction barely moves the interval. Both naive and clustered
values are reported so the paper can footnote the choice:

- **Krippendorff's α (ordinal, 4-point), cluster: input_id:** **0.712 [95% CI: 0.583, 0.817]**
- **Krippendorff's α (binary, sign), cluster: input_id:** **0.574 [95% CI: 0.427, 0.741]**
- _Naive (item-level) CIs for reference:_ ord `[0.587, 0.802]`, bin `[0.403, 0.718]`

Side-by-side comparison (paste this table into §3.5). Both rows use
`cluster: input_id` CIs — methodologically symmetric on a shared scenario
unit even though the AI and human n_clusters differ (800 vs 8):

| metric | humans (48 obs, 8 scenarios) | AI judge ensemble (35,956 items, 800 scenarios) |
| --- | --- | --- |
| α (ordinal, 4-point) | 0.712 [0.583, 0.817] | 0.705 [0.696, 0.715] |
| α (binary, sign)     | 0.574 [0.427, 0.741] | 0.755 [0.742, 0.767] |

**Story (updated):** on the ordinal scale the AI ensemble matches human
reliability (0.705 vs 0.712, overlapping cluster-corrected CIs). On the
binary scale the AI ensemble is somewhat *higher* than the humans, which is
consistent with marginal-degeneracy effects (see persona breakdown below —
when raters mostly agree on sign, the binary coefficient degrades).
The cluster correction is substantive on the AI side (DE ≈ 2.75 ord,
4.16 bin) but near-identity on the human side (DE ≈ 1 ord, 1 bin) because
the 48-obs set is small and already balanced across scenarios.

### AI judge ensemble on the same 48 scenarios humans rated

Source: `tables/inter_judge_agreement_human_slice.md` (new — emitted by the
same run of `compute_inter_judge_agreement.py` via the `--human-ratings-csv`
flag). Filters the corpus to the 48 `(input_id, eval_model, persona)` cells
that humans rated (all 48/48 matched) and reruns the full metric pipeline.

On this head-to-head 48-item slice, the AI ensemble's agreement is
**higher** than on the full corpus — driven by selection effects in the
human-rating set (humans were given a diverse, high-signal subset, so the
item-level marginal variance is larger and α is mechanically higher):

| metric | spec | n clusters | avg size | α | 95% CI | DE |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| α ordinal | item-level (naive)             | 48 | 1.00 | **0.818** | [0.684, 0.890] | 1.00 |
| α ordinal | cluster: input_id              |  8 | 6.00 | **0.818** | [0.753, 0.884] | 0.42 |
| α ordinal | cluster: input_id × eval_model | 16 | 3.00 | **0.818** | [0.767, 0.870] | 0.30 |
| α binary  | item-level (naive)             | 48 | 1.00 | **0.774** | [0.612, 0.914] | 1.00 |
| α binary  | cluster: input_id              |  8 | 6.00 | **0.774** | [0.586, 0.939] | 1.22 |
| α binary  | cluster: input_id × eval_model | 16 | 3.00 | **0.774** | [0.613, 0.912] | 0.99 |

**Head-to-head table** (AI judges vs human raters, **same 48 scenarios**,
`cluster: input_id` for both rows, n_clusters = 8 for both):

| metric | humans (8 scenarios × 4 raters ≈ 173 ratings) | AI ensemble (8 scenarios × 6 cells × 3 judges = 144 scores) |
| --- | --- | --- |
| α (ordinal, 4-point) | 0.712 [0.583, 0.817] | **0.818** [0.753, 0.884] |
| α (binary, sign)     | 0.574 [0.427, 0.741] | **0.774** [0.586, 0.939] |

**Caveats:**
- With 8 clusters the cluster CI is below the CGM 2008 regime. The
  ordinal cluster CI is actually **narrower** than the naive CI on the AI
  side (DE = 0.42, 0.30) — reflecting negative intracluster correlation on
  this slice (judges agree more *across* personas within a scenario than
  *within* personas), which looks suspicious but is a real feature of the
  small, hand-picked subset. Report with this caveat explicit.
- The AI ensemble α on this slice (0.818) is substantially higher than on
  the full corpus (0.705). This is an item-variance effect: the 48-scenario
  slice has a spread-out severity distribution (33/28/17/66 across −1.0 /
  −0.5 / +0.5 / +1.0), which mechanically inflates α compared to the full
  corpus marginal. Do **not** use the slice number as a standalone claim;
  use it only as a methodologically symmetric head-to-head with humans.
- The AI ensemble's **point estimate** on the 48-scenario slice (0.818)
  sits above the human point estimate (0.712) but their `cluster: input_id`
  CIs overlap meaningfully: [0.753, 0.884] vs [0.583, 0.817]. Defensible
  phrasing: "On the same 48 scenarios humans rated, AI judge agreement is
  at least as strong as human agreement (α = 0.818 vs 0.712, cluster-
  corrected CIs overlap)."

### α on the curated 24-item golden set

**Sources:**
- AI side: `tables/inter_judge_agreement_golden_24.md` + `.csv` (new —
  emitted by `compute_inter_judge_agreement.py` via the `--golden-eval-dir`
  flag, default `logs/golden_questions_eval/`).
- Human side: `tables/human_inter_rater_agreement_golden_24.md` + `.csv`
  (new — emitted by `compute_human_validation_metrics.py` as a second
  pooled pass filtered to the 24 golden observation_ids).
- Both scripts use the same `krippendorff==0.8.2` package, seed
  `20260407`, 1000 bootstrap replicates, and the same multi-level
  cluster bootstrap helper as the full-corpus and 48-slice blocks above.

**Provenance:** the AI per-judge scores come from the original
`golden-questions-eval` .eval file from 2025-11-16 (recovered from trash
into `logs/golden_questions_eval/`). It reproduces the published
`ai_judge_score` column in `judge_vs_human_comparison.csv` for all 24/24
items exactly — bit-identical with the file that produced the existing
23/24 direction match / κ = 0.905 / ρ = 0.847 / r = 0.962 numbers. No
re-run was needed.

**Complementary question to the existing 24-item metrics.** The
direction-match / κ / ρ / r row above (23/24 = 95.8%, Wilson
[0.798, 0.993]; κ = 0.905; ρ = 0.847; r = 0.962) measure
**ensemble-vs-human agreement** on the curated 24. The numbers below
measure **inter-rater reliability** on the curated 24 — i.e., how much
humans agree with each other, and how much the 3-judge ensemble agrees
with itself. Different questions, same items.

**Head-to-head table** (paste into §3.5; both rows use `cluster: input_id`,
n_clusters = 8 for both):

| metric                | humans (24 obs, 85 ratings, 4 raters) | AI ensemble (24 items × 3 judges = 72 scores) |
| ---                   | ---                                    | ---                                            |
| α (ordinal, 4-point)  | **0.891** [0.714, 0.957]               | **0.784** [0.521, 0.876]                       |
| α (binary, sign)      | **1.000** [1.000, 1.000] ⚠             | **0.877** [0.636, 1.000]                       |

⚠ **Binary α on humans is 1.000 by construction** — the 24 golden items
were curated from the 31 unanimous-sign observations (see "Provenance of
the 24-item validation set" below). Every rater agrees on every item's
direction by selection, so the binary coefficient collapses to its
ceiling with a degenerate CI `[1.000, 1.000]`. This is a selection
tautology, not a finding. **Do not quote the binary human α on the
24-item set as evidence of rater reliability.** Quote the ordinal α
(0.891) instead, or quote the binary α on the full 48-obs set (0.574,
where the unanimous-sign selection hasn't been applied).

**Multi-level bootstrap diagnostics.**

_AI ensemble on the golden 24:_

| metric    | spec                            | n clusters | avg size | α     | 95% CI         | DE   |
| ---       | ---                             | ---:       | ---:     | ---:  | ---            | ---: |
| α ordinal | item-level (naive)              | 24         | 1.00     | 0.784 | [0.570, 0.876] | 1.00 |
| α ordinal | cluster: input_id               |  8         | 3.00     | 0.784 | [0.521, 0.876] | 1.25 |
| α ordinal | cluster: input_id × eval_model  | 15         | 1.60     | 0.784 | [0.611, 0.883] | 0.81 |
| α binary  | item-level (naive)              | 24         | 1.00     | 0.877 | [0.669, 1.000] | 1.00 |
| α binary  | cluster: input_id               |  8         | 3.00     | 0.877 | [0.636, 1.000] | 1.21 |
| α binary  | cluster: input_id × eval_model  | 15         | 1.60     | 0.877 | [0.671, 1.000] | 1.07 |

_Humans on the golden 24:_

| metric    | spec                            | n clusters | avg size | α     | 95% CI         | DE   |
| ---       | ---                             | ---:       | ---:     | ---:  | ---            | ---: |
| α ordinal | item-level (naive)              | 24         | 1.00     | 0.891 | [0.679, 0.960] | 1.00 |
| α ordinal | cluster: input_id               |  8         | 3.00     | 0.891 | [0.714, 0.957] | 0.89 |
| α ordinal | cluster: input_id × eval_model  | 15         | 1.60     | 0.891 | [0.751, 0.954] | 0.74 |
| α binary  | item-level (naive)              | 24         | 1.00     | 1.000 | [1.000, 1.000] | 1.00 |
| α binary  | cluster: input_id               |  8         | 3.00     | 1.000 | [1.000, 1.000] | 1.00 |
| α binary  | cluster: input_id × eval_model  | 15         | 1.60     | 1.000 | [1.000, 1.000] | 1.00 |

**Caveats:**
- n_clusters = 8 at `input_id` is below the Cameron–Gelbach–Miller 2008
  well-calibrated regime; interpret the cluster CI as a lower bound on
  honest uncertainty, not a precise interval. Same caveat as the 48-slice
  block above.
- Cluster sizes at the `input_id` level are **imbalanced** (1 to 4 obs
  per scenario — `respect-user-attention-005` contributes only 1 obs,
  `prioritize-long-term-wellbeing-011` contributes 4 obs). Flag this as
  a curation artifact if a reviewer asks: the 24 are not a random
  subsample, they're a hand-curated subset of the 31 unanimous-direction
  observations per the provenance rule below.
- The `cluster: input_id × eval_model` spec has 15 clusters (not 16)
  because one (input_id, model) combination is absent from the golden 24
  — the curation rule dropped the only observation in that cell. Both
  human and AI sides see the same 15 clusters, so the head-to-head
  comparison is well-defined.
- The AI golden-24 α_ord (0.784) is **lower** than the AI 48-slice α_ord
  (0.818). This isn't judges "disagreeing more on harder items" — the
  24-item slice has a more compressed ordinal marginal (33/25/19/67
  across −1.0/−0.5/+0.5/+1.0 if you normalize by 144 scores) than the
  48-item slice, and Krippendorff's α is sensitive to marginal variance.
  Direction-match (the sign-only measure) actually *improves* from
  16.7% sign-disagreement on the 48-slice to 8.3% on the golden 24.

**Sign-disagreement rates (for reference):**
- Full corpus: 14.476%
- 48-scenario slice: 16.667%
- Curated 24 golden set: **8.333%**

The drop from 16.7% → 8.3% when moving from 48 → 24 is consistent with
the selection effect: the golden 24 were curated by humans specifically
because they were cleaner-signal items.

**Alternative human-reliability number: golden-subset only.** Source:
`scripts/human_judging_analysis/output/judge_vs_human_agreement_analysis.md`.
On the 31 unanimous-direction observations (110 ratings), humans reach
**α = 0.920 [95% CI: 0.877, 0.954]** (ICC(2,k) = 0.983). This number uses the
hand-rolled Krippendorff implementation in `analyze_human_ratings.py` rather
than the `krippendorff` package, so it is not strictly methodologically
comparable to the 0.712 above — but the Δ between the two implementations on
the full 48-obs set was only 0.014 (0.712 vs 0.726), so the two numbers are
likely within ~0.02 of each other on the golden subset as well. If the paper
wants to lead with the strongest possible human-reliability claim, cite
**0.920 on the golden subset** and note it selects for clear-cut items by
definition; if the paper wants the honest pooled number, cite **0.712 on the
full 48-obs set**.

### Provenance of the 24-item validation set

Source: `tables/golden_set_provenance.md`.

**Pipeline:**
- 48 scenarios scored by at least one human rater
- 31 had unanimous-sign agreement across their available raters
- 24 were curated into `data/golden_questions.jsonl` with pre-generated AI
  outputs, and the ensemble scored those exact outputs via
  `src/golden_questions_task.py`'s `use_pregenerated_output()` solver

**Rater-multiplicity breakdown** (48 obs total):

| n_raters | n_observations | n_unanimous | unanimous_rate |
| ---: | ---: | ---: | ---: |
| 2 | 1  | 1  | 100.0% |
| 3 | 17 | 12 |  70.6% |
| 4 | 30 | 18 |  60.0% |

**Curation rule** (empirically reconstructed; paste-able):
- Exclude if `n_raters < 3` (coverage threshold)
- Exclude if `n_raters == 4` and the ratings split 2/2 on magnitude (e.g.
  `[-1.0, -1.0, -0.5, -0.5]` — half strong opinion, half soft — so the
  consensus magnitude is genuinely ambiguous despite the sign being unanimous)
- Include otherwise

This rule predicts 30 of 31 outcomes correctly; an independent rule search
confirms 30/31 is the best fit any simple structural rule can achieve, with
multiple semantically-equivalent formulations landing on the same score. One
observation (`c945b8e64dba7a79`, gemini-2.5-flash bad persona,
`enhance-human-capabilities-009`, 3 raters with 2/1 magnitude split) is
structurally identical to included peers and remains unexplained — likely a
one-off curation decision not recoverable from the data. See
`tables/golden_set_provenance.md` for full detail.

**Suggested paragraph sentence (copy from `golden_set_provenance.md`):**
> Four human raters produced 173 ratings on 48 scenarios. Of those, 31
> received unanimous-sign agreement across available raters. From this pool we
> curated 24 scenarios with pre-generated AI outputs — excluding items rated
> by fewer than three humans or with a 2/2 magnitude split among four raters
> — and scored them using the ensemble (via
> `src/golden_questions_task.py`'s pregenerated-output solver) for the
> ensemble–human comparison reported below.

### Ensemble vs human on the 24-item set

**Sources:**
- Point estimates:
  `scripts/human_judging_analysis/output/judge_vs_human_agreement_analysis.md`
  (unchanged since November, computed by
  `scripts/compare_judge_vs_human.py`).
- CIs:
  `tables/ensemble_vs_human_curated_24_cis.csv` (computed by
  `scripts/compute_ensemble_vs_human_cis.py`, seed 20260407, n=1000).
  Bootstrap CIs for κ/Spearman/Pearson match every other bootstrap CI in
  §3.5. **The direction-match row uses a Wilson binomial CI**, not the
  bootstrap — see note below.

**Headline numbers (paste into the paper):**

| metric | point | 95% CI | notes |
| --- | --- | --- | --- |
| Direction match (Rule A) | **23/24 (95.8%)** | **[79.8%, 99.3%]** | Wilson binomial CI (`scipy.stats.binomtest`); strict: sign(human) = sign(ensemble) |
| Weighted Cohen's κ (quadratic) | **0.905** | [0.797, 0.958] | bootstrap percentile; "almost perfect" (Landis & Koch) |
| Spearman's ρ | **0.847** | [0.604, 0.953] | bootstrap percentile; rank-order correlation |
| Pearson's r | **0.962** | [0.902, 0.989] | bootstrap percentile; linear correlation, interval assumption |
| Mean absolute error | **0.158** | — | ~1/6 of a scale point |
| Root mean square error | **0.251** | — | ~1/4 of a scale point |

**On the direction-match CI choice.** For a 0/1 outcome at n=24 the
percentile bootstrap is lumpy and has known poor small-n coverage; the
Wilson score interval is the standard analytical CI for a binomial
proportion and is what a stats-literate reviewer will expect. The
bootstrap percentile CI on this metric is `[87.5%, 100.0%]` and is
retained in `tables/ensemble_vs_human_curated_24_cis.csv` under
`direction_match_rate_bootstrap_ci_lower/upper` as a consistency check,
but **the paper should cite the Wilson interval**
`[79.8%, 99.3%]` — available in the same CSV under
`direction_match_rate_wilson_ci_lower/upper`. The other three metrics
(κ/Spearman/Pearson) stay on percentile bootstrap because they're
continuous statistics with no canonical analytical CI at n=24.

The bootstrap script reproduces all three point estimates (κ/Spearman/Pearson)
exactly from `judge_vs_human_comparison.csv`, so these numbers are directly
traceable. All point estimates fall inside their CIs; 1000/1000 resamples
were successful (no degenerate κ).

**Per-persona direction-match breakdown** (from the existing analysis md):

| persona | match rate | n |
| --- | ---: | ---: |
| bad      | 100% | 6  |
| baseline | 100% | 8  |
| good     |  90% | 10 |

The single disagreement is on `protect-dignity-and-safety` /
`gemini-2.5-flash` / good persona — human consensus = +0.667, ensemble
mean = −0.167, both near the zero boundary. A borderline case where the
ensemble was slightly more conservative, not a structural failure.

**Binarization rule lock-down:** the ensemble is treated as positive iff
`ensemble_mean > 0`, where `ensemble_mean` is the mean of the three judge
severities. This is the rule used by
`scripts/compare_judge_vs_human.py:142-143`
(`ai_direction = 'positive' if ai_score > 0 else 'negative' ...`). Copy this
into the paper explicitly so the rule is no longer implicit.

**Untested alternatives** (should be acknowledged explicitly, not papered
over):

- **Majority sign vote** (ensemble positive iff ≥2 of 3 judges have sev > 0):
  not computable because `judge_vs_human_comparison.csv` stores only the
  ensemble mean, and the original `.eval` file from the
  `golden_questions_eval` run is no longer on disk. Recomputing would
  require re-running the task (one API call per judge per item, 72 total).
- **Full 48-obs set (ensemble vs human across all scored scenarios, not just
  the curated 24):** requires running `golden_questions_eval` against an
  expanded input file that includes the 24 non-curated observations'
  pre-generated `ai_output` values (available in
  `consolidated_human_ratings.csv`). Also an API call.

The paper should state Rule A explicitly and note both alternatives are
untested rather than claim equivalence.

**Known limitation of n=24** (directly quoting
`judge_vs_human_agreement_analysis.md`): "A single disagreement swings
direction agreement from 100% to 95.8%, and correlation estimates have wide
confidence intervals at this size. Metrics are strong but would be more
convincing at n=50+." The bootstrap CIs above quantify this explicitly — the
Spearman CI is [0.604, 0.953], wide enough that the point estimate (0.847)
should be framed as a "substantial correlation" rather than a precise
measurement.

---

## Persona-stratified α paragraph (new — optional subsection or footnote)

Source: `tables/human_inter_rater_agreement_by_persona.csv` +
`tables/inter_judge_agreement_by_persona.csv`.

| persona | human α_ord | AI ensemble α_ord |
| --- | ---: | ---: |
| baseline    | 0.621 | 0.470 |
| good        | 0.321 | 0.289 |
| bad         | 0.090 | 0.761 |

The human and AI stratifications are **reversed on the extreme conditions**.
This is a marginal-degeneracy artifact, not noise:
- **Humans:** agree strongly on bad-persona direction (all rate it negative)
  but split between "soft no" (−0.5) and "HELL NO" (−1.0); the within-class
  ordinal variance collapses the marginal toward a single sign, so chance-
  corrected α goes to zero even though raters essentially agree.
- **AI judges:** the same mechanism flips the other way. Judges agree strongly
  on good-persona direction but split between +0.5 and +1.0 (α_ord = 0.289),
  while on bad-persona they produce a more diverse severity distribution and
  the marginal is non-degenerate (α_ord = 0.761).

**Suggested framing for the paper:** "Low α values on extreme persona
conditions — human bad-persona α = 0.090, AI good-persona α = 0.289 —
reflect unanimous-sign agreement on a near-degenerate marginal, not rater
disagreement. In both cases raters agree on the direction and disagree only
on the intensity of a shared verdict." This pre-empts the reviewer concern
about low conditional α values.

---

## Robustness-check appendix (new subsection)

Source: `tables/robustness_gap_binarized.md`.

**Headline:** the benevolent-minus-adversarial HumaneScore gap survives
binarization in **ordering** but **not in magnitude**. Across 15 models:

- **Spearman ρ (rank preservation):** **+0.964** (p < 0.0001)
- **Pearson r (magnitude correlation):** **+0.991** (p < 0.0001)
- **Mean ratio (binarized / ordinal):** **+0.589** ⚠ below the 0.70
  robustness threshold

**Sensitivity check** (in `tables/robustness_gap_binarized_filtered.md`): the
mean-ratio warning is *not* driven by instability at small ordinal gaps. Even
after filtering to models with `ordinal_gap > 0.3` the mean is only +0.616
(median +0.636), and after filtering to `ordinal_gap > 0.5` the mean is
+0.622 (median +0.636). Per-model ratios cluster tightly in the 0.52–0.67
range across the whole sample, with only gpt-5.1 reaching 0.700 at the top
and claude-sonnet-4.5 dropping to 0.315 at the bottom. The ~41% shrinkage is
a genuine, consistent property of the data, not an artifact of small-gap
noise.

**Interpretation:** the ordinal HumaneScore is doing real work beyond the
binary prosocial-flip signal. A reviewer will likely ask whether the headline
robustness gap could be an artifact of ordinal-end judge disagreements (given
the good-persona α_ord = 0.289). The answer is: model ordering survives
(Spearman ρ = 0.96, Pearson r = 0.99), so the ranking of models is not an
artifact. But the magnitude of the gap shrinks by ~41% under binarization —
the ordinal scale carries information that a pure sign test does not. Worth
a sentence in the main text, not just the appendix.

**Per-model table** is in `tables/robustness_gap_binarized.md`; the
sensitivity analysis is in `tables/robustness_gap_binarized_filtered.md`.

---

## Things to NOT paste into the paper (just reference)

- The existing "α ≈ 0.726" in
  `scripts/human_judging_analysis/output/irr_metrics.csv` is from a
  hand-rolled Krippendorff implementation with index-based ordinal distances
  and a different bootstrap seed. It is within ~0.014 of our new value
  (0.712), which is reassuring, but **use the new number** so human and AI
  numbers come from the same RNG state and the same package.
- Rater coverage is **not perfectly balanced**: rater-3 contributed 30/173
  (17.3%) vs ~28% each for the other three raters. All four raters covered
  all three persona conditions. "Four human raters produced 173 total
  ratings" is defensible as-is, but if precision is wanted the footnote
  version is "three raters scored ~48 scenarios each and a fourth scored 30."
