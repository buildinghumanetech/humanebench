# HumaneBench — supplementary material

Code, data, and derived tables for the submission. Everything here is
self-contained: no downloads, no API keys, no network access is needed to
reproduce any number in the paper.

The one thing this package does **not** contain is the raw evaluation logs.
They are 45 Inspect `.eval` archives totalling 584 MB, already compressed, so
they cannot be brought under the submission size limit. Every analysis that
read them now also reads the long-format table it used to emit, and both routes
are verified to produce byte-identical output. What that means for what you can
and cannot check is spelled out in **Provenance** below.

---

## Layout

```
README.md                  this file
supplement.pdf             the Supplementary Document (compiled appendices)
requirements.txt           pinned environment
pytest.ini                 test markers

humanebench/               library: scorer, bootstrap, provenance, exclusions
src/                       Inspect task definitions the runs were launched from
scripts/                   one script per paper table or figure
tests/                     unit tests (no API access required)
rubrics/                   the scoring rubric the judges were shown
docs/                      design notes referenced by the methods section

data/
  humane_bench.jsonl       the 800-scenario benchmark
  golden_questions.jsonl   24-item human-agreement validation set
  human_ratings/           four raters' scores on the 48-scenario slice
  decomposition/           frozen 200-scenario subsample for conditions C/D/E
  discriminant/            multi-label judgements for the discriminant analysis

data_generation/           generation pipeline + the principle-embedding cache

tables/                    every derived table the paper cites
  inter_judge_raw_regenerated.csv.gz   per-judge severities: 106,248 rows,
                                       the input that stands in for the logs
  inter_judge_raw_stats.json           log-scan counts for the three passes
  decomposition/           goal-vs-tactics results
  discriminant/            designed x scored matrix, original 12-per-principle draw
  discriminant_expansion/  fresh 24-per-principle redraw
  discriminant_pooled/     pooled 36-per-principle family + the domain-stratified probe
  discriminant_rules27/    rules-suppressed replication (five of seven global rules removed)
  discriminant_canary/     judge-drift gate: 96 byte-identical calls re-issued
  lopo_*.csv               leave-one-principle-out sensitivity
  emc_pltw_*.csv           EMC/PLTW column non-redundancy checks

provenance/                per-run hashes + the independent verifier's inputs
figures/                   paper figures and their alt text
helm_integration/          HELM capability scores and the scraper
results/
  rubric_appendix.md              the rubric, extracted verbatim from the logs
  serving_sensitivity_rows.csv.gz one row per scored generation call with the
                                  provider that served it, both run periods
  decomposition_precommitment.md  interpretation fixed before any
                                  decomposition run existed
```

## Environment

Developed and run on **Python 3.13.3**. The syntactic floor is 3.10 (several
modules use `X | None` annotations that are evaluated at runtime), but only
3.13 was tested.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest -m unit          # no network, no API keys
```

Run every command below from the root of this package.

**Write recomputed tables somewhere else.** Several scripts emit files whose
names match the ones shipped in `tables/`, and two of them read those same
files as input. Left to the defaults they would overwrite the published numbers
you are trying to compare against — so the commands below direct output to
`reproduced/`, and you diff that against `tables/`:

```bash
mkdir -p reproduced
diff tables/inter_judge_agreement.csv reproduced/inter_judge_agreement.csv
```

## Reproducing the paper

`tables/inter_judge_raw_regenerated.csv.gz` is the pivot: it holds one row per
(scenario, evaluated model, persona, judge) with that judge's severity, which
is everything the ensemble score is computed from. Most scripts read it
directly. Gunzip it once if you prefer an uncompressed copy — the scripts read
either form, and `pandas` handles `.gz` transparently:

```bash
gunzip -k tables/inter_judge_raw_regenerated.csv.gz
```

Bootstrap CIs use seed `20260407` and 1,000 replicates
(`humanebench/bootstrap.py`), so a rerun reproduces the published intervals
exactly, not merely closely. Expect the alpha bootstraps to take a few minutes;
pass `--n-bootstrap 0` for point estimates only.

### Script -> paper map

Every Python file in this package carries a `# Paper:` header naming the
paper section, table, or figure it implements or produces (or stating that it
is a utility with no paper-facing output). The table below indexes the
paper-facing analysis scripts in one place; the per-file headers carry the
same mapping next to the code. The main paper's headings are unnumbered; the
only numbered object is Table 1, the 15-model master results table.

| Script | Paper artifact it produces / implements |
| --- | --- |
| `scripts/compute_score_cis.py` | Table 1 scores, paired deltas, CIs; the 10/15 flip count ("Overall Performance", "The Anti-Humane Flip") |
| `scripts/compute_cohort_principle_cis.py` | supplement "Per-Principle Cohort Scores" |
| `scripts/compute_inter_judge_agreement.py` | Krippendorff's alpha / Cohen's kappa (scoring section; supplement "Judge Validation Details") |
| `scripts/compute_human_validation_metrics.py`, `scripts/compute_ensemble_vs_human_cis.py`, `scripts/compare_judge_vs_human.py` | human-validation numbers (scoring section; supplement "Judge Validation Details") |
| `scripts/compute_loo_sensitivity.py` | leave-one-judge-out ablation (supplement "Judge Validation Details") |
| `scripts/compute_judge_self_preference.py` | judge-independence paragraph (supplement "Judge Validation Details") |
| `scripts/compute_rubric_sensitivity.py`, `scripts/compute_binarized_robustness_gap.py` | rubric / threshold sensitivity checks |
| `scripts/compute_interprinciple_correlation.py` | supplement "Cross-Model Correlation and Construct Redundancy" |
| `scripts/compute_discriminant_validity.py`, `scripts/compute_discriminant_pairwise.py` | designed x scored separability (benchmark section) |
| `scripts/build_discriminant_expansion_frame.py`, `scripts/build_discriminant_pooled_tables.py` | fresh 24-per-principle redraw and the pooled 36-per-principle family |
| `scripts/compute_discriminant_rules_comparison.py` | rules-suppressed replication (benchmark section) |
| `scripts/build_discriminant_canary.py`, `scripts/compute_discriminant_canary.py` | judge-drift gate for pooling |
| `scripts/compute_discriminant_domain_pairwise.py` | supplement "Domain-Stratified Pairwise Probe" |
| `scripts/simulate_discriminant_power.py` | power context for the original draw's five underpowered pairs |
| `scripts/compute_lopo_sensitivity.py` | supplement "Leave-One-Principle-Out Sensitivity" |
| `scripts/compute_emc_pltw_nonredundancy.py`, `scripts/compute_emc_pltw_correlation_check.py` | EMC/PLTW retention decision (benchmark section) |
| `scripts/compute_decomposition.py` | "Engagement Pressure Alone Drives Degradation"; supplement "Dose-Response Across Adversarial Wordings" |
| `scripts/compute_vp_table5.py`, `scripts/vp_tables_and_figure.py` | supplement "Vulnerable-Population Analysis" + the age-gradient figure |
| `scripts/compute_helm_power.py` | minimum-detectable-effect caveat ("Intelligence != Humaneness") |
| `scripts/create_helm_delta_scatter.py` | supplement "HELM Capability Scatter" figure |
| `scripts/create_steerability_chart.py` | supplement "Per-Model Steerability Ranges" figure |
| `scripts/compute_serving_provenance.py` | supplement "Serving-Variation Sensitivity" incl. the provider-mix table |
| `scripts/compute_dataset_composition.py`, `scripts/compute_similarity_distributions.py` | scenario statistics (benchmark section; supplement "Scenario Construction Pipeline") |
| `scripts/tag_excluded_prompts.py` | supplement "Excluded Scenarios: Confabulation Audit" |
| `scripts/extract_rubric_appendix.py` | supplement "Complete Scoring Rubric" (verbatim extraction + checks) |
| `scripts/build_provenance.py`, `scripts/verify_provenance.py`, `scripts/verify_discriminant_provenance.py` | provenance manifest and its independent verifier |
| `src/baseline_task.py`, `src/good_persona_task.py`, `src/bad_persona_task.py` | the three evaluation conditions |
| `src/decomp_b_xml_objective_task.py` ... `src/decomp_e_abtest_task.py` | decomposition conditions B-E |
| `humanebench/scorer.py`, `humanebench/bootstrap.py`, `humanebench/humane_patterns.py`, `humanebench/excluded.py`, `humanebench/provenance.py` | the judge ensemble, Eq. 2 + the bootstrap protocol, the rubric source, the 12-item exclusion, provenance hashing |

### Headline scores and steerability

```bash
python scripts/compute_score_cis.py
# -> tables/score_cis_long.csv, tables/persona_delta_cis_long.csv
```

Per-model HumaneScore under each persona, the baseline→adversarial delta, and
their CIs. The anti-humane flip count is a cohort statistic over this grid.

### Per-principle cohort means (supplement: "Per-Principle Cohort Scores")

```bash
python scripts/compute_cohort_principle_cis.py
# -> tables/cohort_principle_cis.csv, tables/cohort_principle_delta_cis.csv
```

### Inter-judge agreement (Krippendorff's alpha, Cohen's kappa)

```bash
python scripts/compute_inter_judge_agreement.py \
    --raw-csv tables/inter_judge_raw_regenerated.csv.gz \
    --tables-dir reproduced/
# -> reproduced/inter_judge_agreement.{md,csv}
#    + _by_persona / _by_principle / _by_model
#    + the 48-scenario human slice and the golden-24 set
# --tables-dir is required here: this pass reads
# tables/inter_judge_raw_human_slice.csv and _golden_24.csv, which are also
# output names, so the default would consume its own inputs. The script
# refuses rather than letting that happen.
```

All three passes run without logs. The human slice and golden-24 sets ship
pre-filtered (`tables/inter_judge_raw_human_slice.csv`,
`tables/inter_judge_raw_golden_24.csv`) because the inputs they would otherwise
be derived from — a rater spreadsheet and a separate log directory — are not
distributable.

The sample counts printed in these tables (36,000 scanned, 44 without
individual judge scores, 45 files read) describe the original log scan and
come from `tables/inter_judge_raw_stats.json`. They are carried, not
recomputed; the loader refuses to run if that file and its CSV disagree on how
many samples were included.

### Leave-one-judge-out sensitivity

```bash
python scripts/compute_loo_sensitivity.py \
    --raw-csv tables/inter_judge_raw_regenerated.csv.gz \
    --output-dir reproduced/
# -> reproduced/loo_sensitivity.md, loo_model_scores.csv,
#    loo_cohort_counts.csv, loo_alpha.csv
```

Two of the three judges belong to the robust set, so this drops each judge in
turn and recomputes the flip count, robust-set membership, and alpha.

### Judge self-preference

```bash
python scripts/compute_judge_self_preference.py
# -> tables/judge_self_preference.{md,csv}, judge_relative_generosity_matrix.csv,
#    single_judge_model_scores.csv, loo_config_change.csv
```

### Rubric and threshold sensitivity

```bash
python scripts/compute_rubric_sensitivity.py
python scripts/compute_binarized_robustness_gap.py
# -> tables/rubric_sensitivity.md, tables/robustness_gap_binarized.{md,csv}
```

### Inter-principle correlation

```bash
python scripts/compute_interprinciple_correlation.py
# -> tables/interprinciple_correlation.{md,csv}
```

### Discriminant validity (designed x scored matrix)

```bash
python scripts/compute_discriminant_validity.py \
    --raw-csv tables/inter_judge_raw_regenerated.csv.gz \
    --output-dir reproduced/
# -> tables/discriminant/*.csv
```

Reads the multi-label judgements in `data/discriminant/multilabel_*.jsonl`.

### Separability follow-up: pooled family, rules suppression, domain probe

The pooled 36-per-principle result reported in the paper's benchmark section
merges the original 12-per-principle draw with a fresh 24-per-principle
redraw, gated by a judge-drift canary:

```bash
python scripts/build_discriminant_pooled_tables.py
# -> tables/discriminant_pooled/matrix_long.csv
python scripts/compute_discriminant_pairwise.py --tables-dir tables/discriminant_pooled
# -> tables/discriminant_pooled/pairwise_interactions.csv  (27/28 Holm-separable)
python scripts/compute_discriminant_rules_comparison.py
# -> tables/discriminant_rules27/comparison_pairs.csv  (rules-suppressed replication)
python scripts/compute_discriminant_domain_pairwise.py --min-per-side 5
# -> tables/discriminant_pooled/domain_pairwise.csv
#    (supplement "Domain-Stratified Pairwise Probe"; the shipped table was
#     produced with the gate at 5 scenarios per side, hence the explicit flag)
python scripts/simulate_discriminant_power.py
# -> tables/discriminant/power_simulation.csv
```

### Leave-one-principle-out sensitivity

```bash
python scripts/compute_lopo_sensitivity.py
# -> tables/lopo_model_scores.csv, lopo_cohort_counts.csv,
#    lopo_status_changes.csv, lopo_decomposition.csv
```

### EMC/PLTW non-redundancy

```bash
python scripts/compute_emc_pltw_nonredundancy.py
python scripts/compute_emc_pltw_correlation_check.py
# -> tables/emc_pltw_*.csv
```

### Goal-vs-tactics decomposition

```bash
python scripts/compute_decomposition.py
# -> tables/decomposition/decomposition_summary.md and its CSVs
```

This one needs the decomposition `.eval` logs, which are not included; the
computed outputs ship in `tables/decomposition/` instead. Read
`results/decomposition_precommitment.md` first — it fixes the interpretation of
both possible outcomes, and was written before any decomposition run existed.

### Vulnerable populations (supplement: "Vulnerable-Population Analysis")

```bash
python scripts/compute_vp_table5.py
python scripts/vp_tables_and_figure.py
# -> tables/vp_table5.csv, vp_table5_by_principle.csv,
#    vp_table6_age_gradient.csv, figures/vp_age_gradient.{pdf,png}
```

### HELM capability correlation and its power (main paper: "Intelligence != Humaneness")

```bash
python scripts/compute_helm_power.py
# -> tables/helm_power.md
python scripts/create_helm_delta_scatter.py
# -> figures/helm_vs_humanescore_scatter.{pdf,png}
```

The minimum detectable effect is the number that matters here: with 15 models
the correlation is underpowered, and the script reports by how much.

### Serving provenance

Regenerating this needs the logs, which are not included, so there is no
command to run here — the script exits non-zero rather than overwriting the
shipped tables with empty ones. The results ship as
`tables/serving_provenance.md` plus the per-call rows behind it:
`serving_provenance_responses.csv` (36,000 generation calls, one per
scenario x model x persona) and `serving_provenance_judges.csv`. Six of the
fifteen models were served by more than one upstream provider at a share of at
least 0.5%, so those cells are a mixture of serving stacks rather than one
system — the per-call table lets you check that claim yourself, including the
floor: below it a second provider is a routing blip, not a mixture, and the
full unfiltered mixture is printed in the report regardless.

Generation calls are identified by **position** — the first model event of a
sample — and never by model slug. All three judges are themselves evaluated
models, so a slug comparison books a model's self-judge call as a generation
call. The row counts make this checkable: exactly 2,400 rows for every model,
the three judges included.

The supplement's per-model provider-mix table (both run periods) is computed
from `results/serving_sensitivity_rows.csv.gz` — one row per scored generation
call across the main runs and the decomposition runs, with the provider and
`system_fingerprint` read from the raw API response.

### Dataset composition and near-duplicate rate

```bash
python scripts/compute_dataset_composition.py
python scripts/compute_similarity_distributions.py
```

The second reads `data_generation/cache/principle_embeddings.npz`, included so
the near-duplicate rate can be recomputed without regenerating embeddings.

## Provenance: what you can check here, and what you cannot

`provenance/MANIFEST.json` records, for each of the 45 reported runs, the
`.eval` file's SHA-256, its embedded prompt hash, its creation timestamp, and
the result of each binding check. The verifier runs against what is included:

```bash
python scripts/verify_provenance.py
```

**Checkable from this package alone.** That the 800-prompt dataset hashes to
the frozen prompt hash; that the frozen hash is reproducible from the dataset
by a three-line script that uses none of our code (`PROVENANCE.md` gives it);
that every derived table follows from the shipped per-judge severities; that
the analysis code does what the paper says.

**Not checkable here.** That each `.eval` file hashes to the SHA-256 the
manifest records, and that its embedded prompts hash to the frozen value.
Those checks need the log bytes. The verifier reports them as SKIP rather than
passing them silently, so the output tells you exactly which guarantees are
suspended. The manifest's per-run hashes are a commitment made in advance: the
logs are deposited in a public archive on publication, and a hash recorded now
cannot be satisfied by a log written later.

`results/decomposition_precommitment.md` is a related case. Its priority does
not rest on the file's own timestamp — the document explains what does
establish it, and points at the tracked artifacts that carry the claim.

## What is deliberately not here

- **The `.eval` logs** (584 MB compressed). Deposited on publication.
- **Model response text.** It lives only in the logs. The shipped tables carry
  severities, not generations.
- **Individual annotator records.** `data/human_ratings/` is pseudonymised;
  the raw rater spreadsheets and the consolidation cache are not included.
- **Author, institution, and repository identifiers**, removed for
  dual-anonymous review and restored in the archival release. `LICENSE` is reduced to
  its SPDX identifier for the same reason; the grant is unchanged Apache-2.0.

## Configuration the numbers depend on

| Setting | Value |
| --- | --- |
| Judge ensemble | `claude-4.5-sonnet`, `gpt-5.1`, `gemini-2.5-pro`, via OpenRouter |
| Judge temperature | 0 |
| `score_attempts` | 3 |
| Severity scale | ordinal, {-1.0, -0.5, +0.5, +1.0} |
| Ensemble rule | mean of the three judges; a sample with a partial complement is dropped |
| Bootstrap seed | `20260407` |
| Bootstrap replicates | 1,000 |
| Resampling unit | scenario (`input_id`), shared across every cell of the model x persona grid |
| Excluded items | 12 of 800, flagged `excluded_from_analysis` in `data/humane_bench.jsonl` |
| Decomposition subsample seed | `20260407`, stratified principle → vulnerable-population → domain |

Judge models are addressed through unversioned endpoints, so re-running the
evaluation will not reproduce identical judge outputs. This is why the analysis
ships with the severities it scored rather than a promise that rescoring would
agree — and why `PROVENANCE.md` identifies runs by content hash rather than by
timestamp or model version.
