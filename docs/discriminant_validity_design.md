# Principle discriminant validity: the designed × measured design

Methodology note for the multi-label principle-scoring analysis implemented by
`scripts/build_discriminant_multilabel_dataset.py`,
`src/discriminant_multilabel_task.py` and
`scripts/compute_discriminant_validity.py`.

This is the public statement of the design and of the interpretation committed
to before the run. It is the document those modules cite.

---

## 1. The question

HumaneBench assigns each scenario exactly one of eight principles. Whether those
eight are eight distinct constructs, or a smaller number wearing eight labels, is
not answerable from the stored single-label scores.

## 2. Why not factor analysis

The obvious approach — score a subsample against all eight principles and factor
the resulting item × dimension matrix — does not work here, and the reason is
specific rather than general.

Feuer et al., *When Judgment Becomes Noise: How Design Failures in LLM Judge
Benchmarks Silently Undermine Validity* (arXiv:2509.20293), applied factor
analysis to Arena-Hard-Auto's five-criterion rubric across four judges and found
**factor correlations above 0.93 for most criteria** and roughly **55% of
judgement variance unexplained by the stated rubric**, exceeding 90% for some
judges. They call this *factor collapse*: LLM judges collapse semantically
distinct criteria into a single latent dimension.

So a high-correlation factor result on our matrix would have two explanations
that the analysis cannot separate:

- **(a)** the eight principles genuinely overlap;
- **(b)** the judge collapsed them, as judges are documented to do.

Since (b) is the field's default expectation for LLM-judge rubrics, such a result
is uninformative at best. Reported as loadings, it would be evidence against the
instrument as a whole rather than against the taxonomy specifically.

**No EFA is run. No factor count is chosen. No rotation is applied.** The
inter-principle correlation matrix the analysis does report is labelled
descriptive-only, with this caveat attached, precisely because it cannot settle
the question.

## 3. The design used instead

Every scenario was **designed** to probe one principle. Multi-label scoring tells
us which principle it actually **measures**. Build an 8 × 8 matrix:

- **rows** = the principle a scenario was designed for
- **columns** = the principle it was scored against
- **cells** = mean judge severity

If the principles discriminate, the diagonal is systematically more negative than
the off-diagonal *in its own row*: a scenario built to stress Respect User
Attention should surface attention failures more than equity failures. If the
diagonal is indistinguishable from its row, the scenarios elicit generally good
or generally bad responses and the principle labels are decorative.

This is a **known-groups validity** design. Its value is that the statistic is a
*within-row contrast*, which is robust to the three things that ruin the
alternatives:

| Threat | Why the contrast survives |
|---|---|
| General model-quality factor | Shifts every cell in a row equally; cancels in the diagonal-vs-row contrast |
| Judge factor collapse (Feuer et al.) | Same — collapse compresses the spread but does not move the diagonal preferentially |
| Range restriction | The contrast is within-row, so a ceiling or floor affects both terms |

It needs no rotation, no factor-count decision, and no psychometric model.

## 4. Specification

**Sampling.** 12 scenarios per principle, 96 total, stratified within principle by
topical domain. Domain stratification is not optional: the principle/domain
association is strong (Cramér's V = 0.432), so without it the matrix would partly
measure domain rather than principle. Drawn with a fixed seed from the frozen
decomposition subsample, so the scenarios nest inside that arm rather than
forming a separate incompatible set.

**Condition: baseline responses only.** Three reasons.

1. The variance is already there — cohort baseline principle means span a wide
   range, and the question is whether a principle's scenarios score low
   *specifically* on that principle or low on everything.
2. No adversarial contamination. Under an engagement-maximizing system prompt a
   model may violate everything at once, flattening the matrix and destroying the
   contrast.
3. It sidesteps rubric/prompt vocabulary overlap. The adversarial system prompt
   shares vocabulary with some rubrics; at baseline there is no system prompt, so
   that overlap cannot drive anything.

**Models.** Three, spanning the robustness range — one robust, one mid, one that
fails under adversarial pressure. Three rather than one guards against the matrix
being an artifact of a single model's response style.

**Judging.** One judge, temperature 0, and **two details that are load-bearing**:

- **Eight separate judge calls per response, one per principle. Never one call
  scoring all eight.** Analytic (per-criterion) scoring is the documented defence
  against the criterion conflation and halo that holistic scoring induces;
  scoring all eight in one call would manufacture exactly the correlation
  structure the analysis exists to measure.
- **The existing per-principle rubric scaffolds are reused verbatim.** No new
  rubric text is written anywhere. The scaffold is a deterministic function of
  the principle, which is what makes this checkable: the eight scaffold hashes
  the run produces are compared against those of the reported runs.

The judge is chosen from the main run's own ensemble, excluding any model whose
responses are being scored. This makes the diagonal a *same-judge* replication of
the main-run procedure, which turns the sanity check below into a genuine
judge-drift test rather than a judge comparison.

## 5. What is reported

1. **Mean diagonal minus mean off-diagonal, per row and pooled, with
   scenario-level cluster-bootstrap CIs.** The headline. Negative means the
   principles discriminate. One scenario draw per row is carried across all eight
   columns and all source models, preserving the within-row pairing the contrast
   depends on.
   1b. The same contrast **column-centred**, which additionally removes
   per-rubric leniency — the objection the raw contrast cannot answer, that a
   principle's diagonal looks low only because its rubric is the harshest. This
   is the two-way additive residual contrast; the *pooled* figure needs no such
   correction, since an additive column offset cancels exactly in the pool.
2. **Rank of the diagonal cell within its row, and within its column.** Ordinal,
   so it survives any monotone distortion of a four-point scale.
3. **A named principle pair**, both directions, as a paired difference.
4. **Sanity check against the main run**, split into the two questions it would
   otherwise conflate: against the main run's own single-judge severity (same
   judge, same prompt, months apart — a drift test), and against the three-judge
   ensemble mean (does single-judge multi-label scoring reproduce the primary
   procedure). A gap in the second is judge composition, not drift.
5. **Inter-principle correlations at the item level**, descriptive only, with the
   Feuer et al. caveat stated in the same breath.

## 6. Interpretation, committed before the run

Written down in advance and selected between *in code* from the numbers, not in
prose after seeing them.

**If the diagonal is distinct** (contrast negative in most rows, CI excluding
zero): the principles measure what they were designed to measure. Report as
construct-validity evidence; note that inter-principle correlations are high but
that this is expected both for related normative constructs and as documented
LLM-judge behaviour, and that the diagonal contrast is robust to both.

**If the diagonal is distinct for some principles and not others** — the most
likely outcome: report per-principle. Flat rows are candidates for merging in a
future revision or for a stated limitation. Conceding a merge candidate while
defending the rest is more credible than defending all eight.

**If the diagonal is flat everywhere**: the scenarios do not discriminate between
principles. The honest move is to report that, reframe the eight as facets of a
single humaneness construct rather than eight distinct constructs, and revise the
taxonomy claim accordingly. This does not touch the headline robustness finding,
which does not depend on the principles being separable. **This outcome is not to
be suppressed.**

A **reversed** row — the diagonal scoring *higher* than its own row — was not
anticipated by any of the three. If it occurs it is reported as-is and left open
rather than folded into the nearest category.

Note the asymmetry worth stating in any write-up: a flat diagonal under LLM-judge
scoring is consistent with *either* genuine construct overlap *or* judge
collapse, so a null bounds what the instrument can resolve rather than refuting
the framework. A non-flat diagonal carries no such ambiguity.

## 7. Limitations

12 scenarios per principle is modest — 36 observations per cell across three
models. Adequate for a mean with a CI, thin for anything finer. The n per cell is
stated in the table caption. Where per-row contrasts are noisy but the pooled
contrast is not, the pooled number leads.

The matrix uses a different procedure from the main benchmark (multi-label rather
than single-label scoring), so it could be called a methods artifact. This is
mitigated by reusing the rubric scaffolds verbatim and by reporting the diagonal
against the main-run scores for the same scenarios — the diagonal *is* the
main-run procedure, and `scripts/verify_discriminant_provenance.py` checks that
byte-for-byte on every diagonal prompt.

If the run cannot complete, the analysis refuses to emit a matrix rather than
publishing a partial one. The fallback evidence is the prompt-level semantic
separation already computed (between-principle cosine 0.152 ± 0.076 vs
within-principle 0.177–0.274), which establishes that the *scenarios* are
distinguishable rather than that the *measurements* are — a weaker claim, and
stated as such.
