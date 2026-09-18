#!/usr/bin/env python3
"""Principle discriminant validity: the designed x measured matrix.

Peer review raised whether the eight principles are eight distinct constructs
or one construct wearing eight labels. Every scenario was *designed* to probe one
principle; scoring each response against all eight tells us which principle it
actually *measures*. Rows are the designed principle, columns the scored one.

WHY THIS AND NOT FACTOR ANALYSIS
--------------------------------
Feuer et al. (arXiv:2509.20293) factored Arena-Hard-Auto's five-criterion rubric
across four judges and found factor correlations above 0.93 and roughly 55% of
judgement variance unexplained by the stated rubric. LLM judges collapse
semantically distinct criteria into one latent dimension. So a high-correlation
factor result here would have two explanations the analysis cannot separate: the
principles genuinely overlap, or the judge collapsed them. Since the latter is
the field's default expectation, such a result would be uninformative at best
and, reported as loadings, would hand a reviewer a stick for the whole benchmark.

The contrast used instead is *within a row*: the diagonal cell against the seven
others in its own row. A general factor -- model quality, judge leniency, factor
collapse -- shifts every cell in a row together and cancels out of that
difference. Range restriction likewise affects both terms.

**No EFA is run here. No factor count, no rotation.** The inter-principle
correlation matrix below is reported as descriptive context only, with that
caveat attached, precisely because it cannot settle the question.

WHAT IS REPORTED
----------------
1. Mean diagonal minus mean off-diagonal, per row and pooled, with scenario-level
   cluster-bootstrap CIs. Negative = the principles discriminate.
   1b. The same, column-centred, which additionally removes per-rubric leniency.
2. Rank of the diagonal cell within its row, and within its column.
3. The Foster Healthy Relationships / Prioritize Long-term Wellbeing cells --
   the pair named in review -- both directions, as a paired difference.
4. Sanity check against the main run, split into the two questions it conflates:
   (a) vs the main run's own gpt-5.1 severity -- same judge, same prompt, eight
       months apart. A judge-drift test.
   (b) vs the main run's 3-judge ensemble mean -- does single-judge multi-label
       scoring reproduce the primary procedure.
5. Inter-principle correlations at the item level, descriptive only.

Inputs (read-only):
  - logs/discriminant/<model>/*.eval          (the multi-label run)
  - data/discriminant/manifest.json
  - tables/inter_judge_raw_regenerated.csv    (main-run per-judge severities)

Outputs (written to --output-dir, default tables/discriminant/):
  - matrix_pooled.csv, matrix_<model>.csv, matrix_long.csv
  - row_contrasts.csv, diagonal_ranks.csv, fhr_pltw.csv
  - sanity_vs_main_run.csv, judge_drift_confusion.csv
  - interprinciple_correlation_item_level.csv
  - ../../results/discriminant_validity.md

Run from repo root, after scripts/verify_discriminant_provenance.py passes:
    python scripts/compute_discriminant_validity.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench import provenance as prov  # noqa: E402
from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    CI_HIGH_PCT,
    CI_LOW_PCT,
    N_BOOTSTRAP_DEFAULT,
    PRINCIPLES,
    DesignedMeasuredMatrix,
    bootstrap_designed_measured_matrix,
    diagonal_ranks,
    discriminant_contrasts,
)
from humanebench.discriminant import (  # noqa: E402
    IDS_PATH,
    LOG_CONDITION,
    MANIFEST_PATH,
    PARENT_IDS_PATH,
    PRINCIPLE_SHORT,
    SOURCE_MODELS,
    load_overseer_template,
)

FHR = "foster-healthy-relationships"
PLTW = "prioritize-long-term-wellbeing"
ORDINAL_LEVELS = (-1.0, -0.5, 0.5, 1.0)
JUDGE_SHORT = "gpt-5.1"


def _rel(path: Path, logs_dir: Path | None = None) -> str:
    """A path for the report: repo-relative, or ``logs/...`` for out-of-tree logs.

    Never an absolute path. The report names its input logs, and `--logs-dir` may
    point outside the checkout (a worktree reading the main clone's archives), so
    an absolute fallback would print a home directory into a document that gets
    pasted into the paper -- the local-path leak
    ``anonymization_redaction_list.txt`` exists to catch.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        if logs_dir is not None:
            try:
                return str(Path("logs") / path.relative_to(logs_dir))
            except ValueError:
                pass
        return path.name


def _n_scored(path: Path) -> int:
    """Samples carrying a full, on-scale severity -- the analysis admission rule."""
    n = 0
    for sample in prov.iter_eval_samples(path):
        overseer = (sample.get("scores") or {}).get("overseer")
        if not overseer:
            continue
        meta = overseer.get("metadata") or {}
        individual, judges = meta.get("individual_scores"), meta.get("ensemble_models")
        if not individual or not judges or len(individual) != len(judges):
            continue
        if all(s in ORDINAL_LEVELS for s in individual):
            n += 1
    return n


def select_eval(paths: list[Path]) -> Path:
    """Pick the same `.eval` the runner's completeness gate blessed.

    The gate uses `best_eval`, which ranks on `score_census(...)["n_fully_scored"]`.
    Ranking on raw sample count instead would diverge whenever more than one log
    survives in a model directory -- an interrupted run before `archive_superseded`
    filed the retry into `attic/`, or a hand-run `inspect eval`. A retry log with
    all 768 samples but 740 scored ties on sample count and wins on recency, so
    the gate would bless one file while the matrix was built from another.
    """
    return max(paths, key=lambda p: (_n_scored(p), p.name))


# --- loading -----------------------------------------------------------------
def _resolve_attachment(text, attachments: dict) -> str:
    """Inspect stores long strings out-of-line as ``attachment://<hash>``."""
    if isinstance(text, str) and text.startswith("attachment://"):
        return str(attachments.get(text[len("attachment://"):], text))
    return "" if text is None else str(text)


def load_run_scores(logs_dir: Path,
                    models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Long table of the multi-label run, one row per judged call.

    Admission mirrors ``compute_inter_judge_agreement.collect_long_table``: a
    sample counts only if its severity is on the canonical 4-point scale. Using
    a looser rule here would build the matrix from a different population than
    every other table in the paper.

    The judge's reasoning is returned as a *separate* frame rather than a column
    on the matrix table, so it cannot leak into the bootstrap input or change
    the schema of ``matrix_long.csv``.
    """
    rows: list[dict] = []
    reasons: list[dict] = []
    stats = {"samples_seen": 0, "admitted": 0, "no_score": 0, "off_scale": 0,
             "invalid_flagged": 0, "files": []}

    for model in models:
        model_dir = logs_dir / LOG_CONDITION / model
        paths = sorted(model_dir.glob("*.eval")) if model_dir.is_dir() else []
        if not paths:
            continue
        path = select_eval(paths)
        stats["files"].append(_rel(path, logs_dir))
        for sample in prov.iter_eval_samples(path):
            stats["samples_seen"] += 1
            meta = (sample.get("metadata") or {}).get("metadata") or {}
            overseer = (sample.get("scores") or {}).get("overseer")
            if not overseer:
                stats["no_score"] += 1
                continue
            smeta = overseer.get("metadata") or {}
            individual = smeta.get("individual_scores")
            judges = smeta.get("ensemble_models")
            if not individual or not judges or len(individual) != len(judges):
                stats["no_score"] += 1
                continue
            if any(s not in ORDINAL_LEVELS for s in individual):
                stats["off_scale"] += 1
                continue
            value = overseer.get("value")
            if isinstance(value, float) and math.isnan(value):
                # All judges answered but one flagged the response unassessable.
                # The severity survives; the analysis admits it, as upstream does.
                stats["invalid_flagged"] += 1
            key = {
                "scenario_id": meta.get("scenario_id"),
                "source_model": meta.get("source_model") or model,
                "designed_principle": meta.get("designed_principle"),
                "scored_principle": meta.get("scored_principle") or sample.get("target"),
            }
            rows.append({
                **key,
                "domain": meta.get("domain", ""),
                "score": float(individual[0]),
            })
            reasons.append({
                **key,
                "score": float(individual[0]),
                "reasoning": " ".join(_resolve_attachment(
                    overseer.get("explanation"), sample.get("attachments") or {}).split()),
            })
            stats["admitted"] += 1

    df = pd.DataFrame(rows)
    if not df.empty and df[["scenario_id", "designed_principle", "scored_principle"]].isna().any().any():
        raise SystemExit("run logs are missing the metadata the analysis keys on")
    return df, pd.DataFrame(reasons), stats


def load_main_run(raw_csv: Path, scenarios: set[str], models: list[str]) -> pd.DataFrame:
    """Main-run baseline severities for the same scenarios: per-judge and ensemble."""
    raw = pd.read_csv(raw_csv)
    sub = raw[(raw["persona"] == "baseline")
              & (raw["model"].isin(models))
              & (raw["sample_id"].isin(scenarios))]
    if sub.empty:
        return pd.DataFrame(columns=["scenario_id", "source_model", "principle",
                                     "main_single_judge", "main_ensemble", "n_judges"])
    single = (sub[sub["judge_name"] == JUDGE_SHORT]
              .groupby(["sample_id", "model", "principle"], as_index=False)
              .agg(main_single_judge=("severity", "mean")))
    ensemble = (sub.groupby(["sample_id", "model", "principle"], as_index=False)
                .agg(main_ensemble=("severity", "mean"), n_judges=("severity", "size")))
    merged = ensemble.merge(single, on=["sample_id", "model", "principle"], how="left")
    return merged.rename(columns={"sample_id": "scenario_id", "model": "source_model"})


# --- statistics --------------------------------------------------------------
def _cluster_bootstrap_mean(
    values: np.ndarray, clusters: np.ndarray, strata: np.ndarray,
    n_bootstrap: int, seed: int,
) -> tuple[float, float, float]:
    """Mean with a cluster bootstrap over ``clusters``, stratified by ``strata``.

    The scenario is the cluster: it contributes one observation per source model
    and those share whatever makes it hard. Stratifying by designed principle
    keeps each replicate's per-principle n fixed, matching every other CI in the
    paper.
    """
    rng = np.random.default_rng(seed)
    by_cluster: dict[str, list[float]] = {}
    cluster_stratum: dict[str, str] = {}
    for v, c, s in zip(values, clusters, strata):
        by_cluster.setdefault(c, []).append(float(v))
        cluster_stratum[c] = s

    # Sort the pools. `rng.integers` indexes them positionally, so unsorted pools
    # make the CI depend on the physical order Inspect happened to write samples
    # into the zip -- two logs with byte-identical content would give different
    # intervals under the same seed. `bootstrap._build_cells` sorts by sample_id
    # for exactly this reason, and every other CI in the paper inherits it.
    pools: dict[str, list[str]] = {}
    for c in sorted(cluster_stratum):
        pools.setdefault(cluster_stratum[c], []).append(c)
    for s in pools:
        pools[s].sort()

    point = float(np.mean(values))
    reps = np.empty(n_bootstrap)
    keys = sorted(pools)
    for r in range(n_bootstrap):
        drawn: list[float] = []
        for s in keys:
            pool = pools[s]
            picks = rng.integers(0, len(pool), size=len(pool))
            for i in picks:
                drawn.extend(by_cluster[pool[i]])
        reps[r] = np.mean(drawn)
    return point, float(np.percentile(reps, CI_LOW_PCT)), float(np.percentile(reps, CI_HIGH_PCT))


def sanity_check(joined: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Agreement between the new diagonal and the main run, two comparators.

    Kept in its own table and deliberately NOT pooled into the published
    Krippendorff alpha, design effects or the 35,416-item count: those are
    conditioned on the three reported personas and the three-judge ensemble, and
    concatenating a fourth condition into them would silently redefine them.
    """
    rows = []
    for label, col in (
        (f"vs main-run {JUDGE_SHORT} (same judge, Nov 2025 vs Jul 2026)", "main_single_judge"),
        ("vs main-run 3-judge ensemble mean", "main_ensemble"),
    ):
        sub = joined.dropna(subset=[col, "score"])
        if sub.empty:
            continue
        diff = (sub["score"] - sub[col]).to_numpy()
        point, lo, hi = _cluster_bootstrap_mean(
            diff, sub["scenario_id"].to_numpy(), sub["designed_principle"].to_numpy(),
            n_bootstrap, seed)
        exact = float((sub["score"] == sub[col]).mean())
        # `>= 0` is the acceptable side, matching `bootstrap.binarize_long` and
        # compute_binarized_robustness_gap.py. An ensemble mean of exactly 0.0
        # happens whenever the three judges split (e.g. -1, +0.5, +0.5) and
        # occurs on ~1.4% of these items; `np.sign` equality would score every
        # one of them as a disagreement no matter what this run's judge said.
        same_sign = float(((sub["score"] >= 0) == (sub[col] >= 0)).mean())
        rho, rho_p = sp.spearmanr(sub["score"], sub[col])
        rows.append({
            "comparator": label,
            "n": int(len(sub)),
            "mean_signed_difference": point,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                     and (hi < 0 or lo > 0)),
            "mean_absolute_difference": float(np.abs(diff).mean()),
            "exact_agreement": exact,
            "same_sign_agreement": same_sign,
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
        })
    return pd.DataFrame(rows)


def item_level_correlations(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Pearson and Spearman between the 8 scored principles, across items.

    The unit is one (scenario, source model) response -- an *item*, not a model
    average. That is a genuinely different matrix from the n=15 model-level one
    already in the paper, and it is still not evidence about factor structure:
    see the caveat written into the report.
    """
    wide = long.pivot_table(index=["scenario_id", "source_model"],
                            columns="scored_principle", values="score")
    wide = wide[[p for p in PRINCIPLES if p in wide.columns]].dropna()
    return wide.corr(method="pearson"), wide.corr(method="spearman"), int(len(wide))


# --- post-hoc ----------------------------------------------------------------
# Everything below this line was written AFTER the run returned a diagonal
# *above* its row rather than below it. It is labelled post-hoc in the report
# for that reason. It adds no new judge calls, changes no pre-committed
# statistic, and is reported alongside section 1, never in place of it. Its
# purpose is to say why the sign came out as it did, since "reversed, cause
# unknown" invites the reader to conclude the instrument is broken when the
# cause is a documented property of the instrument.

_NO_NEUTRAL_RULE_PREFIX = "2. Minimal but technically harmless"


def rubric_no_neutral_rule() -> str | None:
    """The global rule that forces an unengaged principle to a negative score.

    Read out of the live template rather than quoted from memory, so the report
    cannot keep asserting a rule the scorer no longer contains.
    """
    for line in load_overseer_template().splitlines():
        text = line.strip()
        if text.startswith(_NO_NEUTRAL_RULE_PREFIX):
            return text
    return None


def severity_distribution(long: pd.DataFrame) -> pd.DataFrame:
    """Share of each severity level on the diagonal vs off it."""
    diag = long["designed_principle"] == long["scored_principle"]
    rows = []
    for label, sub in (("diagonal", long[diag]), ("off-diagonal", long[~diag])):
        row = {"cells": label, "n": int(len(sub))}
        for level in ORDINAL_LEVELS:
            row[f"share_{level:+.1f}"] = (float((sub["score"] == level).mean())
                                          if len(sub) else float("nan"))
            row[f"n_{level:+.1f}"] = int((sub["score"] == level).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def offdiagonal_column_means(long: pd.DataFrame) -> pd.DataFrame:
    """Per-rubric leniency, measured only where the principle is off-target.

    A column's own diagonal is excluded so the leniency estimate is not
    contaminated by the cell the contrast is about.
    """
    diag = long["designed_principle"] == long["scored_principle"]
    off = long[~diag].groupby("scored_principle")["score"].mean()
    on = long[diag].groupby("scored_principle")["score"].mean()
    out = pd.DataFrame({"off_diagonal_mean": off, "diagonal": on})
    out.index.name = "scored_principle"
    return out.reindex([p for p in PRINCIPLES if p in out.index]).reset_index()


def offdiagonal_examples(reasons: pd.DataFrame) -> pd.DataFrame:
    """One verbatim judge rationale per scored principle, chosen deterministically.

    The rule is fixed in advance of reading any of the text, and rotates the
    *source* of the example so the eight do not all land on whichever scenario
    happens to sort first: for the j-th principle in canonical order, prefer a
    scenario designed for the (j+1)-th principle and the (j mod n_models)-th
    model, then take the first (scenario, model) in sort order. Where that cell
    has no -0.5 call, fall back to the first in sort order for the column.
    Selecting by content instead would be cherry-picking; this is auditable
    against the logs either way.
    """
    if reasons.empty:
        return reasons
    off = reasons[(reasons["designed_principle"] != reasons["scored_principle"])
                  & (reasons["score"] == -0.5)]
    if off.empty:
        return off
    off = off.sort_values(["scored_principle", "designed_principle",
                           "scenario_id", "source_model"])
    models = sorted(off["source_model"].unique())

    picks = []
    for j, scored in enumerate(PRINCIPLES):
        column = off[off["scored_principle"] == scored]
        if column.empty:
            continue
        want_designed = PRINCIPLES[(j + 1) % len(PRINCIPLES)]
        want_model = models[j % len(models)]
        for candidate in (
            column[(column["designed_principle"] == want_designed)
                   & (column["source_model"] == want_model)],
            column[column["designed_principle"] == want_designed],
            column,
        ):
            if not candidate.empty:
                picks.append(candidate.iloc[0])
                break
    return pd.DataFrame(picks).reset_index(drop=True)


# --- report ------------------------------------------------------------------
def _fmt(x: float, places: int = 2) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{places}f}"


def matrix_table(matrix: DesignedMeasuredMatrix) -> list[str]:
    cols = [PRINCIPLE_SHORT[p] for p in matrix.principles]
    lines = ["| designed \\ scored | " + " | ".join(cols) + " |",
             "| --- |" + " ---: |" * len(cols)]
    for i, p in enumerate(matrix.principles):
        cells = []
        for j in range(len(matrix.principles)):
            v = _fmt(matrix.point[i, j])
            cells.append(f"**{v}**" if i == j else v)
        lines.append(f"| **{PRINCIPLE_SHORT[p]}** | " + " | ".join(cells) + " |")
    return lines


def frame_composition() -> dict:
    """VP composition of the 96 and of the 200 it was drawn from.

    The parent draw is VP-stratified; this one is stratified on domain only, as
    specified. Recording both shows whether the parent's VP mix survived the
    nesting, instead of leaving it to be assumed either way.
    """
    def vp_bucket(row: dict) -> str:
        tag = (row.get("metadata") or {}).get("vulnerable-population") or ""
        return tag if tag in ("children", "teenagers", "elderly") else (
            "other-vp" if tag else "none")

    def shares(ids: set[str], rows: dict) -> dict:
        counts: dict[str, int] = {}
        for i in ids:
            if i in rows:
                b = vp_bucket(rows[i])
                counts[b] = counts.get(b, 0) + 1
        n = sum(counts.values()) or 1
        order = ("none", "children", "teenagers", "elderly", "other-vp")
        return {k: counts.get(k, 0) / n for k in order if counts.get(k)}

    rows: dict[str, dict] = {}
    with prov.DATASET_PATH.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    try:
        frame_ids = {ln.strip() for ln in IDS_PATH.read_text().splitlines() if ln.strip()}
        parent_ids = {ln.strip() for ln in PARENT_IDS_PATH.read_text().splitlines() if ln.strip()}
    except OSError:
        return {}
    return {
        "vulnerable_population": shares(frame_ids, rows),
        "parent_vulnerable_population": shares(parent_ids, rows),
    }


def interpretation_section(
    raw: pd.DataFrame, ranks: pd.DataFrame, fhr_pltw: pd.DataFrame
) -> list[str]:
    """Apply the interpretation committed to *before* the run, mechanically.

    `docs/discriminant_validity_design.md` section 5 wrote down three outcomes and
    what each licenses, before any data existed. Selecting between them in code
    rather than in prose is the point: it removes the freedom to describe a weak
    result as a strong one, which is exactly the freedom the paper has been
    suspected of using.
    """
    rows = raw[raw.designed_principle != "pooled"]
    pooled = raw[raw.designed_principle == "pooled"].iloc[0]
    est = rows[rows.estimable]
    discriminating = est[est.ci_upper < 0]["designed_principle"].tolist()
    # A CI entirely ABOVE zero is the reversed result -- the scenario scores
    # *better* on the principle it was designed for. Lumping it into "flat" and
    # recommending a merge would report a reversal as construct collapse, which
    # is a different finding pointing the opposite way.
    reversed_rows = est[est.ci_lower > 0]["designed_principle"].tolist()
    unestimable = rows[~rows.estimable]["designed_principle"].tolist()
    flat = [p for p in est["designed_principle"]
            if p not in discriminating and p not in reversed_rows]
    n_d, n_total = len(discriminating), len(rows)

    L = ["## Interpretation, pre-committed\n"]
    L.append(
        "The three outcomes below and what each licenses were written down in "
        "`docs/discriminant_validity_design.md` section 5 **before the run**. The one "
        "that applies is selected in code from the numbers above, not chosen "
        "afterwards.\n"
    )
    pooled_txt = (f"{_fmt(pooled.contrast)} [{_fmt(pooled.ci_lower)}, "
                  f"{_fmt(pooled.ci_upper)}]" if pooled.estimable
                  else "**not estimable** (no row had usable data)")
    L.append(f"**{n_d} of {n_total} rows** have a diagonal-minus-off-diagonal CI "
             f"entirely below zero. Pooled: {pooled_txt}"
             + (f", over {int(pooled.n_rows_used)} of {n_total} rows"
                if pooled.estimable and pooled.n_rows_used < n_total else "")
             + ".\n")

    if unestimable:
        L.append(
            f"**{len(unestimable)} row(s) could not be estimated at all** and are "
            "excluded from every count in this section, including the "
            f"denominators: {', '.join(f'`{PRINCIPLE_SHORT.get(p, p)}`' for p in unestimable)}. "
            "An unestimated row is missing data, not a null result, and must not "
            "be read as either evidence for or against discrimination.\n"
        )
    if reversed_rows:
        L.append(
            f"**{len(reversed_rows)} row(s) came out REVERSED** -- CI entirely "
            "*above* zero, i.e. the scenarios score better on the principle they "
            "were designed for than on the seven they were not: "
            + ", ".join(f"`{PRINCIPLE_SHORT.get(p, p)}`" for p in reversed_rows)
            + ". This is neither discrimination as defined nor flatness, and it "
            "is not a merge candidate. The pre-committed interpretation did not "
            "anticipate this outcome, so it is reported as-is and left for the "
            "authors rather than resolved by this script.\n"
        )

    # Branch on the rows that could actually be estimated. An unestimable row is
    # absent evidence and must not tip the outcome either way.
    n_est = len(est)
    if n_est and n_d == n_est:
        L.append(
            "**Outcome: the diagonal is distinct throughout.** The principles "
            "measure what they were designed to measure. This is "
            "construct-validity evidence. Inter-principle correlations remain "
            "high, but that is expected both for related normative constructs "
            "and as documented LLM-judge behaviour, and the diagonal contrast is "
            "robust to both.\n"
        )
    elif n_est and n_d == 0 and not reversed_rows:
        L.append(
            "**Outcome: the diagonal is flat everywhere.** The scenarios do not "
            "discriminate between principles. The honest reading is that the "
            "eight are facets of a single humaneness construct rather than eight "
            "distinct constructs, and section 3.1 should be revised accordingly. "
            "This does not touch the headline finding, which is about robustness "
            "under system-prompt pressure and does not require the principles to "
            "be separable. It is reported rather than suppressed, per the "
            "pre-commitment and because the provenance archive makes selective "
            "reporting detectable.\n"
        )
    elif not n_est:
        L.append(
            "**Outcome: not determinable.** No row could be estimated, so none "
            "of the three pre-committed outcomes applies. This is a failed run, "
            "not a null result.\n"
        )
    else:
        L.append(
            f"**Outcome: mixed -- distinct for {n_d}, flat for {len(flat)}"
            + (f", reversed for {len(reversed_rows)}" if reversed_rows else "")
            + f", of {n_est} estimable row(s).** "
            "Report per-principle. The flat rows are candidates for merging in "
            "v2 or for a stated limitation; conceding a merge candidate while "
            "defending the rest is more credible than defending all eight.\n"
        )
        L.append("Flat rows: " + (", ".join(
            f"`{PRINCIPLE_SHORT.get(p, p)}`" for p in flat) if flat else "none") + ".\n")
        named = {FHR, PLTW}
        if named & set(flat):
            both = ("both members of that pair are" if named <= set(flat)
                    else "one member of that pair is")
            L.append(
                f"**The pair named in review -- Foster Healthy Relationships and Prioritize "
                f"Long-term Wellbeing, and {both} among the flat "
                "rows.** That is the objection confirmed on its own example, and "
                "it should be said plainly and cited "
                "as a v2 revision target rather than argued around.\n"
            )
        elif named & set(discriminating) == named:
            L.append(
                "**The pair named in review -- Foster Healthy Relationships and "
                "Prioritize Long-term Wellbeing -- is not among the flat rows.** "
                "Both discriminate, so the specific objection is answered "
                "directly, while other rows are conceded.\n"
            )

    if flat:
        L.append(
            "The asymmetry is worth stating in the paper: a flat diagonal under "
            "LLM-judge scoring is consistent with *either* genuine construct "
            "overlap *or* judge collapse, so a null here bounds what the "
            "instrument can resolve rather than refuting the framework. A "
            "*non*-flat diagonal has no such ambiguity, which is why the "
            "positive rows carry more weight than the flat ones.\n"
        )

    # Ranks are only counted over rows that have a rank -- an unestimable row
    # carries None, and counting it as "not lowest" would be as wrong as
    # counting it as lowest.
    ranked = ranks[ranks.estimable]
    n_ranked = len(ranked)
    if n_ranked:
        n_lowest = int((ranked.rank_in_row == 1).sum())
        n_bottom2 = int((ranked.rank_in_row <= 2).sum())
        n_col = int((ranked.rank_in_column == 1).sum())
        L.append(
            f"Ordinally: the diagonal is the lowest cell in its row for "
            f"**{n_lowest} of {n_ranked}** principles and in the bottom two for "
            f"**{n_bottom2} of {n_ranked}**; it is lowest in its *column* for "
            f"**{n_col} of {n_ranked}**, which is the version that cannot be "
            "explained by a harsh rubric."
            + (f" {n_total - n_ranked} row(s) had no rankable diagonal.\n"
               if n_ranked < n_total else "\n")
        )
    return L


def posthoc_section(
    dist: pd.DataFrame, colmeans: pd.DataFrame, examples: pd.DataFrame,
    ranks: pd.DataFrame, centered: pd.DataFrame, rule: str | None,
) -> list[str]:
    """Why the sign came out reversed. Post-hoc, and labelled as such."""
    d = dist.set_index("cells")
    on, off = d.loc["diagonal"], d.loc["off-diagonal"]
    n_all = int(on["n"] + off["n"])
    n_worst = int(on["n_-1.0"] + off["n_-1.0"])

    L = ["## Post-hoc: why the diagonal sits above its row\n"]
    L.append(
        "**Written after seeing the sign.** Section 1 and the pre-committed "
        "interpretation above are untouched; nothing here was used to select "
        "between the committed outcomes. It is here because a reversed result "
        "reported without a mechanism invites the reader to conclude the "
        "instrument is broken, when the cause is a documented property of the "
        "instrument.\n"
    )

    L.append("### The scale has no neutral level\n")
    if rule:
        L.append(f"> {rule}\n")
    else:
        L.append(
            "> **The global rule this section is about was not found in the "
            "current scorer template.** The paragraph below describes the "
            "instrument as it was at the time of the run and must be re-checked "
            "against `humanebench/scorer.py` before being quoted.\n"
        )
    L.append(
        "The severity levels are -1.0, -0.5, +0.5, +1.0. There is no zero and no "
        "*not applicable*: a response that neither violates a principle nor "
        "engages it has nowhere to go but -0.5. Under single-label scoring, "
        "where every scenario engages the principle it was written for, that is "
        "an anti-hedging rule and it does what it was meant to do. Under "
        "multi-label scoring it becomes the dominant term, because seven of the "
        "eight rubrics applied to any given response ask about something the "
        "scenario never raised.\n"
    )

    L.append("### What the distribution shows\n")
    L.append("| cells | n | -1.0 | -0.5 | +0.5 | +1.0 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label in ("diagonal", "off-diagonal"):
        r = d.loc[label]
        L.append(f"| {label} | {int(r['n']):,} | " + " | ".join(
            f"{r[f'share_{lv:+.1f}']:.1%}" for lv in ORDINAL_LEVELS) + " |")
    L.append("")
    L.append(
        f"Off-diagonal calls land on -0.5 **{off['share_-0.5']:.1%}** of the "
        f"time against **{on['share_-0.5']:.1%}** on the diagonal, while the "
        f"diagonal takes +1.0 **{on['share_+1.0']:.1%}** of the time against "
        f"**{off['share_+1.0']:.1%}** off it. Outright violations are "
        f"**{n_worst} of {n_all:,}** calls ({n_worst / n_all:.1%}). The section 1 "
        "gap is therefore mostly *earned credit versus unaddressed*, not "
        "*complied versus violated*.\n"
    )

    L.append("### What this licenses, and what it does not\n")
    L.append(
        "**It does not license reading the off-diagonal as violation.** A "
        "scenario is not evidence that a model breaches the seven principles it "
        "was not written to probe; off-diagonal -0.5 overwhelmingly means the "
        "response never engaged that principle. The main benchmark scores each "
        "scenario against one principle, so no published number is affected by "
        "this -- but a reader meeting the matrix cold could easily conclude "
        "otherwise, and should not.\n"
    )
    L.append(
        "**It does not rescue the pre-committed direction.** That prediction "
        "assumed failure concentrates on the designed principle. At baseline it "
        "does not, because at baseline these models largely satisfy the "
        "principle their scenario stresses -- which is what the benchmark's own "
        "baseline scores say.\n"
    )
    L.append(
        "**It does support the claim the reviewer actually asked about.** The "
        "eight rubrics are not interchangeable. The same response, scored eight "
        f"times, takes +1.0 under the rubric its scenario was designed for "
        f"{on['share_+1.0']:.1%} of the time while drawing -0.5 under the other "
        f"seven {off['share_-0.5']:.1%} of the time. A judge that had collapsed "
        "the rubrics into one latent dimension -- the Feuer et al. failure mode, "
        "and the reason no factor model is reported here -- would not produce "
        "that, because collapse makes the eight move together. Which principle a "
        "response satisfies is predicted by the label its scenario was written "
        "under. That is the known-groups claim, with engagement rather than "
        "failure as the thing that concentrates on the diagonal.\n"
    )
    L.append(
        "**The honest limit of that claim.** What the diagonal establishes is "
        "that the rubrics are *differentially responsive to scenario content*: "
        "the eight do not return the same verdict on the same response. That "
        "rules out one construct measured eight times, and it rules out a fully "
        "collapsed judge. It does not by itself establish that any particular "
        "pair of principles is non-synonymous -- two near-synonymous rubrics "
        "keyed to the same content would both light up on the same scenarios and "
        "both stay quiet elsewhere. A named pair is answered by the paired test "
        "in section 3, not by this contrast, and the correlations in section 5 "
        "remain descriptive-only for the reason given there.\n"
    )

    est = ranks[ranks.estimable]
    col_lowest = [PRINCIPLE_SHORT.get(r.designed_principle, r.designed_principle)
                  for _, r in est.iterrows() if r.rank_in_column == 1]
    neg_diag = [PRINCIPLE_SHORT.get(r.designed_principle, r.designed_principle)
                for _, r in est.iterrows() if r.diagonal < 0]
    if col_lowest:
        L.append(
            "The pre-committed pattern does appear in "
            f"`{'`, `'.join(col_lowest)}`: the diagonal is the **lowest cell in "
            "its own column**, i.e. of all scenarios scored against that "
            "principle, the twelve written for it score lowest. "
            + (f"That is also where the only negative diagonal sits "
               f"(`{'`, `'.join(neg_diag)}`). " if neg_diag else "")
            + "Where a scenario set does concentrate failure on its own "
              "principle, the design detects it.\n"
        )

    L.append("### The same ranks, mirrored\n")
    L.append(
        "Section 2 asks whether the diagonal is the *lowest* cell, which was the "
        "committed direction. The mirror is reported here rather than there so "
        "that neither direction can be chosen after the fact.\n"
    )
    L.append("| designed principle | diagonal | rank from top in row | "
             "rank from top in column | replicates highest in row | top two |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    n_high = n_top2 = n_est = 0
    for _, r in ranks.iterrows():
        if r.estimable:
            n_est += 1
            n_high += int(r.rank_in_row_from_top == 1)
            n_top2 += int(r.rank_in_row_from_top <= 2)
            cells = (f"{int(r.rank_in_row_from_top)}/{int(r.n_cells_ranked_in_row)} | "
                     f"{int(r.rank_in_column_from_top)}/{int(r.n_cells_ranked_in_column)} | "
                     f"{r.share_highest_in_row:.0%} | {r.share_top_two_in_row:.0%}")
        else:
            cells = "no data | no data | no data | no data"
        L.append(f"| {PRINCIPLE_SHORT.get(r.designed_principle, r.designed_principle)} "
                 f"| {_fmt(r.diagonal)} | {cells} |")
    L.append("")
    cen = centered[centered.designed_principle != "pooled"]
    n_cen_pos = int((cen.excludes_zero & (cen.contrast > 0)).sum())
    L.append(
        f"The diagonal is the highest cell in its row for **{n_high} of "
        f"{n_est}** principles and in the top two for **{n_top2} of {n_est}** -- "
        "so the row-level result is a contrast against the row *mean*, not a "
        "claim that the designed principle always wins outright. It is beaten "
        "by the leniently-scored columns below, which is exactly the effect the "
        f"column-centred contrast removes; centring leaves {n_cen_pos} row(s) "
        "positive with a CI excluding zero.\n"
    )

    L.append("### Per-rubric leniency\n")
    L.append(
        "Column means computed **off the diagonal only**, so the leniency "
        "estimate is not contaminated by the cell the contrast is about.\n"
    )
    L.append("| scored principle | off-diagonal mean | diagonal |")
    L.append("| --- | ---: | ---: |")
    for _, r in colmeans.sort_values("off_diagonal_mean").iterrows():
        L.append(f"| {PRINCIPLE_SHORT.get(r.scored_principle, r.scored_principle)} | "
                 f"{_fmt(r.off_diagonal_mean)} | {_fmt(r.diagonal)} |")
    L.append("")

    if not examples.empty:
        L.append("### The judge's own account\n")
        L.append(
            "One rationale per scored principle, selected by a rule fixed before "
            "reading any of them: among off-diagonal calls scoring -0.5 for that "
            "principle, rotate the source -- the j-th principle draws from a "
            "scenario designed for the (j+1)-th and from the (j mod 3)-th model "
            "-- then take the first in sort order, falling back within the "
            "column if that cell is empty. The rotation is there so the eight "
            "examples do not all land on whichever scenario sorts first; it is "
            "not a content filter. Full text is in "
            "`tables/discriminant/offdiagonal_examples.csv`, and every rationale "
            "in the run is in the logs.\n"
        )
        for _, r in examples.iterrows():
            short = PRINCIPLE_SHORT.get(r.scored_principle, r.scored_principle)
            L.append(f"- **scored as {short}**, scenario designed for "
                     f"{PRINCIPLE_SHORT.get(r.designed_principle, r.designed_principle)} "
                     f"(`{r.scenario_id}`, {r.source_model}): "
                     f"\"{r.reasoning[:300]}{'...' if len(r.reasoning) > 300 else ''}\"")
        L.append("")
    return L


def write_report(
    out: Path, matrix: DesignedMeasuredMatrix, per_model: dict,
    raw: pd.DataFrame, centered: pd.DataFrame, ranks: pd.DataFrame,
    fhr_pltw: pd.DataFrame, sanity: pd.DataFrame, pearson: pd.DataFrame,
    spearman: pd.DataFrame, n_items: int, stats: dict, manifest: dict,
    n_bootstrap: int, seed: int, frame_composition: dict,
    expected_calls: int, dist: pd.DataFrame, colmeans: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    pooled_raw = raw[raw.designed_principle == "pooled"].iloc[0]
    n_cell = int(np.median(matrix.n_per_cell))

    L = ["# Principle discriminant validity: the designed x measured matrix\n"]
    shortfall = expected_calls - stats["admitted"]
    if shortfall > 0:
        L.append(
            f"> **This matrix is incomplete.** {stats['admitted']:,} of "
            f"{expected_calls:,} planned judge calls survived "
            f"admission ({stats['admitted'] / max(expected_calls, 1):.1%}); "
            f"{shortfall} did not. Cells are ragged and the numbers below are "
            "computed on what exists. Do not quote them as a completed run.\n"
        )
    L.append(
        f"Rows are the principle a scenario was **designed** for; columns the "
        f"principle it was **scored** against. {matrix.n_scenarios[0]} scenarios "
        f"per row x {len(matrix.models)} source models = **n = {n_cell} per "
        f"cell**, {stats['admitted']:,} judged calls in total. Baseline responses "
        "only; one judge; one separate judge call per principle.\n"
    )
    L.append(
        "Cells are mean judge severity on the 4-point rubric, **not HumaneScore** "
        "-- HumaneScore is the mean of the eight principle means, and a cell here "
        "is one such principle mean computed on a 12-scenario slice.\n"
    )

    L.append("## Why this design and not factor analysis\n")
    L.append(
        "Feuer et al. (arXiv:2509.20293) factored Arena-Hard-Auto's rubric across "
        "four judges and found factor correlations above 0.93 with roughly 55% of "
        "judgement variance unexplained by the stated rubric: LLM judges collapse "
        "semantically distinct criteria into a single latent dimension. A "
        "high-correlation factor result here would therefore be equally "
        "consistent with the principles genuinely overlapping and with the judge "
        "having collapsed them, and could not distinguish the two.\n"
    )
    L.append(
        "The statistic below is a contrast **within a row**. Anything that shifts "
        "a whole row together -- model quality, judge leniency, factor collapse, "
        "a ceiling -- cancels. **No EFA was run, no factor count chosen, no "
        "rotation applied.**\n"
    )

    L.append("## 1. Headline: mean diagonal minus mean off-diagonal\n")
    L.append(
        "**Negative means the designed principle scores lower than the seven it "
        "was not designed for** -- i.e. the scenario surfaces the failure it was "
        f"built to surface. CIs are the scenario-level cluster bootstrap "
        f"({n_bootstrap:,} replicates, seed {seed}, 2.5/97.5 "
        "percentiles), with one scenario draw per row carried across all eight "
        "columns and all source models so the within-row pairing is preserved.\n"
    )
    L.append("| designed principle | contrast | 95% CI | excludes 0 | column-centred | 95% CI |")
    L.append("| --- | ---: | :---: | :---: | ---: | :---: |")
    cen = centered.set_index("designed_principle")
    for _, r in raw.iterrows():
        c = cen.loc[r.designed_principle]
        name = "**pooled**" if r.designed_principle == "pooled" else PRINCIPLE_SHORT.get(
            r.designed_principle, r.designed_principle)
        L.append(
            f"| {name} | {_fmt(r.contrast)} | [{_fmt(r.ci_lower)}, {_fmt(r.ci_upper)}] | "
            f"{'yes' if r.excludes_zero else 'no'} | {_fmt(c.contrast)} | "
            f"[{_fmt(c.ci_lower)}, {_fmt(c.ci_upper)}] |"
        )
    L.append("")
    L.append(
        "**Column-centred** removes each column's mean first, which cancels a "
        "per-rubric leniency effect: without it, a principle whose rubric is "
        "simply harsh depresses its whole column, and that column's own diagonal "
        "then looks discriminating for a reason that has nothing to do with the "
        "scenario design. It is the two-way additive residual contrast "
        "(subtracting the row mean as well would cancel out of a within-row "
        "difference). Reported alongside the raw contrast, not instead of it.\n"
    )
    L.append(
        "The **pooled** figure needs no such correction: an additive column "
        "offset `d` raises its own row's contrast by `d` and lowers each of the "
        "other seven by `d/7`, which sums to exactly zero. The raw and centred "
        "pooled values are therefore identical by construction, and this is "
        "asserted in the script rather than assumed.\n"
    )

    L.append("## 2. Rank of the diagonal cell\n")
    L.append(
        "Rank 1 is the lowest (most negative) cell. Ordinal, so it survives any "
        "monotone distortion of a 4-point scale -- which is not an interval "
        "scale. The **column** rank is the companion to the centred contrast: a "
        "diagonal that is lowest in its row *and* in its column is not explained "
        "by that rubric being harsh.\n"
    )
    L.append("| designed principle | diagonal | rank in row | rank in column | "
             "replicates lowest in row | bottom two |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in ranks.iterrows():
        if r.estimable:
            row_r = f"{int(r.rank_in_row)}/{int(r.n_cells_ranked_in_row)}"
            col_r = f"{int(r.rank_in_column)}/{int(r.n_cells_ranked_in_column)}"
            lowest = f"{r.share_lowest_in_row:.0%}"
            bottom2 = f"{r.share_bottom_two_in_row:.0%}"
        else:
            # An unranked row prints as absent data, never as an extreme rank.
            row_r = col_r = lowest = bottom2 = "no data"
        L.append(f"| {PRINCIPLE_SHORT.get(r.designed_principle, r.designed_principle)} | "
                 f"{_fmt(r.diagonal)} | {row_r} | {col_r} | {lowest} | {bottom2} |")
    L.append("")

    L.append("## 3. Foster Healthy Relationships vs Prioritize Long-term Wellbeing\n")
    L.append(
        "The literal example raised in review: is FHR a subset of PLTW? The "
        "direct test is "
        "whether FHR-designed scenarios score differently on FHR than on PLTW, "
        "and the reverse. Paired within the bootstrap replicate.\n"
    )
    L.append("| comparison | difference | 95% CI | excludes 0 |")
    L.append("| --- | ---: | :---: | :---: |")
    for _, r in fhr_pltw.iterrows():
        verdict = ("yes" if r.excludes_zero else "no") if r.estimable else "not estimable"
        L.append(f"| {r.comparison} | {_fmt(r.difference)} | "
                 f"[{_fmt(r.ci_lower)}, {_fmt(r.ci_upper)}] | {verdict} |")
    L.append("")

    L.append("## 4. Sanity check against the main run\n")
    L.append(
        "The diagonal *is* the main-run procedure: same rubric scaffold, same "
        "scenario prompt, same response. `verify_discriminant_provenance.py` "
        "checks that byte-for-byte on all 288 diagonal prompts. So the diagonal "
        "and the main run should agree, and the two comparators separate the two "
        "reasons they might not.\n"
    )
    if sanity.empty:
        L.append("*No overlapping main-run scores found; the check could not run.*\n")
    else:
        L.append("| comparator | n | mean signed diff | 95% CI | mean abs diff | "
                 "exact | same sign | Spearman rho |")
        L.append("| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |")
        for _, r in sanity.iterrows():
            L.append(f"| {r.comparator} | {r.n} | {_fmt(r.mean_signed_difference)} | "
                     f"[{_fmt(r.ci_lower)}, {_fmt(r.ci_upper)}] | "
                     f"{r.mean_absolute_difference:.3f} | {r.exact_agreement:.1%} | "
                     f"{r.same_sign_agreement:.1%} | {r.spearman_rho:+.2f} |")
        L.append("")
        L.append(
            "The first row is a **judge-drift test**: identical prompt, identical "
            "judge slug, eight months apart. Any difference there is the judge "
            "moving, not the procedure changing. The second row asks a different "
            "question -- whether one judge reproduces the three-judge ensemble -- "
            "and a gap there is judge composition, not drift. Reporting only the "
            "ensemble comparison would conflate the two.\n"
        )
        L.append(
            "These agreement figures live in their own table and are **not** "
            "pooled into the published Krippendorff alpha, the design effects, "
            "the 35,416-item count, or the judge self-preference contrast. Those "
            "are conditioned on the three reported personas and the three-judge "
            "ensemble; folding a fourth condition into them would redefine them.\n"
        )

    L.append("## 5. Inter-principle correlation, descriptive only\n")
    L.append(
        f"Pearson correlations between the eight scored principles across the "
        f"{n_items:,} (scenario x source model) items. The unit is the item, not "
        "the model average, which makes this a different matrix from the n=15 "
        "one already in the paper.\n"
    )
    cols = [PRINCIPLE_SHORT[c] for c in pearson.columns]
    L.append("| | " + " | ".join(cols) + " |")
    L.append("| --- |" + " ---: |" * len(cols))
    for pr in pearson.index:
        cells = ["--" if pr == pc else f"{pearson.loc[pr, pc]:+.2f}" for pc in pearson.columns]
        L.append(f"| **{PRINCIPLE_SHORT[pr]}** | " + " | ".join(cells) + " |")
    L.append("")
    off = pearson.to_numpy()[~np.eye(len(pearson), dtype=bool)]
    off_s = spearman.to_numpy()[~np.eye(len(spearman), dtype=bool)]
    L.append(f"Median off-diagonal Pearson **{np.median(off):+.2f}**, "
             f"Spearman **{np.median(off_s):+.2f}** (the rubric is a 4-point "
             "ordinal scale, so Pearson is attenuated; both are given).\n")
    L.append(
        "**This table cannot settle construct distinctness and is not offered as "
        "if it could.** High inter-criterion correlation is the documented "
        "behaviour of LLM judges (Feuer et al.), so a large value here is equally "
        "consistent with genuine overlap and with judge collapse. That is exactly "
        "why the argument rests on the diagonal contrast in section 1, which is "
        "robust to both, and not on this matrix.\n"
    )

    L.extend(interpretation_section(raw, ranks, fhr_pltw))
    L.extend(posthoc_section(dist, colmeans, examples, ranks, centered,
                             rubric_no_neutral_rule()))

    L.append("## The matrix\n")
    L.append(f"Mean severity, n = {n_cell} per cell "
             f"({matrix.n_scenarios[0]} scenarios x {len(matrix.models)} models). "
             "Diagonal in bold.\n")
    L.extend(matrix_table(matrix))
    L.append("")
    for model, m in per_model.items():
        L.append(f"### {model}\n")
        L.append(f"n = {int(np.median(m.n_per_cell))} per cell.\n")
        L.extend(matrix_table(m))
        L.append("")
    L.append(
        "Three source models spanning the robustness range, so the matrix cannot "
        "be an artifact of one model's response style. Per-model row contrasts "
        "are in `tables/discriminant/row_contrasts.csv`.\n"
    )

    L.append("## Provenance\n")
    L.append(f"- Frame: {manifest['n_scenarios']} scenarios, "
             f"{manifest['n_scenarios'] // len(PRINCIPLES)} per principle, "
             "domain-stratified within principle, drawn from the frozen "
             f"{Path(manifest['parent_ids_file']).name if manifest.get('parent_ids_file') else 'dataset'} "
             f"(draw seed {BOOTSTRAP_SEED}, fixed at draw time and independent "
             f"of --seed). Nesting inside the decomposition "
             "subsample keeps these scenarios a subset of that arm rather than a "
             "fourth incompatible scenario set.\n")
    if frame_composition:
        vp = frame_composition.get("vulnerable_population", {})
        parent_vp = frame_composition.get("parent_vulnerable_population", {})
        if vp:
            L.append(
                "- Vulnerable-population composition, carried through from the "
                "parent's VP-stratified draw rather than re-stratified (a third "
                "stratum at n=12 would fragment the domain level): "
                + ", ".join(f"{k} {v:.0%}" for k, v in vp.items())
                + (f" (parent: " + ", ".join(f"{k} {parent_vp.get(k, 0):.0%}"
                                             for k in vp) + ")" if parent_vp else "")
                + ". The analysis reports no VP breakdown; this is recorded so "
                "the composition is visible rather than incidental.\n"
            )
    L.append(f"- Frame ids sha256: `{manifest['frame_ids_sha256'][:16]}`; "
             f"subset prompt hash `{manifest['frame_subset_prompt_hash'][:16]}`.\n")
    L.append(f"- Judge: `{manifest['judge']['models'][0]}`, temperature "
             f"{manifest['judge']['temperature']}, "
             f"{manifest['judge']['score_attempts']} attempts. "
             "The only main-run ensemble judge not among the scored models.\n")
    L.append(f"- {manifest['n_judge_calls']:,} judge calls, one per "
             "(scenario, model, principle). Never one call scoring eight.\n")
    L.append(f"- Admitted into the analysis: {stats['admitted']:,} of "
             f"{stats['samples_seen']:,} samples "
             f"(no score {stats['no_score']}, off-scale {stats['off_scale']}, "
             f"judge flagged the response unassessable {stats['invalid_flagged']}).\n")
    L.append("- Logs: " + ", ".join(f"`{f}`" for f in stats["files"]) + "\n")
    L.append(
        "- Rubric reuse and prompt byte-equality are established by "
        "`scripts/verify_discriminant_provenance.py`, which must pass before "
        "these numbers are used. It checks all eight scaffold hashes against the "
        "reported runs and all 288 diagonal prompts against their November "
        "counterparts. This script does not re-derive that and does not assert "
        "it on the verifier's behalf.\n"
    )
    out.write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--raw-csv", type=Path,
                    default=REPO_ROOT / "tables" / "inter_judge_raw_regenerated.csv")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant")
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "results" / "discriminant_validity.md")
    ap.add_argument("--models", nargs="+", default=list(SOURCE_MODELS))
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--min-complete", type=float, default=0.98,
                    help="refuse to write a matrix below this share of the "
                         "planned judge calls")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write a matrix anyway; the shortfall is stated in the "
                         "report header, not buried")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The report's default parent is results/, which is gitignored and absent in
    # a fresh worktree -- without this, the whole analysis runs and then dies on
    # FileNotFoundError at the very last write.
    args.report.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_PATH.read_text())
    long, reasons, stats = load_run_scores(args.logs_dir, args.models)
    if long.empty:
        print("No discriminant run found under "
              f"{(args.logs_dir / LOG_CONDITION)}.\n"
              "The matrix cannot be built. Report the run as not completed "
              "rather than analysing a partial matrix.", file=sys.stderr)
        return 2

    # "If the run cannot complete, say so rather than reporting a partial
    # matrix." A shortfall is refused by default rather than warned about,
    # because a warning on stdout does not survive into the document a reader
    # sees, and a matrix with holes looks exactly like a matrix without them.
    # Scale the denominator to the models actually being analysed. The manifest
    # plans all three; `--models claude-sonnet-4.5` alone is a complete run of
    # one model, not a 33%-complete run of three, and refusing it as incomplete
    # (or stamping "this matrix is incomplete" on it) would be false.
    # Count checks can be fooled by a wrong dataset of the right size; identity
    # checks cannot. Every scenario in the logs must belong to the frozen frame.
    frame_ids = {ln.strip() for ln in IDS_PATH.read_text().splitlines() if ln.strip()}
    alien = sorted(set(long["scenario_id"]) - frame_ids)
    if alien:
        print(f"WRONG FRAME: {len(alien)} scenario id(s) in the logs are not in "
              f"the frozen 96 (e.g. {alien[:3]}). Not writing a matrix.",
              file=sys.stderr)
        return 4

    expected_calls = manifest["n_judge_calls"] // manifest["n_source_models"] * len(args.models)
    shortfall = expected_calls - stats["admitted"]
    if shortfall < 0:
        # More admitted calls than the frame plans is not "extra data" -- it is
        # exactly what a run against a stale or oversized dataset looks like,
        # and --allow-incomplete must not override it: a wrong-frame run is
        # wrong, not incomplete.
        print(
            f"WRONG FRAME: {stats['admitted']:,} judge calls admitted but the "
            f"frame plans only {expected_calls:,} for {sorted(args.models)}.\n"
            "Not writing a matrix. The logs contain samples outside the frozen "
            "frame -- check which dataset the run actually scored.",
            file=sys.stderr,
        )
        return 4
    if shortfall > 0:
        frac = stats["admitted"] / expected_calls
        if frac < args.min_complete and not args.allow_incomplete:
            print(
                f"INCOMPLETE: {stats['admitted']:,} of {expected_calls:,} judge "
                f"calls survived admission ({frac:.1%}, threshold "
                f"{args.min_complete:.0%}).\n"
                "Not writing a matrix. Re-run the incomplete cells, or pass "
                "--allow-incomplete to produce a matrix that states its own "
                "shortfall in the report header.",
                file=sys.stderr,
            )
            return 3
        print(f"WARNING: {shortfall} of {expected_calls} judge calls did not "
              "survive admission; cells are ragged and the shortfall is stated "
              "in the report header.")

    long.to_csv(args.output_dir / "matrix_long.csv", index=False)

    matrix = bootstrap_designed_measured_matrix(
        long, n_bootstrap=args.n_bootstrap, seed=args.seed)
    raw = discriminant_contrasts(matrix, centered=False)
    centered = discriminant_contrasts(matrix, centered=True)

    # Pooled raw and pooled centred are equal by construction, but ONLY on a
    # complete matrix: an additive column offset d raises its own row's contrast
    # by d and lowers each of the other seven by d/7, summing to zero. A missing
    # cell breaks the arithmetic that makes those terms cancel -- its row then
    # averages six off-diagonal cells instead of seven, and its column's mean is
    # taken over fewer rows -- so the identity is asserted only where it holds
    # and reported as a diagnostic where it does not.
    complete = bool(np.isfinite(matrix.point).all())
    gap = abs(raw.iloc[-1].contrast - centered.iloc[-1].contrast)
    if complete:
        if not math.isclose(raw.iloc[-1].contrast, centered.iloc[-1].contrast,
                            abs_tol=1e-9):
            raise SystemExit(
                "pooled raw and column-centred contrasts differ on a COMPLETE "
                f"matrix (gap {gap:.2e}); an additive column effect must cancel "
                "in the pool, so one of them is computed wrong"
            )
    elif gap > 0:
        print(f"note: pooled raw and centred differ by {gap:.2e} because the "
              "matrix has empty cells; the cancellation identity only holds on a "
              "complete matrix")

    ranks = diagonal_ranks(matrix)
    per_model: dict[str, DesignedMeasuredMatrix] = {}
    contrast_rows = [raw.assign(source_model="pooled"),
                     centered.assign(source_model="pooled")]
    for model in sorted(long["source_model"].unique()):
        m = bootstrap_designed_measured_matrix(
            long, n_bootstrap=args.n_bootstrap, seed=args.seed, models=[model])
        per_model[model] = m
        contrast_rows.append(discriminant_contrasts(m).assign(source_model=model))
        pd.DataFrame(m.point, index=list(m.principles), columns=list(m.principles)).to_csv(
            args.output_dir / f"matrix_{model}.csv")

    pd.DataFrame(matrix.point, index=list(matrix.principles),
                 columns=list(matrix.principles)).to_csv(
        args.output_dir / "matrix_pooled.csv")
    pd.concat(contrast_rows, ignore_index=True).to_csv(
        args.output_dir / "row_contrasts.csv", index=False)
    ranks.to_csv(args.output_dir / "diagonal_ranks.csv", index=False)

    # 3. The pair named in review, both directions. Same estimable discipline as
    # the row contrasts: cell_difference returns NaN CIs for an empty cell, and
    # NaN comparisons read False -- which would print "excludes 0: no" and be
    # read as a null result where there is no data at all.
    fhr_rows = []
    for designed, other in ((FHR, PLTW), (PLTW, FHR)):
        d, lo, hi = matrix.cell_difference(designed, designed, other)
        estimable = bool(np.isfinite(d) and np.isfinite(lo) and np.isfinite(hi))
        fhr_rows.append({
            "comparison": f"{PRINCIPLE_SHORT[designed]}-designed: "
                          f"{PRINCIPLE_SHORT[designed]} minus {PRINCIPLE_SHORT[other]}",
            "designed_principle": designed,
            "scored_a": designed,
            "scored_b": other,
            "difference": d,
            "ci_lower": lo,
            "ci_upper": hi,
            "estimable": estimable,
            "excludes_zero": bool(estimable and (hi < 0 or lo > 0)),
        })
    fhr_pltw = pd.DataFrame(fhr_rows)
    fhr_pltw.to_csv(args.output_dir / "fhr_pltw.csv", index=False)

    # 4. Sanity check against the main run, on the diagonal only.
    diagonal = long[long["designed_principle"] == long["scored_principle"]]
    main = load_main_run(args.raw_csv, set(long["scenario_id"]), args.models)
    sanity = pd.DataFrame()
    if not main.empty:
        joined = diagonal.merge(
            main, left_on=["scenario_id", "source_model", "designed_principle"],
            right_on=["scenario_id", "source_model", "principle"], how="left")
        joined.to_csv(args.output_dir / "diagonal_vs_main_run.csv", index=False)
        sanity = sanity_check(joined, args.n_bootstrap, args.seed)
        sanity.to_csv(args.output_dir / "sanity_vs_main_run.csv", index=False)

        confusion = pd.crosstab(
            joined["main_single_judge"], joined["score"],
            rownames=[f"main-run {JUDGE_SHORT}"], colnames=["this run"], dropna=False)
        confusion.to_csv(args.output_dir / "judge_drift_confusion.csv")
        n_missing = int(joined["main_ensemble"].isna().sum())
        if n_missing:
            print(f"note: {n_missing} diagonal cell(s) have no main-run comparator "
                  "(a judge failed in the reported run); they are scored here and "
                  "excluded from the agreement statistics only")

    # 5. Item-level correlations, descriptive.
    pearson, spearman, n_items = item_level_correlations(long)
    pearson.to_csv(args.output_dir / "interprinciple_correlation_item_level.csv")
    spearman.to_csv(args.output_dir / "interprinciple_correlation_item_level_spearman.csv")

    # 6. Post-hoc mechanism diagnostics for the reversed sign. No new calls, no
    # effect on anything above; see the block comment above `posthoc_section`.
    dist = severity_distribution(long)
    colmeans = offdiagonal_column_means(long)
    examples = offdiagonal_examples(reasons)
    dist.to_csv(args.output_dir / "severity_distribution.csv", index=False)
    colmeans.to_csv(args.output_dir / "offdiagonal_column_means.csv", index=False)
    if not examples.empty:
        examples.to_csv(args.output_dir / "offdiagonal_examples.csv", index=False)

    write_report(args.report, matrix, per_model, raw, centered, ranks, fhr_pltw,
                 sanity, pearson, spearman, n_items, stats, manifest,
                 args.n_bootstrap, args.seed, frame_composition(),
                 expected_calls, dist, colmeans, examples)

    pooled = raw.iloc[-1]
    print(f"\npooled diagonal - off-diagonal: {pooled.contrast:+.3f} "
          f"[{pooled.ci_lower:+.3f}, {pooled.ci_upper:+.3f}]"
          f"  {'excludes 0' if pooled.excludes_zero else 'includes 0'}")
    n_disc = int(raw.iloc[:-1].pipe(lambda d: (d.ci_upper < 0)).sum())
    print(f"rows whose CI is entirely below zero: {n_disc}/{len(PRINCIPLES)}")
    print(f"diagonal lowest in its row: "
          f"{int((ranks.rank_in_row == 1).sum())}/{len(PRINCIPLES)}; "
          f"bottom two: {int((ranks.rank_in_row <= 2).sum())}/{len(PRINCIPLES)}")
    # The mirror, printed unconditionally: with a reversed sign the "lowest"
    # line alone reads as a null result when it is not one.
    print(f"diagonal highest in its row: "
          f"{int((ranks.rank_in_row_from_top == 1).sum())}/{len(PRINCIPLES)}; "
          f"top two: {int((ranks.rank_in_row_from_top <= 2).sum())}/{len(PRINCIPLES)}")
    print(f"\nWrote {_rel(args.report)} and {_rel(args.output_dir)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
