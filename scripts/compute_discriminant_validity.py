#!/usr/bin/env python3
"""Principle discriminant validity: the designed x measured matrix.

Reviewers R1 and R2 asked whether the eight principles are eight constructs or
one construct wearing eight labels. Every scenario was *designed* to probe one
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
3. The Foster Healthy Relationships / Prioritize Long-term Wellbeing cells that
   R1 named, both directions, as a paired difference.
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
)

FHR = "foster-healthy-relationships"
PLTW = "prioritize-long-term-wellbeing"
ORDINAL_LEVELS = (-1.0, -0.5, 0.5, 1.0)
JUDGE_SHORT = "gpt-5.1"


def _rel(path: Path) -> str:
    """Repo-relative where possible; absolute otherwise (e.g. --logs-dir elsewhere)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --- loading -----------------------------------------------------------------
def load_run_scores(logs_dir: Path, models: list[str]) -> tuple[pd.DataFrame, dict]:
    """Long table of the multi-label run, one row per judged call.

    Admission mirrors ``compute_inter_judge_agreement.collect_long_table``: a
    sample counts only if its severity is on the canonical 4-point scale. Using
    a looser rule here would build the matrix from a different population than
    every other table in the paper.
    """
    rows: list[dict] = []
    stats = {"samples_seen": 0, "admitted": 0, "no_score": 0, "off_scale": 0,
             "invalid_flagged": 0, "files": []}

    for model in models:
        model_dir = logs_dir / LOG_CONDITION / model
        paths = sorted(model_dir.glob("*.eval")) if model_dir.is_dir() else []
        if not paths:
            continue
        path = max(paths, key=lambda p: (len(list(prov.iter_eval_samples(p))), p.name))
        stats["files"].append(_rel(path))
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
            rows.append({
                "scenario_id": meta.get("scenario_id"),
                "source_model": meta.get("source_model") or model,
                "designed_principle": meta.get("designed_principle"),
                "scored_principle": meta.get("scored_principle") or sample.get("target"),
                "domain": meta.get("domain", ""),
                "score": float(individual[0]),
            })
            stats["admitted"] += 1

    df = pd.DataFrame(rows)
    if not df.empty and df[["scenario_id", "designed_principle", "scored_principle"]].isna().any().any():
        raise SystemExit("run logs are missing the metadata the analysis keys on")
    return df, stats


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

    pools: dict[str, list[str]] = {}
    for c, s in cluster_stratum.items():
        pools.setdefault(s, []).append(c)

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
        same_sign = float((np.sign(sub["score"]) == np.sign(sub[col])).mean())
        rho, rho_p = sp.spearmanr(sub["score"], sub[col])
        rows.append({
            "comparator": label,
            "n": int(len(sub)),
            "mean_signed_difference": point,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
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

    `discriminant_validity_design.md` section 5 wrote down three outcomes and
    what each licenses, before any data existed. Selecting between them in code
    rather than in prose is the point: it removes the freedom to describe a weak
    result as a strong one, which is exactly the freedom R3 already suspected the
    paper of using.
    """
    rows = raw[raw.designed_principle != "pooled"]
    pooled = raw[raw.designed_principle == "pooled"].iloc[0]
    discriminating = rows[rows.ci_upper < 0]["designed_principle"].tolist()
    flat = [p for p in rows["designed_principle"] if p not in discriminating]
    n_d, n_total = len(discriminating), len(rows)

    L = ["## Interpretation, pre-committed\n"]
    L.append(
        "The three outcomes below and what each licenses were written down in "
        "`discriminant_validity_design.md` section 5 **before the run**. The one "
        "that applies is selected in code from the numbers above, not chosen "
        "afterwards.\n"
    )
    L.append(f"**{n_d} of {n_total} rows** have a diagonal-minus-off-diagonal CI "
             f"entirely below zero. Pooled: {_fmt(pooled.contrast)} "
             f"[{_fmt(pooled.ci_lower)}, {_fmt(pooled.ci_upper)}].\n")

    if n_d == n_total:
        L.append(
            "**Outcome: the diagonal is distinct throughout.** The principles "
            "measure what they were designed to measure. This is "
            "construct-validity evidence. Inter-principle correlations remain "
            "high, but that is expected both for related normative constructs "
            "and as documented LLM-judge behaviour, and the diagonal contrast is "
            "robust to both.\n"
        )
    elif n_d == 0:
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
    else:
        L.append(
            f"**Outcome: mixed -- distinct for {n_d}, flat for {n_total - n_d}.** "
            "The most likely outcome and a workable one. Report per-principle. "
            "The flat rows are candidates for merging in v2 or for a stated "
            "limitation; conceding a merge candidate while defending the rest is "
            "more credible than defending all eight.\n"
        )
        L.append("Flat rows: " + ", ".join(
            f"`{PRINCIPLE_SHORT.get(p, p)}`" for p in flat) + ".\n")
        named = {FHR, PLTW}
        if named & set(flat):
            both = ("both members of that pair are" if named <= set(flat)
                    else "one member of that pair is")
            L.append(
                f"**R1 nominated Foster Healthy Relationships and Prioritize "
                f"Long-term Wellbeing, and {both} among the flat "
                "rows.** That is the reviewer's own hypothesis confirmed on the "
                "reviewer's own example, and it should be said plainly and cited "
                "as a v2 revision target rather than argued around.\n"
            )
        elif named & set(discriminating) == named:
            L.append(
                "**The pair R1 nominated -- Foster Healthy Relationships and "
                "Prioritize Long-term Wellbeing -- is not among the flat rows.** "
                "Both discriminate, so the specific objection is answered "
                "directly, while other rows are conceded.\n"
            )

    if n_d < n_total:
        L.append(
            "The asymmetry is worth stating in the paper: a flat diagonal under "
            "LLM-judge scoring is consistent with *either* genuine construct "
            "overlap *or* judge collapse, so a null here bounds what the "
            "instrument can resolve rather than refuting the framework. A "
            "*non*-flat diagonal has no such ambiguity, which is why the "
            "positive rows carry more weight than the flat ones.\n"
        )

    n_lowest = int((ranks.rank_in_row == 1).sum())
    n_bottom2 = int((ranks.rank_in_row <= 2).sum())
    n_col = int((ranks.rank_in_column == 1).sum())
    L.append(
        f"Ordinally: the diagonal is the lowest cell in its row for "
        f"**{n_lowest} of {n_total}** principles and in the bottom two for "
        f"**{n_bottom2} of {n_total}**; it is lowest in its *column* for "
        f"**{n_col} of {n_total}**, which is the version that cannot be "
        "explained by a harsh rubric.\n"
    )
    return L


def write_report(
    out: Path, matrix: DesignedMeasuredMatrix, per_model: dict,
    raw: pd.DataFrame, centered: pd.DataFrame, ranks: pd.DataFrame,
    fhr_pltw: pd.DataFrame, sanity: pd.DataFrame, pearson: pd.DataFrame,
    spearman: pd.DataFrame, n_items: int, stats: dict, manifest: dict,
    n_bootstrap: int, frame_composition: dict,
) -> None:
    pooled_raw = raw[raw.designed_principle == "pooled"].iloc[0]
    n_cell = int(np.median(matrix.n_per_cell))

    L = ["# Principle discriminant validity: the designed x measured matrix\n"]
    shortfall = manifest["n_judge_calls"] - stats["admitted"]
    if shortfall > 0:
        L.append(
            f"> **This matrix is incomplete.** {stats['admitted']:,} of "
            f"{manifest['n_judge_calls']:,} planned judge calls survived "
            f"admission ({stats['admitted'] / manifest['n_judge_calls']:.1%}); "
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
        f"({n_bootstrap:,} replicates, seed {BOOTSTRAP_SEED}, 2.5/97.5 "
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
        L.append(f"| {PRINCIPLE_SHORT.get(r.designed_principle, r.designed_principle)} | "
                 f"{_fmt(r.diagonal)} | {int(r.rank_in_row)}/{int(r.n_cells)} | "
                 f"{int(r.rank_in_column)}/{int(r.n_cells)} | "
                 f"{r.share_lowest_in_row:.0%} | {r.share_bottom_two_in_row:.0%} |")
    L.append("")

    L.append("## 3. Foster Healthy Relationships vs Prioritize Long-term Wellbeing\n")
    L.append(
        "R1's literal example: is FHR a subset of PLTW? The direct test is "
        "whether FHR-designed scenarios score differently on FHR than on PLTW, "
        "and the reverse. Paired within the bootstrap replicate.\n"
    )
    L.append("| comparison | difference | 95% CI | excludes 0 |")
    L.append("| --- | ---: | :---: | :---: |")
    for _, r in fhr_pltw.iterrows():
        L.append(f"| {r.comparison} | {_fmt(r.difference)} | "
                 f"[{_fmt(r.ci_lower)}, {_fmt(r.ci_upper)}] | "
                 f"{'yes' if r.excludes_zero else 'no'} |")
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
             f"(seed {BOOTSTRAP_SEED}). Nesting inside the decomposition "
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

    manifest = json.loads(MANIFEST_PATH.read_text())
    long, stats = load_run_scores(args.logs_dir, args.models)
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
    expected_calls = manifest["n_judge_calls"]
    shortfall = expected_calls - stats["admitted"]
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

    # Pooled raw and pooled centred are equal by construction (an additive column
    # effect cancels in the pool). If they diverge, the centring is wrong.
    if not math.isclose(raw.iloc[-1].contrast, centered.iloc[-1].contrast, abs_tol=1e-9):
        raise SystemExit(
            "pooled raw and column-centred contrasts differ; an additive column "
            "effect must cancel in the pool, so one of them is computed wrong"
        )

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

    # 3. The pair R1 named, both directions.
    fhr_rows = []
    for designed, other in ((FHR, PLTW), (PLTW, FHR)):
        d, lo, hi = matrix.cell_difference(designed, designed, other)
        fhr_rows.append({
            "comparison": f"{PRINCIPLE_SHORT[designed]}-designed: "
                          f"{PRINCIPLE_SHORT[designed]} minus {PRINCIPLE_SHORT[other]}",
            "designed_principle": designed,
            "scored_a": designed,
            "scored_b": other,
            "difference": d,
            "ci_lower": lo,
            "ci_upper": hi,
            "excludes_zero": bool(hi < 0 or lo > 0),
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

    write_report(args.report, matrix, per_model, raw, centered, ranks, fhr_pltw,
                 sanity, pearson, spearman, n_items, stats, manifest,
                 args.n_bootstrap, frame_composition())

    pooled = raw.iloc[-1]
    print(f"\npooled diagonal - off-diagonal: {pooled.contrast:+.3f} "
          f"[{pooled.ci_lower:+.3f}, {pooled.ci_upper:+.3f}]"
          f"  {'excludes 0' if pooled.excludes_zero else 'includes 0'}")
    n_disc = int(raw.iloc[:-1].pipe(lambda d: (d.ci_upper < 0)).sum())
    print(f"rows whose CI is entirely below zero: {n_disc}/{len(PRINCIPLES)}")
    print(f"diagonal lowest in its row: "
          f"{int((ranks.rank_in_row == 1).sum())}/{len(PRINCIPLES)}; "
          f"bottom two: {int((ranks.rank_in_row <= 2).sum())}/{len(PRINCIPLES)}")
    print(f"\nWrote {_rel(args.report)} and {_rel(args.output_dir)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
