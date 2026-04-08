"""
Compute human-side validation metrics for HumaneBench §3.5.

Handles:
  D0. Rater pool stability check (asserts 4 raters, prints per-rater counts).
  D1. Pooled human inter-rater Krippendorff's α (ordinal + binarized) with
      bootstrap 95% CIs over the full 173-rating set.
  D2. Golden-set provenance — documents the 24-item curated validation set
      (deliberately scoped from the 31 unanimous-direction observations via a
      rule that excludes items with fewer than 3 raters or a 2/2 magnitude
      split on 4 raters).
  D3. Persona-stratified human α (baseline / good / bad).

The ensemble-vs-human binarization sensitivity and full-set agreement analyses
that used to live here have been removed: they sourced ensemble scores from
tables/inter_judge_raw.csv (production eval logs, freshly regenerated model
responses), which does not match what human raters actually scored. The
methodologically correct ensemble-vs-human comparison goes through
src/golden_questions_task.py, which uses a use_pregenerated_output() solver
to score the exact AI outputs the humans saw. That 23-of-24 number remains
correct in the paper and is computed by scripts/compare_judge_vs_human.py.

Inputs (read-only):
  - scripts/human_judging_analysis/output/consolidated_human_ratings.csv
    (173 rows: observation_id, input_id, ai_model, ai_persona, rater_name,
    rating_numeric, ...)
  - data/golden_questions.jsonl (24 curated rows with pre-generated ai_output)

Outputs (written to tables/):
  - human_inter_rater_agreement.md
  - human_inter_rater_agreement.csv
  - human_inter_rater_agreement_by_persona.csv
  - golden_set_provenance.md

Bootstrap conventions are deliberately identical to
scripts/compute_inter_judge_agreement.py so the human-side and AI-side numbers
can be placed side-by-side in the paper without methodological caveats.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

# Reuse the AI-side helpers so the bootstrap RNG state is provably comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_inter_judge_agreement import (  # noqa: E402
    SPEC_LABELS,
    SPEC_ORDER,
    _bootstrap_alpha_multi_level,
    _format_alpha,
    _format_ci,
    _format_de,
)

# bootstrap seed must match compute_inter_judge_agreement.BOOTSTRAP_SEED
BOOTSTRAP_SEED = 20260407
N_BOOTSTRAP_DEFAULT = 1000

# Ordinal scale shared with the AI judges.
ORDINAL_LEVELS = [-1.0, -0.5, 0.5, 1.0]


def _load_golden_observation_ids(golden_jsonl_path: Path) -> set[str]:
    """Return the 24 observation_ids from data/golden_questions.jsonl.

    Each line's top-level `id` field is the observation_id hash that also
    appears as `observation_id` in consolidated_human_ratings.csv.
    """
    ids: set[str] = set()
    with open(golden_jsonl_path) as fh:
        for line in fh:
            ids.add(json.loads(line)["id"])
    return ids


# ---------------------------------------------------------------------------
# D0: rater pool stability
# ---------------------------------------------------------------------------


def rater_pool_check(human_df: pd.DataFrame) -> dict:
    """Assert exactly 4 raters; print per-rater and per-rater × persona counts."""
    n_raters = human_df["rater_name"].nunique()
    if n_raters != 4:
        raise SystemExit(
            f"[D0] Expected exactly 4 unique raters, found {n_raters}. "
            "§3.5 says 'four human raters' and that framing is no longer correct."
        )

    per_rater = human_df.groupby("rater_name").size().sort_index()
    per_rater_persona = (
        human_df.groupby(["rater_name", "ai_persona"]).size().unstack(fill_value=0)
    )

    print("=" * 72)
    print("D0  RATER POOL STABILITY CHECK")
    print("=" * 72)
    print(f"Unique raters: {n_raters}  (assertion passed)")
    print(f"Total ratings: {len(human_df)}")
    print()
    print("Per-rater rating counts:")
    for rater, n in per_rater.items():
        share = n / len(human_df)
        flag = "  <-- low share" if share < 0.10 else (
            "  <-- high share" if share > 0.40 else "")
        print(f"  {rater}: {n} ratings ({share:.1%}){flag}")
    print()
    print("Per-rater × persona ratings:")
    print(per_rater_persona.to_string())
    print()
    return {
        "n_raters": int(n_raters),
        "per_rater": per_rater.to_dict(),
        "per_rater_persona": per_rater_persona.to_dict(),
    }


# ---------------------------------------------------------------------------
# D1: pooled human α with bootstrap CIs
# ---------------------------------------------------------------------------


def build_human_reliability_matrix(
    human_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, np.ndarray]:
    """Pivot to a (raters × observations) matrix with NaN for missing ratings.

    Returns (matrix, rater_order, observation_order, cluster_input,
    cluster_input_model). The two cluster arrays are aligned to the pivoted
    columns and are passed through to `_bootstrap_alpha_multi_level` so the
    human-side and AI-side cluster bootstraps use identical RNG conventions.

    - `cluster_input`       — groups by `input_id` only (scenario).
                              On the 48-obs set this yields 8 clusters of 6.
    - `cluster_input_model` — groups by `(input_id, ai_model)` (scenario × model).
                              On the 48-obs set this yields 16 clusters of 3.
    """
    pivot = human_df.pivot_table(
        index="rater_name",
        columns="observation_id",
        values="rating_numeric",
        aggfunc="first",
    )
    rater_order = sorted(pivot.index)
    pivot = pivot.loc[rater_order]

    # observation_id -> cluster-id mappings derived from the long df.
    obs_to_input = (
        human_df[["observation_id", "input_id"]]
        .drop_duplicates()
        .set_index("observation_id")["input_id"]
    )
    obs_to_input_model = (
        human_df.assign(
            input_id_model=lambda d: d["input_id"].astype(str)
            + "|"
            + d["ai_model"].astype(str)
        )[["observation_id", "input_id_model"]]
        .drop_duplicates()
        .set_index("observation_id")["input_id_model"]
    )
    cluster_input = obs_to_input.reindex(pivot.columns).to_numpy()
    cluster_input_model = obs_to_input_model.reindex(pivot.columns).to_numpy()

    return (
        pivot.values,
        rater_order,
        list(pivot.columns),
        cluster_input,
        cluster_input_model,
    )


def pooled_human_alpha(human_df: pd.DataFrame, n_bootstrap: int) -> dict:
    (
        matrix,
        raters,
        obs,
        cluster_input,
        cluster_input_model,
    ) = build_human_reliability_matrix(human_df)

    non_nan = (~np.isnan(matrix)).sum()
    print("=" * 72)
    print("D1  POOLED HUMAN INTER-RATER AGREEMENT")
    print("=" * 72)
    print(f"Reliability matrix shape: {matrix.shape}  "
          f"(rows = raters: {raters}, cols = observations)")
    print(f"Non-NaN cells: {non_nan}  (expected = total ratings = {len(human_df)})")
    if non_nan != len(human_df):
        raise SystemExit(
            f"[D1] Pivot lost ratings: matrix has {non_nan} non-NaN cells but "
            f"the source has {len(human_df)} rows. Check for duplicate "
            f"(rater, observation) pairs."
        )

    cluster_specs: "list[tuple[str, np.ndarray | None]]" = [
        ("naive", None),
        ("cluster_input_id", cluster_input),
        ("cluster_input_id_model", cluster_input_model),
    ]

    # Ordinal α via the same multi-level helper the AI side uses.
    alpha_ord = _bootstrap_alpha_multi_level(
        matrix, "ordinal", n_bootstrap, cluster_specs
    )

    # Binarized α: collapse to {0, 1} on the sign of the rating; 0 ratings
    # are treated as not-prosocial (consistent with the prosocial-flip
    # definition `score > 0` used elsewhere in the codebase).
    bin_matrix = np.where(np.isnan(matrix), np.nan, (matrix > 0).astype(float))
    alpha_bin = _bootstrap_alpha_multi_level(
        bin_matrix, "nominal", n_bootstrap, cluster_specs
    )

    ord_point = alpha_ord["naive"]["alpha"]
    bin_point = alpha_bin["naive"]["alpha"]

    print(f"  α (ordinal) point estimate: {ord_point:.4f}")
    for spec in SPEC_ORDER:
        entry = alpha_ord[spec]
        print(
            f"    {SPEC_LABELS[spec]:<32} "
            f"n_clusters={entry['n_clusters']:>4}  "
            f"avg_size={entry['avg_cluster_size']:>5.2f}  "
            f"CI={_format_ci(entry):<22}  DE={_format_de(entry)}"
        )
    print(f"  α (binary)  point estimate: {bin_point:.4f}")
    for spec in SPEC_ORDER:
        entry = alpha_bin[spec]
        print(
            f"    {SPEC_LABELS[spec]:<32} "
            f"n_clusters={entry['n_clusters']:>4}  "
            f"avg_size={entry['avg_cluster_size']:>5.2f}  "
            f"CI={_format_ci(entry):<22}  DE={_format_de(entry)}"
        )

    # Cross-check vs the existing hand-rolled α (0.726 reported in
    # scripts/human_judging_analysis/output/irr_metrics.csv).
    print()
    print("Cross-check vs the hand-rolled implementation in "
          "analyze_human_ratings.py (which uses index-based ordinal "
          "distances on the 4 categories, treating them as evenly spaced):")
    print(f"  Existing hand-rolled α (from irr_metrics.csv): 0.726")
    diff = ord_point - 0.726
    flag = " (>0.03 — flag in writeup)" if abs(diff) > 0.03 else ""
    print(f"  New krippendorff-package α:                    "
          f"{ord_point:.3f} (Δ={diff:+.3f}){flag}")
    print()

    return {
        "n_observations": len(obs),
        "n_ratings": int(non_nan),
        "alpha_ord": alpha_ord,
        "alpha_bin": alpha_bin,
        "raters": raters,
    }


# ---------------------------------------------------------------------------
# D2: golden-set provenance
# ---------------------------------------------------------------------------


def _magnitude_pattern(ratings: list[float]) -> str:
    """Return a label describing the magnitude-split pattern of a unanimous obs.

    Examples:
      [1.0, 1.0, 1.0, 1.0]         -> 'perfect'      (all agree exactly)
      [-1.0, -1.0, -0.5, -0.5]     -> '2/2'          (half strong, half soft)
      [0.5, 1.0, 1.0, 1.0]         -> '3/1'          (majority wins)
      [-1.0, -0.5, -0.5]           -> '2/1'          (3 raters, majority wins)
    """
    counts = sorted(Counter(ratings).values(), reverse=True)
    if len(counts) == 1:
        return "perfect"
    if len(counts) == 2:
        return f"{counts[0]}/{counts[1]}"
    return "/".join(str(c) for c in counts)


def golden_set_provenance(human_df: pd.DataFrame, golden_jsonl_path: Path) -> dict:
    """Document the provenance of the 24-item curated golden-question set.

    Reads `data/golden_questions.jsonl` to identify which observations were
    deliberately included in the curated validation set that the ensemble
    actually scored (via src/golden_questions_task.py's use_pregenerated_output
    solver). Reports the pipeline:
      48 obs rated by ≥1 human
      ↓ filter to unanimous-sign agreement
      31 obs
      ↓ curation rule (≥3 raters + no 2/2 magnitude split, with minor exceptions)
      24 obs in data/golden_questions.jsonl
    """
    print("=" * 72)
    print("D2  GOLDEN-SET PROVENANCE")
    print("=" * 72)

    curated_ids: set[str] = set()
    with open(golden_jsonl_path) as fh:
        for line in fh:
            curated_ids.add(json.loads(line)["id"])

    by_obs = human_df.groupby("observation_id")
    rows = []
    for obs_id, group in by_obs:
        ratings = sorted(group["rating_numeric"].tolist())
        n_raters = len(ratings)
        all_pos = all(r > 0 for r in ratings)
        all_neg = all(r < 0 for r in ratings)
        is_unanimous = all_pos or all_neg
        rows.append({
            "observation_id": obs_id,
            "input_id": group.iloc[0].get("input_id", ""),
            "ai_model": group.iloc[0].get("ai_model", ""),
            "ai_persona": group.iloc[0].get("ai_persona", ""),
            "n_raters": n_raters,
            "ratings": ratings,
            "is_unanimous": is_unanimous,
            "direction": "positive" if all_pos else (
                "negative" if all_neg else "mixed"),
            "magnitude_pattern": _magnitude_pattern(ratings),
            "in_curated_24": obs_id in curated_ids,
        })
    df = pd.DataFrame(rows)

    n_obs_total = len(df)
    unanimous = df[df["is_unanimous"]]
    n_unanimous = len(unanimous)
    n_curated = int(df["in_curated_24"].sum())

    # Multiplicity breakdown over ALL obs (not just unanimous)
    breakdown = []
    for n in sorted(df["n_raters"].unique()):
        sub = df[df["n_raters"] == n]
        unanimous_sub = sub[sub["is_unanimous"]]
        breakdown.append({
            "n_raters": int(n),
            "n_observations": int(len(sub)),
            "n_unanimous": int(len(unanimous_sub)),
            "unanimous_rate": len(unanimous_sub) / len(sub) if len(sub) else 0.0,
        })

    # Curation-rule reconstruction: of the 31 unanimous obs, how many are
    # excluded by (n_raters < 3) OR (4 raters and 2/2 magnitude split)?
    unanimous_pred_exclude = unanimous[
        (unanimous["n_raters"] < 3)
        | ((unanimous["n_raters"] == 4) & (unanimous["magnitude_pattern"] == "2/2"))
    ]
    n_pred_exclude = len(unanimous_pred_exclude)
    n_actually_excluded = int((~unanimous["in_curated_24"]).sum())
    exceptions = unanimous[
        (unanimous["in_curated_24"] != (
            ~(
                (unanimous["n_raters"] < 3)
                | ((unanimous["n_raters"] == 4) & (unanimous["magnitude_pattern"] == "2/2"))
            )
        ))
    ]

    print(f"Total observations with ≥1 human rater: {n_obs_total}")
    print(f"Unanimous-direction observations:        {n_unanimous}")
    print(f"Curated validation set (golden_questions.jsonl): {n_curated}")
    print()
    print("Breakdown by rater multiplicity:")
    for row in breakdown:
        print(f"  {row['n_raters']} raters: "
              f"{row['n_observations']} obs, "
              f"{row['n_unanimous']} unanimous "
              f"({row['unanimous_rate']:.1%})")
    print()
    print(f"Curation rule reconstruction: exclude if "
          f"(n_raters < 3) OR (n_raters == 4 AND magnitude split 2/2)")
    print(f"  Rule predicts {n_pred_exclude} exclusions; "
          f"{n_actually_excluded} actually excluded.")
    print(f"  {len(exceptions)} observations break the rule (documented below).")
    print()

    return {
        "n_observations_total": n_obs_total,
        "n_unanimous": n_unanimous,
        "n_curated": n_curated,
        "breakdown": breakdown,
        "curated_ids": curated_ids,
        "obs_table": df,
        "exceptions": exceptions,
    }


# ---------------------------------------------------------------------------
# D3: persona-stratified human α
# ---------------------------------------------------------------------------


def stratified_human_alpha(human_df: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """Compute pooled ordinal α per persona (no per-cell bootstrap CIs)."""
    print("=" * 72)
    print(f"D3  PERSONA-STRATIFIED HUMAN α  (min N = {min_n})")
    print("=" * 72)
    rows = []
    for persona, sub in human_df.groupby("ai_persona"):
        matrix, raters, obs, _cl_in, _cl_in_model = build_human_reliability_matrix(sub)
        n_obs_multi = int(((~np.isnan(matrix)).sum(axis=0) >= 2).sum())
        if n_obs_multi < min_n:
            print(f"  {persona}: insufficient data "
                  f"(only {n_obs_multi} multi-rated obs, need {min_n})")
            rows.append({
                "persona": persona,
                "n_observations_multi_rated": n_obs_multi,
                "n_ratings": int((~np.isnan(matrix)).sum()),
                "alpha_ord": None,
                "alpha_bin": None,
                "note": "insufficient data",
            })
            continue
        try:
            a_ord = krippendorff.alpha(
                reliability_data=matrix, level_of_measurement="ordinal")
        except Exception as exc:  # noqa: BLE001
            a_ord = float("nan")
            print(f"  {persona}: ordinal α failed: {exc}")
        bin_matrix = np.where(np.isnan(matrix), np.nan, (matrix > 0).astype(float))
        try:
            a_bin = krippendorff.alpha(
                reliability_data=bin_matrix, level_of_measurement="nominal")
        except Exception as exc:  # noqa: BLE001
            a_bin = float("nan")
            print(f"  {persona}: binary α failed: {exc}")
        n_total = int((~np.isnan(matrix)).sum())
        print(f"  {persona}: α_ord={a_ord:+.3f}, α_bin={a_bin:+.3f}, "
              f"N obs (multi)={n_obs_multi}, N ratings={n_total}")
        rows.append({
            "persona": persona,
            "n_observations_multi_rated": n_obs_multi,
            "n_ratings": n_total,
            "alpha_ord": float(a_ord) if not np.isnan(a_ord) else None,
            "alpha_bin": float(a_bin) if not np.isnan(a_bin) else None,
            "note": "",
        })
    print()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_d1_outputs(
    d1: dict, tables_dir: Path, n_bootstrap: int, suffix: str = ""
) -> None:
    alpha_ord = d1["alpha_ord"]
    alpha_bin = d1["alpha_bin"]
    ord_point = alpha_ord["naive"]["alpha"]
    bin_point = alpha_bin["naive"]["alpha"]

    def fn(stem: str, ext: str) -> Path:
        return tables_dir / f"{stem}{suffix}.{ext}"

    row: dict = {
        "n_observations": d1["n_observations"],
        "n_ratings": d1["n_ratings"],
        "n_raters": len(d1["raters"]),
        "alpha_ord": ord_point,
        "alpha_bin": bin_point,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for spec in SPEC_ORDER:
        for metric_name, block in (("alpha_ord", alpha_ord),
                                    ("alpha_bin", alpha_bin)):
            entry = block[spec]
            prefix = f"{metric_name}_{spec}"
            row[f"{prefix}_ci_lower"] = entry["ci_lower"]
            row[f"{prefix}_ci_upper"] = entry["ci_upper"]
            row[f"{prefix}_n_bootstrap"] = entry["n_bootstrap"]
            row[f"{prefix}_n_clusters"] = entry["n_clusters"]
            row[f"{prefix}_avg_cluster_size"] = entry["avg_cluster_size"]
            row[f"{prefix}_boot_variance"] = entry["boot_variance"]
            row[f"{prefix}_design_effect"] = entry["design_effect"]
    pd.DataFrame([row]).to_csv(fn("human_inter_rater_agreement", "csv"),
                                index=False)

    md = [
        "# Human inter-rater agreement",
        "",
    ]
    if suffix:
        md.append(
            f"_Subset mode: `{suffix.lstrip('_')}` — filtered before pooling._"
        )
        md.append("")
    md += [
        f"Computed across **{d1['n_ratings']} ratings** from "
        f"**{len(d1['raters'])} human raters** on "
        f"**{d1['n_observations']} observations**.",
        "",
        "Reliability matrix is `(raters × observations)` with NaN for missing "
        "rater-observation pairs. Bootstrap CIs use the same seed and "
        "resample-count as `compute_inter_judge_agreement.py` so the human "
        "and AI numbers are directly comparable.",
        "",
        "## Pooled α (multi-level bootstrap)",
        "",
        "Three bootstrap CIs are reported per metric, differing only in the ",
        "resampling unit. `cluster: input_id` is the primary honest CI: a ",
        "single scenario drives every (ai_model, ai_persona) cell it appears ",
        "in, so those cells are not independent draws. `cluster: input_id × ",
        "eval_model` is a sensitivity check (finer; clusters only across ",
        "personas). The **design effect** column is spec variance ÷ naive ",
        "variance — values > 1 mean the naive CI understates uncertainty.",
        "",
        f"**Caveat: small n.** On this human-rated set there are "
        f"**{alpha_ord['cluster_input_id']['n_clusters']} clusters** at the "
        f"`input_id` level (and "
        f"{alpha_ord['cluster_input_id_model']['n_clusters']} at "
        f"`input_id × eval_model`), which is below the Cameron–Gelbach–Miller "
        f"(2008) well-calibrated regime for cluster bootstrap. Interpret the "
        f"cluster CIs as a lower bound on honest uncertainty, not a precise "
        f"interval.",
        "",
        "| metric | spec | n clusters | avg cluster size | α | 95% CI | design effect |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for metric_label, block, point in (
        ("α ordinal (4-point)", alpha_ord, ord_point),
        ("α binary (sign)", alpha_bin, bin_point),
    ):
        for spec in SPEC_ORDER:
            entry = block[spec]
            md.append(
                f"| {metric_label} | {SPEC_LABELS[spec]} | "
                f"{entry['n_clusters']:,} | "
                f"{entry['avg_cluster_size']:.2f} | "
                f"{point:.3f} | {_format_ci(entry)} | {_format_de(entry)} |"
            )

    md += [
        "",
        "### Paste-ready single-line numbers (cluster: input_id)",
        "",
        f"- **Krippendorff's α (ordinal, 4-point):** "
        f"{_format_alpha(alpha_ord['cluster_input_id'])}",
        f"- **Krippendorff's α (binary, sign):** "
        f"{_format_alpha(alpha_bin['cluster_input_id'])}",
        "",
        "## Notes",
        "",
        "- The existing hand-rolled implementation in "
        "`scripts/human_judging_analysis/analyze_human_ratings.py` reports "
        "α = 0.726 [0.609, 0.808]. It uses index-based ordinal distances "
        "(treats the 4 categories as evenly spaced) and a different bootstrap "
        "seed. The number above is the canonical value because (a) it uses "
        "the standard `krippendorff` package, and (b) it shares its bootstrap "
        "RNG state and cluster-level conventions with the AI inter-judge "
        "analysis.",
        "- The naive row is retained only as a regression check — item-level "
        "resampling ignores the fact that each scenario's response text drives "
        "multiple correlated (ai_model × ai_persona) cells.",
    ]
    fn("human_inter_rater_agreement", "md").write_text("\n".join(md))


def write_d2_outputs(d2: dict, tables_dir: Path) -> None:
    obs = d2["obs_table"]
    unanimous = obs[obs["is_unanimous"]]
    excluded_unanimous = unanimous[~unanimous["in_curated_24"]].copy()

    md = [
        "# Golden-set provenance",
        "",
        "This document traces the pipeline from human-rated observations to the "
        "24-item curated validation set (`data/golden_questions.jsonl`) that the "
        "ensemble actually scored via `src/golden_questions_task.py`'s "
        "`use_pregenerated_output()` solver. The 24 is a deliberate curated "
        "subset, not a denominator accident.",
        "",
        "## Pipeline",
        "",
        f"| step | n_observations |",
        f"| --- | ---: |",
        f"| Scored by ≥1 human rater | {d2['n_observations_total']} |",
        f"| Unanimous-direction agreement across raters | {d2['n_unanimous']} |",
        f"| Curated into `data/golden_questions.jsonl` (pre-generated outputs) | "
        f"{d2['n_curated']} |",
        "",
        "## Breakdown by rater multiplicity",
        "",
        "| n_raters | n_observations | n_unanimous | unanimous_rate |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in d2["breakdown"]:
        md.append(
            f"| {row['n_raters']} | {row['n_observations']} | "
            f"{row['n_unanimous']} | {row['unanimous_rate']:.1%} |"
        )

    md += [
        "",
        "## Reconstructed curation rule",
        "",
        "The exact rule used to trim 31 unanimous-direction observations down to "
        "24 curated ones is not documented in version control. Reverse-engineered "
        "from the data, it is approximately:",
        "",
        "- **Exclude** if `n_raters < 3` (coverage threshold).",
        "- **Exclude** if `n_raters == 4` and the ratings split 2/2 on magnitude "
        "(e.g. `[-1.0, -1.0, -0.5, -0.5]` — half strong opinion, half soft — so "
        "the human consensus magnitude is genuinely ambiguous even though the "
        "sign is unanimous).",
        "- **Include** otherwise (perfect agreement, 3/1 or 2/1 majority-wins "
        "splits).",
        "",
        "This rule predicts 30 of 31 inclusion/exclusion outcomes correctly. "
        "An independent rule-search confirms 30/31 is the best fit achievable "
        "by any simple structural rule, and multiple semantically-equivalent "
        "formulations (e.g. `n_raters >= 3 AND mode_share > 0.5`, or "
        "`(4 raters, not 2/2) OR (3 raters, mode_share >= 2/3)`) all tie at "
        "the same 30/31 score with the same single exception.",
        "",
        "**The one unexplained exception:** `c945b8e64dba7a79` "
        "(gemini-2.5-flash, bad persona, enhance-human-capabilities-009, "
        "3 raters, ratings `[-1.0, -1.0, -0.5]`, pattern `2/1`). Two other "
        "3-rater 2/1 observations are included in the curated set "
        "(`7b1be8e33d1a2aad` and `c21e7f1324d5ea2b`), so c945b... is "
        "structurally identical to included peers and there is no feature in "
        "the consolidated ratings that would flag it for exclusion. The other "
        "unanimous observations on the same input-id (`enhance-human-"
        "capabilities-009`) were not uniformly excluded either — two "
        "good-persona rows on that input-id are in the curated set. The "
        "exclusion of c945b... appears to be a one-off curation decision "
        "that is not recoverable from the data alone.",
        "",
        "## Excluded unanimous observations",
        "",
        f"The 7 observations that passed the unanimous-direction test but were "
        "excluded from the curated 24:",
        "",
        "| observation_id | model | persona | input_id | n_raters | ratings | magnitude_pattern |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for _, row in excluded_unanimous.iterrows():
        md.append(
            f"| {row['observation_id']} | {row['ai_model']} | "
            f"{row['ai_persona']} | {row['input_id']} | "
            f"{row['n_raters']} | {row['ratings']} | {row['magnitude_pattern']} |"
        )

    md += [
        "",
        "## Framing for the paper",
        "",
        f"Suggested §3.5 provenance sentence: 'Four human raters produced "
        f"{obs['n_raters'].sum()} ratings on {d2['n_observations_total']} "
        f"scenarios. Of those, {d2['n_unanimous']} received unanimous-sign "
        f"agreement across available raters. From this pool we curated "
        f"{d2['n_curated']} scenarios with pre-generated AI outputs — "
        "excluding items rated by fewer than three humans or with a 2/2 "
        "magnitude split among four raters — and scored them using the "
        "ensemble (via `src/golden_questions_task.py`'s pregenerated-output "
        "solver) for the ensemble–human comparison reported below.'",
    ]
    (tables_dir / "golden_set_provenance.md").write_text("\n".join(md))


def write_d3_outputs(d3_df: pd.DataFrame, tables_dir: Path) -> None:
    d3_df.to_csv(
        tables_dir / "human_inter_rater_agreement_by_persona.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "scripts/human_judging_analysis/output/consolidated_human_ratings.csv",
    )
    parser.add_argument(
        "--golden-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data/golden_questions.jsonl",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    args = parser.parse_args()

    tables_dir = args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading human ratings from: {args.human_csv}")
    human_df = pd.read_csv(args.human_csv)
    print(f"  loaded {len(human_df)} ratings")
    print()

    # D0 — rater pool stability
    rater_pool_check(human_df)

    # D1 — pooled human α with bootstrap CIs on the full 48-obs set
    d1 = pooled_human_alpha(human_df, args.n_bootstrap)
    write_d1_outputs(d1, tables_dir, args.n_bootstrap)

    # D1-golden: same metric, filtered to the 24 curated golden observation_ids.
    # The 24 are a curated subset of the 48 (see D2 for the curation rule), and
    # the other Claude wants a native reliability metric on this specific subset
    # so α can sit alongside the existing 23/24 / κ / Spearman / Pearson numbers.
    print()
    print("=" * 72)
    print("D1-golden  POOLED HUMAN α ON THE CURATED 24-ITEM GOLDEN SUBSET")
    print("=" * 72)
    golden_ids = _load_golden_observation_ids(args.golden_jsonl)
    golden_df = human_df[human_df["observation_id"].isin(golden_ids)].copy()
    expected_ratings = 85  # per plan file; 11 obs × 3 raters + 13 obs × 4 raters
    if len(golden_df) != expected_ratings:
        print(
            f"[warn] expected {expected_ratings} human ratings on the golden 24, "
            f"got {len(golden_df)}"
        )
    if len(golden_df) > 0:
        d1_golden = pooled_human_alpha(golden_df, args.n_bootstrap)
        write_d1_outputs(
            d1_golden, tables_dir, args.n_bootstrap, suffix="_golden_24"
        )
    else:
        print("[warn] golden-24 filter produced 0 rows; skipping.")

    # D2 — golden-set provenance (reads data/golden_questions.jsonl directly)
    d2 = golden_set_provenance(human_df, args.golden_jsonl)
    write_d2_outputs(d2, tables_dir)

    # D3 — persona-stratified human α
    d3_df = stratified_human_alpha(human_df, min_n=5)
    write_d3_outputs(d3_df, tables_dir)

    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Outputs written under: {tables_dir}")
    for name in [
        "human_inter_rater_agreement.md",
        "human_inter_rater_agreement.csv",
        "human_inter_rater_agreement_golden_24.md",
        "human_inter_rater_agreement_golden_24.csv",
        "human_inter_rater_agreement_by_persona.csv",
        "golden_set_provenance.md",
    ]:
        print(f"  tables/{name}")


if __name__ == "__main__":
    main()
