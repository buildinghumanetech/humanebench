#!/usr/bin/env python3
"""Compute §4.6 vulnerable-population (VP) stratified breakdown with cluster-
bootstrap CIs and the difference-in-differences (DiD) headline stat.

Reads tables/vp_sample_scores.csv (produced by compute_vp_breakdown.py) and
writes:
  - tables/vp_breakdown.{csv,md}        — headline 6 strata × 3 personas
  - tables/vp_robustness_gap.{csv,md}   — baseline→bad_persona gap per stratum
  - tables/vp_breakdown_full.csv        — all 18 raw VP groups, no pooling
  - tables/section_4_6_paste_ready.md   — consolidated paper numbers

Cluster-bootstrap CIs at the scenario level (cluster: sample_id), seed 20260407,
n_bootstrap=1000. Matches the §3.5 scripts' convention.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from humanebench.excluded import load_excluded_ids  # noqa: E402

BOOTSTRAP_SEED = 20260407
N_BOOTSTRAP_DEFAULT = 1000
DATASET_PATH = REPO_ROOT / "data" / "humane_bench.jsonl"

# 6-stratum design (see plan).
NAMED_STRATA = {
    "": "general",
    "teenagers": "teenagers",
    "elderly": "elderly",
    "children": "children",
    "people-with-disabilities": "people-with-disabilities",
}
STRATA_ORDER = [
    "general",
    "teenagers",
    "elderly",
    "children",
    "people-with-disabilities",
    "other-VP",
]
PERSONA_ORDER = ["baseline", "good_persona", "bad_persona"]
PARTITION_LABELS = ["general", "VP_pooled"]
LOW_N_THRESHOLD = 5  # appendix flag


def map_to_stratum(vp: str) -> str:
    return NAMED_STRATA.get(vp, "other-VP")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_live_vp_map(dataset_path: Path) -> dict[str, str]:
    """Return {sample_id: vulnerable_population} from the live JSONL.

    Source of truth for VP labels — overrides the CSV's column, which can
    carry stale labels from the .eval files when the JSONL taxonomy was
    edited after the eval ran.
    """
    out: dict[str, str] = {}
    with dataset_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            sid = row.get("id")
            if not sid:
                continue
            out[sid] = (row.get("metadata") or {}).get("vulnerable-population", "") or ""
    return out


def load_data(input_csv: Path, include_excluded: bool) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    # Override CSV's vulnerable_population with the live JSONL value (the
    # CSV column reflects .eval-time metadata, which can be stale after
    # JSONL taxonomy edits).
    live_vp = _load_live_vp_map(DATASET_PATH)
    df["vulnerable_population"] = df["sample_id"].map(live_vp).fillna("")
    relabeled = (df["sample_id"].map(live_vp).notna()).sum()
    print(f"Joined live VP labels for {df['sample_id'].nunique()} unique "
          f"sample_ids ({relabeled:,} rows)")

    if not include_excluded:
        excluded = load_excluded_ids()
        before = len(df)
        df = df[~df["sample_id"].isin(excluded)]
        print(f"Dropped {before - len(df)} rows for {len(excluded)} excluded sample_ids")
    else:
        print("Loaded 0 excluded IDs (--include-excluded set)")

    df["stratum"] = df["vulnerable_population"].map(map_to_stratum)
    # >=0 prosocial threshold matches scripts/compute_binarized_robustness_gap.py:44
    df["prosocial"] = (df["score"] >= 0).astype(float)
    return df


def make_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample_id with per-persona sums/counts plus stratum + raw VP.

    Columns produced (for each persona p in PERSONA_ORDER):
      score_sum_<p>, score_n_<p>, proso_sum_<p>
    Plus: stratum, vulnerable_population, principle (one principle per sample_id by construction).
    """
    per_persona = df.groupby(["sample_id", "persona"]).agg(
        score_sum=("score", "sum"),
        score_n=("score", "count"),
        proso_sum=("prosocial", "sum"),
    )
    wide = per_persona.unstack("persona", fill_value=0.0)
    wide.columns = [f"{m}_{p}" for m, p in wide.columns]
    # Make sure all expected columns exist (a persona could be empty in pathological inputs)
    for p in PERSONA_ORDER:
        for m in ("score_sum", "score_n", "proso_sum"):
            col = f"{m}_{p}"
            if col not in wide.columns:
                wide[col] = 0.0

    meta = df.groupby("sample_id").agg(
        stratum=("stratum", "first"),
        vulnerable_population=("vulnerable_population", "first"),
        principle=("principle", "first"),
    )
    return wide.join(meta).reset_index()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bincount_means(codes, weights_sum, weights_n, n_groups):
    """Return weighted means per group (NaN where n=0)."""
    grp_n = np.bincount(codes, weights=weights_n, minlength=n_groups)
    grp_s = np.bincount(codes, weights=weights_sum, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = grp_s / grp_n
    out[grp_n == 0] = np.nan
    return out


def cluster_bootstrap(cluster_df: pd.DataFrame, n_bootstrap: int, seed: int):
    """Run one cluster-bootstrap pass; return point estimates plus bootstrap
    arrays for three groupings: 6 strata, general/VP partition, all raw VPs.

    Each replicate resamples `sample_id` clusters with replacement (size =
    n_clusters). The same resample is reused for every (grouping, persona,
    metric) so within-replicate correlation is preserved (necessary for the
    DiD CI to be valid).
    """
    n_clusters = len(cluster_df)

    stratum_codes = pd.Categorical(
        cluster_df["stratum"], categories=STRATA_ORDER
    ).codes.astype(np.int64)
    n_strata = len(STRATA_ORDER)

    partition_codes = np.where(
        cluster_df["stratum"].to_numpy() == "general", 0, 1
    ).astype(np.int64)

    raw_vps = sorted(cluster_df["vulnerable_population"].unique().tolist())
    raw_vp_codes = pd.Categorical(
        cluster_df["vulnerable_population"], categories=raw_vps
    ).codes.astype(np.int64)
    n_raw_vp = len(raw_vps)

    sums = {p: cluster_df[f"score_sum_{p}"].to_numpy(dtype=float) for p in PERSONA_ORDER}
    ns = {p: cluster_df[f"score_n_{p}"].to_numpy(dtype=float) for p in PERSONA_ORDER}
    proso = {p: cluster_df[f"proso_sum_{p}"].to_numpy(dtype=float) for p in PERSONA_ORDER}

    # Point estimates (full data, no resampling).
    def point_means(codes, n_groups):
        out = {}
        for p in PERSONA_ORDER:
            out[p] = {
                "score": _bincount_means(codes, sums[p], ns[p], n_groups),
                "proso": _bincount_means(codes, proso[p], ns[p], n_groups),
                "n_items": np.bincount(codes, weights=ns[p], minlength=n_groups).astype(int),
                "n_scenarios": np.bincount(codes, minlength=n_groups).astype(int),
            }
        return out

    point = {
        "strata": {"labels": STRATA_ORDER, "data": point_means(stratum_codes, n_strata)},
        "partition": {"labels": PARTITION_LABELS, "data": point_means(partition_codes, 2)},
        "raw_vp": {"labels": raw_vps, "data": point_means(raw_vp_codes, n_raw_vp)},
    }

    # Bootstrap arrays.
    def alloc(n_groups):
        return {
            p: {
                "score": np.empty((n_bootstrap, n_groups)),
                "proso": np.empty((n_bootstrap, n_groups)),
            }
            for p in PERSONA_ORDER
        }

    boot_strata = alloc(n_strata)
    boot_partition = alloc(2)
    boot_raw_vp = alloc(n_raw_vp)

    rng = np.random.default_rng(seed=seed)
    for b in range(n_bootstrap):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        s_pick = stratum_codes[picks]
        p_pick = partition_codes[picks]
        r_pick = raw_vp_codes[picks]
        for persona in PERSONA_ORDER:
            n_p = ns[persona][picks]
            sc_p = sums[persona][picks]
            pr_p = proso[persona][picks]
            for codes, n_g, dest in [
                (s_pick, n_strata, boot_strata),
                (p_pick, 2, boot_partition),
                (r_pick, n_raw_vp, boot_raw_vp),
            ]:
                dest[persona]["score"][b] = _bincount_means(codes, sc_p, n_p, n_g)
                dest[persona]["proso"][b] = _bincount_means(codes, pr_p, n_p, n_g)

    return {
        "n_clusters": n_clusters,
        "point": point,
        "boot": {
            "strata": {"labels": STRATA_ORDER, "data": boot_strata},
            "partition": {"labels": PARTITION_LABELS, "data": boot_partition},
            "raw_vp": {"labels": raw_vps, "data": boot_raw_vp},
        },
    }


def percentile_ci(boot_array: np.ndarray) -> tuple[float, float]:
    """95% percentile CI; returns (NaN, NaN) if all-NaN."""
    if np.all(np.isnan(boot_array)):
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(boot_array, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Output 1 — vp_breakdown
# ---------------------------------------------------------------------------

def build_breakdown_table(result, raw=False) -> pd.DataFrame:
    """One row per (stratum_or_vp, persona). If `raw=True`, uses the 18 raw VPs;
    otherwise uses the 6 strata."""
    section = result["point"]["raw_vp" if raw else "strata"]
    boot_section = result["boot"]["raw_vp" if raw else "strata"]
    labels = section["labels"]
    rows = []
    for i, label in enumerate(labels):
        for persona in PERSONA_ORDER:
            score_pt = section["data"][persona]["score"][i]
            proso_pt = section["data"][persona]["proso"][i]
            n_items = section["data"][persona]["n_items"][i]
            n_scen = section["data"][persona]["n_scenarios"][i]
            score_lo, score_hi = percentile_ci(boot_section["data"][persona]["score"][:, i])
            proso_lo, proso_hi = percentile_ci(boot_section["data"][persona]["proso"][:, i])
            row = {
                ("stratum" if not raw else "vulnerable_population"): label,
                "persona": persona,
                "n_items": int(n_items),
                "n_scenarios": int(n_scen),
                "humane_score": score_pt,
                "humane_score_ci_lower": score_lo,
                "humane_score_ci_upper": score_hi,
                "prosocial_rate": proso_pt,
                "prosocial_rate_ci_lower": proso_lo,
                "prosocial_rate_ci_upper": proso_hi,
            }
            if raw:
                row["low_n"] = bool(n_scen < LOW_N_THRESHOLD)
                if row["low_n"]:
                    row["humane_score_ci_lower"] = float("nan")
                    row["humane_score_ci_upper"] = float("nan")
                    row["prosocial_rate_ci_lower"] = float("nan")
                    row["prosocial_rate_ci_upper"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def write_breakdown_md(df: pd.DataFrame, did: dict, out_path: Path) -> None:
    lines = [
        "# Vulnerable-population breakdown (§4.6 headline)",
        "",
        "HumaneScore (mean of judge-ensemble severities on `{-1, -0.5, +0.5, +1}`) "
        "and prosocial rate (`severity >= 0`) per stratum × persona, pooled across "
        "items × 15 models. CIs are scenario-level cluster-bootstrap "
        f"(cluster: `sample_id`, seed {BOOTSTRAP_SEED}, n_bootstrap=1000, "
        "percentile method).",
        "",
        "## Headline DiD",
        "",
        f"`(Δ_VP_pooled) − (Δ_general)` where `Δ_group = mean(bad) − mean(baseline)`. "
        f"Both Δs are typically negative (adversarial erosion); a more-negative DiD ⇒ "
        f"adversarial prompting erodes VP HumaneScore *more* than general.",
        "",
        "| metric | value | 95% CI |",
        "| --- | ---: | --- |",
        f"| Δ general | {did['delta_general']:+.4f} | "
        f"[{did['delta_general_ci'][0]:+.4f}, {did['delta_general_ci'][1]:+.4f}] |",
        f"| Δ VP pooled | {did['delta_vp']:+.4f} | "
        f"[{did['delta_vp_ci'][0]:+.4f}, {did['delta_vp_ci'][1]:+.4f}] |",
        f"| **DiD** (Δ_VP − Δ_general) | **{did['did']:+.4f}** | "
        f"**[{did['did_ci'][0]:+.4f}, {did['did_ci'][1]:+.4f}]** |",
        "",
        f"**Verdict:** {did['verdict']}",
        "",
        "## Per-stratum × persona",
        "",
        "| stratum | persona | n_scenarios | n_items | HumaneScore | 95% CI | "
        "prosocial rate | 95% CI |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['stratum']} | {r['persona']} | "
            f"{r['n_scenarios']} | {r['n_items']} | "
            f"{r['humane_score']:+.3f} | "
            f"[{r['humane_score_ci_lower']:+.3f}, {r['humane_score_ci_upper']:+.3f}] | "
            f"{r['prosocial_rate']:.3f} | "
            f"[{r['prosocial_rate_ci_lower']:.3f}, {r['prosocial_rate_ci_upper']:.3f}] |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Output 2 — vp_robustness_gap
# ---------------------------------------------------------------------------

def build_robustness_gap_table(result) -> pd.DataFrame:
    point = result["point"]["strata"]["data"]
    boot = result["boot"]["strata"]["data"]
    rows = []
    for i, stratum in enumerate(STRATA_ORDER):
        n_baseline = point["baseline"]["n_items"][i]
        n_bad = point["bad_persona"]["n_items"][i]
        n_scen = point["baseline"]["n_scenarios"][i]
        bin_baseline = point["baseline"]["proso"][i]
        bin_bad = point["bad_persona"]["proso"][i]
        ord_baseline = point["baseline"]["score"][i]
        ord_bad = point["bad_persona"]["score"][i]
        bin_gap = bin_baseline - bin_bad
        ord_gap = ord_baseline - ord_bad

        bin_gap_boot = boot["baseline"]["proso"][:, i] - boot["bad_persona"]["proso"][:, i]
        ord_gap_boot = boot["baseline"]["score"][:, i] - boot["bad_persona"]["score"][:, i]
        bin_lo, bin_hi = percentile_ci(bin_gap_boot)
        ord_lo, ord_hi = percentile_ci(ord_gap_boot)

        ratio = bin_gap / ord_gap if ord_gap not in (0.0,) and not np.isnan(ord_gap) else float("nan")
        rows.append({
            "stratum": stratum,
            "n_scenarios": int(n_scen),
            "n_items_baseline": int(n_baseline),
            "n_items_bad_persona": int(n_bad),
            "prosocial_rate_baseline": bin_baseline,
            "prosocial_rate_bad_persona": bin_bad,
            "binarized_gap_baseline_to_bad": bin_gap,
            "gap_bin_ci_lower": bin_lo,
            "gap_bin_ci_upper": bin_hi,
            "mean_severity_baseline": ord_baseline,
            "mean_severity_bad_persona": ord_bad,
            "ordinal_gap_baseline_to_bad": ord_gap,
            "gap_ord_ci_lower": ord_lo,
            "gap_ord_ci_upper": ord_hi,
            "gap_ratio": ratio,
        })
    return pd.DataFrame(rows)


def write_robustness_gap_md(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# VP robustness gap (baseline → bad_persona)",
        "",
        "Per-stratum erosion under adversarial prompting. **Positive gap = "
        "baseline behavior eroded by the bad-persona system prompt.** "
        "Mirrors `tables/robustness_gap_binarized.md` but stratifies by "
        "vulnerable-population stratum instead of by model. Cluster-bootstrap "
        "CIs at scenario level (cluster: `sample_id`, seed "
        f"{BOOTSTRAP_SEED}, n_bootstrap=1000).",
        "",
        "Conventions: `binarized_gap = prosocial_rate_baseline − "
        "prosocial_rate_bad_persona`; `ordinal_gap = mean_severity_baseline − "
        "mean_severity_bad_persona`; `gap_ratio = binarized_gap / ordinal_gap`.",
        "",
        "| stratum | n_scen | proso_base | proso_bad | bin_gap | bin 95% CI | "
        "ord_base | ord_bad | ord_gap | ord 95% CI | ratio |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['stratum']} | {r['n_scenarios']} | "
            f"{r['prosocial_rate_baseline']:.3f} | {r['prosocial_rate_bad_persona']:.3f} | "
            f"{r['binarized_gap_baseline_to_bad']:+.3f} | "
            f"[{r['gap_bin_ci_lower']:+.3f}, {r['gap_bin_ci_upper']:+.3f}] | "
            f"{r['mean_severity_baseline']:+.3f} | {r['mean_severity_bad_persona']:+.3f} | "
            f"{r['ordinal_gap_baseline_to_bad']:+.3f} | "
            f"[{r['gap_ord_ci_lower']:+.3f}, {r['gap_ord_ci_upper']:+.3f}] | "
            f"{r['gap_ratio']:+.3f} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# DiD
# ---------------------------------------------------------------------------

def compute_did(result) -> dict:
    point = result["point"]["partition"]["data"]
    boot = result["boot"]["partition"]["data"]
    # Index 0 = general, 1 = VP_pooled.
    # Both deltas are typically negative (bad < baseline). Define
    # DiD = Δ_VP − Δ_general so that:
    #   DiD < 0  ⇒  Δ_VP is more negative than Δ_general  ⇒  VP eroded MORE
    #              (the amplification finding).
    #   DiD > 0  ⇒  general eroded more than VP.
    delta_general = point["bad_persona"]["score"][0] - point["baseline"]["score"][0]
    delta_vp = point["bad_persona"]["score"][1] - point["baseline"]["score"][1]
    did = delta_vp - delta_general

    delta_general_boot = boot["bad_persona"]["score"][:, 0] - boot["baseline"]["score"][:, 0]
    delta_vp_boot = boot["bad_persona"]["score"][:, 1] - boot["baseline"]["score"][:, 1]
    did_boot = delta_vp_boot - delta_general_boot

    delta_general_ci = percentile_ci(delta_general_boot)
    delta_vp_ci = percentile_ci(delta_vp_boot)
    did_ci = percentile_ci(did_boot)

    if did_ci[0] > 0 or did_ci[1] < 0:
        if did < 0:
            verdict = (
                f"Adversarial prompting erodes HumaneScore **more** for VP items than "
                f"general (DiD = {did:+.4f}, 95% CI [{did_ci[0]:+.4f}, {did_ci[1]:+.4f}])."
            )
        else:
            verdict = (
                f"General-audience HumaneScore drops **more** than VP under adversarial "
                f"prompting (DiD = {did:+.4f}, 95% CI [{did_ci[0]:+.4f}, {did_ci[1]:+.4f}])."
            )
    else:
        verdict = (
            f"No detectable amplification: DiD = {did:+.4f}, 95% CI "
            f"[{did_ci[0]:+.4f}, {did_ci[1]:+.4f}] crosses zero."
        )

    n_items_general = int(point["baseline"]["n_items"][0]) + int(point["bad_persona"]["n_items"][0])
    n_items_vp = int(point["baseline"]["n_items"][1]) + int(point["bad_persona"]["n_items"][1])
    return {
        "delta_general": float(delta_general),
        "delta_general_ci": delta_general_ci,
        "delta_vp": float(delta_vp),
        "delta_vp_ci": delta_vp_ci,
        "did": float(did),
        "did_ci": did_ci,
        "n_items_general": n_items_general,
        "n_items_vp": n_items_vp,
        "n_scenarios_general": int(point["baseline"]["n_scenarios"][0]),
        "n_scenarios_vp": int(point["baseline"]["n_scenarios"][1]),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Output 4 — paste-ready
# ---------------------------------------------------------------------------

def write_paste_ready(
    breakdown_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    did: dict,
    n_clusters: int,
    out_path: Path,
) -> None:
    # Identify the 3 strata with the largest baseline→bad ordinal gap (excluding 'general').
    non_general = gap_df[gap_df["stratum"] != "general"].sort_values(
        "ordinal_gap_baseline_to_bad", ascending=False
    )
    top_gap_lines = []
    for _, r in non_general.head(3).iterrows():
        top_gap_lines.append(
            f"- **{r['stratum']}** (n={r['n_scenarios']}): ordinal gap "
            f"{r['ordinal_gap_baseline_to_bad']:+.3f} "
            f"[{r['gap_ord_ci_lower']:+.3f}, {r['gap_ord_ci_upper']:+.3f}]; "
            f"prosocial-rate gap {r['binarized_gap_baseline_to_bad']:+.3f} "
            f"[{r['gap_bin_ci_lower']:+.3f}, {r['gap_bin_ci_upper']:+.3f}]"
        )

    # Compact per-stratum HumaneScore table (3 personas as columns).
    pivot = breakdown_df.pivot(index="stratum", columns="persona", values="humane_score")
    pivot = pivot.reindex(STRATA_ORDER)[PERSONA_ORDER]
    n_scen_by_stratum = breakdown_df[breakdown_df["persona"] == "baseline"].set_index(
        "stratum"
    )["n_scenarios"].reindex(STRATA_ORDER)
    compact_lines = [
        "| stratum | n_scen | baseline | good_persona | bad_persona |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stratum in STRATA_ORDER:
        compact_lines.append(
            f"| {stratum} | {int(n_scen_by_stratum[stratum])} | "
            f"{pivot.loc[stratum, 'baseline']:+.3f} | "
            f"{pivot.loc[stratum, 'good_persona']:+.3f} | "
            f"{pivot.loc[stratum, 'bad_persona']:+.3f} |"
        )

    lines = [
        "# §4.6 (Vulnerable-Population Breakdown) — Paste-Ready Numbers",
        "",
        "Single source of truth for the §4.6 paragraph. All numbers come from "
        "`tables/vp_breakdown.{csv,md}`, `tables/vp_robustness_gap.{csv,md}`, "
        "and `tables/vp_breakdown_full.csv` — produced by "
        "`scripts/compute_vp_stats.py`. Cluster-bootstrap CIs at the scenario "
        f"level (cluster: `sample_id`, seed {BOOTSTRAP_SEED}, n_bootstrap=1000, "
        "percentile method). Default-excludes 12 confabulation-flagged items "
        "via `humanebench.excluded.load_excluded_ids()` for a working n of "
        f"**{n_clusters}** scenarios.",
        "",
        "---",
        "",
        "## Headline DiD",
        "",
        f"**{did['verdict']}**",
        "",
        "Decomposition (item-pooled HumaneScore on `{-1, -0.5, +0.5, +1}`):",
        "",
        "| component | value | 95% CI |",
        "| --- | ---: | --- |",
        f"| Δ general (bad − baseline)      | {did['delta_general']:+.4f} | "
        f"[{did['delta_general_ci'][0]:+.4f}, {did['delta_general_ci'][1]:+.4f}] |",
        f"| Δ VP pooled (bad − baseline)    | {did['delta_vp']:+.4f} | "
        f"[{did['delta_vp_ci'][0]:+.4f}, {did['delta_vp_ci'][1]:+.4f}] |",
        f"| **DiD** (Δ_VP − Δ_general)      | **{did['did']:+.4f}** | "
        f"**[{did['did_ci'][0]:+.4f}, {did['did_ci'][1]:+.4f}]** |",
        "",
        f"Sample sizes: general n_scenarios = {did['n_scenarios_general']} "
        f"(n_items = {did['n_items_general']:,}); VP_pooled n_scenarios = "
        f"{did['n_scenarios_vp']} (n_items = {did['n_items_vp']:,}).",
        "",
        "---",
        "",
        "## Per-stratum HumaneScore by persona",
        "",
        "Item-pooled across 15 models. Full CIs in `tables/vp_breakdown.md`.",
        "",
        *compact_lines,
        "",
        "---",
        "",
        "## Per-stratum baseline→bad erosion (top 3 non-general)",
        "",
        *(top_gap_lines or ["- (no non-general strata available)"]),
        "",
        "Full per-stratum gaps (with prosocial-rate gap CIs and gap ratios) "
        "in `tables/vp_robustness_gap.md`.",
        "",
        "---",
        "",
        "## Caveat: `other-VP` is not a coherent subpopulation",
        "",
        "The `other-VP` stratum pools 13 distinct groups (non-native-speakers, "
        "low-tech-literacy, women, low-income-communities, neurodivergent-people, "
        "gender-diverse-people, low-literacy-users, marginalized-groups, "
        "low-connectivity-users, religious-minorities, transgender-people, "
        "refugees, shift-workers) for statistical power on the DiD, **not** "
        "because those groups share behavior profiles. For per-group detail "
        "see the appendix table `tables/vp_breakdown_full.csv`.",
        "",
        "## Figure references for §4.6 LaTeX",
        "",
        "- Combined two-up bad-persona heatmap (Children + Teenagers): "
        "`figures/vp_heatmap_combined_children_teenagers_bad_persona.png`",
        "- Single-VP heatmaps (children/teenagers/elderly × baseline/good/bad): "
        "`figures/vp_heatmap_<vp>_<persona>.png` (9 figures, restyled)",
        "- VP comparison dot chart: `figures/vp_dot_chart_comparison.png`",
        "- VP grouped bar chart: `figures/vp_grouped_bar_comparison.png`",
        "",
    ]
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "tables" / "vp_sample_scores.csv",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=REPO_ROOT / "tables",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include the 12 confabulation-flagged items if present in the "
             "input CSV (only meaningful if the upstream "
             "compute_vp_breakdown.py was run with the same flag).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading: {args.input}")
    df = load_data(args.input, include_excluded=args.include_excluded)
    print(f"  {len(df):,} rows, {df['sample_id'].nunique()} unique sample_ids")

    cluster_df = make_cluster_summary(df)
    print(f"Cluster-summary rows: {len(cluster_df)}")

    print(f"Bootstrapping (n={args.n_bootstrap}, seed={args.seed}, "
          f"cluster: sample_id) ...")
    result = cluster_bootstrap(cluster_df, args.n_bootstrap, args.seed)

    args.tables_dir.mkdir(parents=True, exist_ok=True)

    # Output 1 — vp_breakdown
    breakdown_df = build_breakdown_table(result, raw=False)
    breakdown_csv = args.tables_dir / "vp_breakdown.csv"
    breakdown_md = args.tables_dir / "vp_breakdown.md"
    breakdown_df.to_csv(breakdown_csv, index=False)

    # Output 2 — vp_robustness_gap
    gap_df = build_robustness_gap_table(result)
    gap_csv = args.tables_dir / "vp_robustness_gap.csv"
    gap_md = args.tables_dir / "vp_robustness_gap.md"
    gap_df.to_csv(gap_csv, index=False)
    write_robustness_gap_md(gap_df, gap_md)

    # Output 3 — vp_breakdown_full (appendix; raw 18-VP, no md)
    full_df = build_breakdown_table(result, raw=True)
    full_csv = args.tables_dir / "vp_breakdown_full.csv"
    full_df.to_csv(full_csv, index=False)

    # DiD + paste-ready
    did = compute_did(result)
    write_breakdown_md(breakdown_df, did, breakdown_md)
    paste_path = args.tables_dir / "section_4_6_paste_ready.md"
    write_paste_ready(breakdown_df, gap_df, did, result["n_clusters"], paste_path)

    print()
    print("=" * 72)
    print("§4.6 VP BREAKDOWN — DONE")
    print("=" * 72)
    print(f"DiD = {did['did']:+.4f}, 95% CI [{did['did_ci'][0]:+.4f}, "
          f"{did['did_ci'][1]:+.4f}]")
    print(f"Verdict: {did['verdict']}")
    print()
    print("Wrote:")
    for p in (breakdown_csv, breakdown_md, gap_csv, gap_md, full_csv, paste_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()
