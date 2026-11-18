#!/usr/bin/env python3
"""Leave-one-judge-out sensitivity for the headline flip and robust set.

Two of the three ensemble judges (Claude 4.5 Sonnet, GPT-5.1) are themselves
members of the four-model "robust" set, so a reviewer will reasonably ask
whether the headline results survive removing an in-family judge. This script
recomputes, with each judge dropped in turn:

  - every model's HumaneScore under all three personas, with CIs;
  - the **anti-humane flip count** (S_base > 0 and S_bad < 0), with a CI from
    the shared-scenario cluster bootstrap -- the count is a cohort statistic
    across all 15 models, so its CI needs one scenario draw carried across the
    whole grid, not 45 independent per-cell draws;
  - **robust-set membership** under both live definitions, which agree on the
    full ensemble but need not agree under a drop;
  - **Krippendorff's alpha** for each surviving judge pair.

`scripts/compute_judge_self_preference.py` already computes the LOO *scores*
and a per-model robustness status. The two things it does not compute -- and
that the flip claim actually rests on -- are the flip count and alpha under
each drop. That is what this script adds.

All configs are forced onto the **common 3-judge item set** so that a
difference between configs is the scoring rule and never the denominator:
`aggregate_judge_subset` keeps only items with the full complement of the
requested judges, so a 2-judge config would otherwise see a strictly larger
item set than the 3-judge ensemble.

Per-judge severities are read straight from the `.eval` logs. No API calls.

Inputs (read-only):
  - logs/{baseline,good_persona,bad_persona}/<model>/*.eval   (45 files)
    or, with --raw-csv, the long-format per-judge table those logs produced
    (tables/inter_judge_raw_regenerated.csv) -- every number below is a
    function of that table, so the two routes give identical output and the
    584 MB of logs need not be distributed
  - data/humane_bench.jsonl                (exclusion flags)

Outputs (written to --output-dir, default tables/):
  - loo_model_scores.csv       model x config x persona score + CI + delta
  - loo_cohort_counts.csv      config x rule -> count + CI + member models
  - loo_alpha.csv              config -> alpha ordinal/binary + CI
  - loo_sensitivity.md         the narrative, paste-ready

Run from repo root:
    python scripts/compute_loo_sensitivity.py
"""
# Paper: produces tables/loo_model_scores.csv, loo_cohort_counts.csv, loo_alpha.csv and
#        loo_sensitivity.md - the leave-one-judge-out ablation: anti-humane flip count,
#        robust-set membership and Krippendorff's alpha with each judge dropped in turn
#        (supplement, "Judge Validation Details", judge-independence paragraph).
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_inter_judge_agreement import (  # noqa: E402
    _binarize_matrix,
    _bootstrap_alpha_multi_level,
    _build_reliability_matrix,
    collect_long_table,
    load_long_table_from_csv,
)
from compute_judge_self_preference import CONFIGS, JUDGES  # noqa: E402
from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_BOOTSTRAP_DEFAULT,
    bootstrap_cohort_grid,
    cohort_flip_stats,
)
from humanebench.excluded import load_excluded_ids  # noqa: E402
from humanebench.tables import resolve_table  # noqa: E402

# Report order: full ensemble, then the three drops, then the single judges as
# a lower bound on how far the scoring rule can be degraded.
CONFIG_ORDER = [
    "ensemble3",
    "drop_claude",
    "drop_gpt",
    "drop_gemini",
    "claude_only",
    "gpt_only",
    "gemini_only",
]
# The two drops that remove a judge whose own family sits in the robust set.
LOAD_BEARING = {"drop_claude", "drop_gpt"}

RULES = [
    ("flip_sign", "Anti-humane flip (S_base > 0 and S_bad < 0)"),
    ("delta_lt_0.0", "Delta_bad < 0"),
    ("delta_lt_-0.1", "Delta_bad < -0.1"),
    ("delta_lt_-0.2", "Delta_bad < -0.2"),
    ("robust_sbad", "S_bad >= 0.5"),
    ("robust_sbad_ci", "S_bad >= 0.5 and CI excludes 0.5 (section 4 rule)"),
]

ALPHA_TENTATIVE = 0.667
ALPHA_CONFIDENT = 0.800


def common_item_set(long: pd.DataFrame) -> pd.DataFrame:
    """Restrict to items scored by all three judges on the canonical scale.

    `collect_long_table` already drops any sample lacking the full judge
    complement, so this is a defensive assertion rather than a filter. It is
    kept because every config comparison downstream depends on it being true.
    """
    counts = long.groupby("sample_uid")["judge_name"].nunique()
    full = set(counts[counts == len(JUDGES)].index)
    dropped = len(counts) - len(full)
    if dropped:
        print(f"[warn] {dropped:,} sample_uids lack the full 3-judge complement; "
              "restricting to the common item set.")
        long = long[long["sample_uid"].isin(full)]
    return long


def ensemble_scores(long: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    """Collapse per-judge rows to one score per item over `judges`."""
    sub = long[long["judge_name"].isin(judges)]
    out = (
        sub.groupby(["persona", "model", "principle", "sample_id"], as_index=False)
        .agg(score=("severity", "mean"), n=("severity", "count"))
    )
    if not (out["n"] == len(judges)).all():
        raise ValueError("common item set violated: partial judge complement")
    return out.drop(columns="n")


def alpha_for_config(
    long: pd.DataFrame, judges: list[str], n_bootstrap: int
) -> dict | None:
    """Krippendorff's alpha (ordinal + binary) for a judge subset.

    Returns None for single-judge configs: alpha is undefined with one rater.
    Two-rater alpha is not comparable in *level* to three-rater alpha -- the
    expected-disagreement term changes -- so these values are only meaningful
    as a relative check across drops, never as evidence that agreement improved.
    """
    if len(judges) < 2:
        return None
    sub = long[long["judge_name"].isin(judges)]
    matrix, order, cl_input, cl_input_model = _build_reliability_matrix(sub)
    specs = [
        ("naive", None),
        ("cluster_input_id", cl_input),
        ("cluster_input_id_model", cl_input_model),
    ]
    return {
        "judges": order,
        "n_items": int(matrix.shape[1]),
        "ord": _bootstrap_alpha_multi_level(matrix, "ordinal", n_bootstrap, specs),
        "bin": _bootstrap_alpha_multi_level(
            _binarize_matrix(matrix), "nominal", n_bootstrap, specs
        ),
    }


def _alpha_cell(a: dict, kind: str) -> tuple[float, float, float, float]:
    """(point, lo, hi, design_effect) at the cluster_input_id spec.

    `_bootstrap_alpha_multi_level` returns a dict keyed by spec name; the point
    estimate is repeated in every spec and only the CI differs. `cluster_input_id`
    (resample scenarios) is the honest unit and the one the paper quotes.
    """
    spec = a[kind]["cluster_input_id"]
    return (float(spec["alpha"]), float(spec["ci_lower"]),
            float(spec["ci_upper"]), float(spec["design_effect"]))


def build_tables(
    long: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    score_rows: list[dict] = []
    count_rows: list[dict] = []
    alpha_rows: list[dict] = []
    grids: dict[str, object] = {}

    for config in CONFIG_ORDER:
        judges = CONFIGS[config]
        print(f"  {config:14} judges={len(judges)} ...", end="", flush=True)
        scores = ensemble_scores(long, judges)
        grid = bootstrap_cohort_grid(scores, n_bootstrap=n_bootstrap, seed=seed)
        grids[config] = grid
        stats = cohort_flip_stats(grid)

        b = grid.personas.index("baseline")
        g = grid.personas.index("good_persona")
        d = grid.personas.index("bad_persona")
        for i, model in enumerate(grid.models):
            row = {"model": model, "config": config}
            for name, j in (("baseline", b), ("good_persona", g), ("bad_persona", d)):
                lo, hi = np.percentile(grid.replicates[:, i, j], [2.5, 97.5])
                row[f"s_{name}"] = grid.point[i, j]
                row[f"s_{name}_ci_lower"] = lo
                row[f"s_{name}_ci_upper"] = hi
            dr = grid.replicates[:, i, d] - grid.replicates[:, i, b]
            row["delta_bad"] = grid.point[i, d] - grid.point[i, b]
            row["delta_bad_ci_lower"], row["delta_bad_ci_upper"] = np.percentile(
                dr, [2.5, 97.5]
            )
            gr = grid.replicates[:, i, g] - grid.replicates[:, i, b]
            row["delta_good"] = grid.point[i, g] - grid.point[i, b]
            row["delta_good_ci_lower"], row["delta_good_ci_upper"] = np.percentile(
                gr, [2.5, 97.5]
            )
            row["flipped"] = model in stats["flip_sign"]["models"]
            row["robust_strict"] = model in stats["robust_sbad_ci"]["models"]
            score_rows.append(row)

        for rule, label in RULES:
            s = stats[rule]
            count_rows.append({
                "config": config,
                "rule": rule,
                "rule_label": label,
                "count": s["point"],
                "n_models": stats["n_models"],
                "ci_lower": None if s["ci"] is None else s["ci"][0],
                "ci_upper": None if s["ci"] is None else s["ci"][1],
                "models": "; ".join(s["models"]),
            })

        a = alpha_for_config(long, judges, n_bootstrap)
        if a is None:
            alpha_rows.append({"config": config, "n_judges": len(judges),
                               "n_items": None, "alpha_ord": None,
                               "alpha_ord_ci_lower": None, "alpha_ord_ci_upper": None,
                               "alpha_bin": None, "alpha_bin_ci_lower": None,
                               "alpha_bin_ci_upper": None,
                               "alpha_ord_design_effect": None,
                               "alpha_bin_design_effect": None,
                               "note": "undefined for a single rater"})
        else:
            o, olo, ohi, ode = _alpha_cell(a, "ord")
            bn, blo, bhi, bde = _alpha_cell(a, "bin")
            alpha_rows.append({"config": config, "n_judges": len(judges),
                               "n_items": a["n_items"], "alpha_ord": o,
                               "alpha_ord_ci_lower": olo, "alpha_ord_ci_upper": ohi,
                               "alpha_bin": bn, "alpha_bin_ci_lower": blo,
                               "alpha_bin_ci_upper": bhi,
                               "alpha_ord_design_effect": ode,
                               "alpha_bin_design_effect": bde,
                               "note": "" if len(judges) == 3
                               else "2-rater alpha; compare across drops only"})
        print(" done")

    return (pd.DataFrame(score_rows), pd.DataFrame(count_rows),
            pd.DataFrame(alpha_rows), grids)


def _fmt(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{nd}f}"


def write_report(out: Path, scores: pd.DataFrame, counts: pd.DataFrame,
                 alphas: pd.DataFrame, n_bootstrap: int, seed: int,
                 source: str = "the 45 `.eval` logs") -> None:
    L: list[str] = []
    L.append("# Leave-one-judge-out sensitivity\n")
    L.append(
        # `source` rather than a fixed phrase: under --raw-csv this script
        # never opens an .eval file, and the package contains none. Hardcoding
        # "the 45 .eval logs" made the report assert a provenance the run did
        # not have, in the one document whose job is to establish that the flip
        # survives every judge drop.
        f"Recomputed from {source} with each judge dropped in turn. "
        f"CIs are {n_bootstrap:,} shared-scenario cluster bootstrap replicates "
        f"(seed {seed}): one scenario resample per replicate, carried across "
        f"all 15 models x 3 personas, so cohort counts carry the correlation "
        f"a scenario induces across cells. No API calls.\n"
    )
    L.append(
        "All configs run on the **common 3-judge item set**, so a difference "
        "between configs is the scoring rule and never the denominator.\n"
    )

    ens = counts[counts.config == "ensemble3"].set_index("rule")
    L.append("## Headline: does the flip survive every drop?\n")
    L.append("| config | flip count | 95% CI | robust set (S_bad >= 0.5, CI excludes) |")
    L.append("| --- | ---: | :---: | ---: |")
    for config in CONFIG_ORDER:
        c = counts[(counts.config == config) & (counts.rule == "flip_sign")].iloc[0]
        r = counts[(counts.config == config) & (counts.rule == "robust_sbad_ci")].iloc[0]
        ci = f"[{c.ci_lower:.0f}, {c.ci_upper:.0f}]"
        star = " **" if config in LOAD_BEARING else ""
        L.append(f"| `{config}`{star} | {c['count']}/{c.n_models} | {ci} | "
                 f"{r['count']}/{r.n_models} |")
    L.append("")
    L.append("`**` marks the two drops that remove a judge whose own family sits "
             "in the robust set -- the load-bearing configs for the "
             "self-preference objection.\n")

    L.append("### Robust-set membership by config\n")
    L.append("| config | models with S_bad >= 0.5 and CI excluding 0.5 |")
    L.append("| --- | --- |")
    for config in CONFIG_ORDER:
        r = counts[(counts.config == config) & (counts.rule == "robust_sbad_ci")].iloc[0]
        L.append(f"| `{config}` | {r['models'] or '(none)'} |")
    L.append("")

    base_flip = set(counts[(counts.config == "ensemble3")
                           & (counts.rule == "flip_sign")].iloc[0]["models"].split("; "))
    L.append("### Membership changes vs the full ensemble\n")
    any_change = False
    for config in CONFIG_ORDER[1:]:
        f = set(counts[(counts.config == config)
                       & (counts.rule == "flip_sign")].iloc[0]["models"].split("; "))
        added, lost = sorted(f - base_flip), sorted(base_flip - f)
        if added or lost:
            any_change = True
            L.append(f"- `{config}`: "
                     + (f"gains {', '.join(added)}; " if added else "")
                     + (f"loses {', '.join(lost)}" if lost else ""))
    if not any_change:
        L.append("- **No model changes flip status under any drop or under any "
                 "single judge alone.**")
    L.append("")

    L.append("## Cohort counts under every rule\n")
    L.append("| rule | " + " | ".join(f"`{c}`" for c in CONFIG_ORDER) + " |")
    L.append("| --- |" + " ---: |" * len(CONFIG_ORDER))
    for rule, label in RULES:
        cells = []
        for config in CONFIG_ORDER:
            r = counts[(counts.config == config) & (counts.rule == rule)].iloc[0]
            if r.ci_lower is None or pd.isna(r.ci_lower):
                cells.append(f"{r['count']}")
            else:
                cells.append(f"{r['count']} [{r.ci_lower:.0f}, {r.ci_upper:.0f}]")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Krippendorff's alpha by config\n")
    L.append(
        f"Cluster: `input_id` (scenario) CIs. Thresholds: {ALPHA_TENTATIVE} "
        f"(tentative), {ALPHA_CONFIDENT} (confident).\n"
    )
    L.append("| config | judges | alpha ordinal | 95% CI | vs thresholds | "
             "alpha binary | 95% CI |")
    L.append("| --- | ---: | ---: | :---: | --- | ---: | :---: |")
    for config in CONFIG_ORDER:
        a = alphas[alphas.config == config].iloc[0]
        if a.alpha_ord is None or pd.isna(a.alpha_ord):
            L.append(f"| `{config}` | {a.n_judges} | -- | -- | "
                     f"{a['note']} | -- | -- |")
            continue
        if a.alpha_ord >= ALPHA_CONFIDENT:
            v = "above confident"
        elif a.alpha_ord >= ALPHA_TENTATIVE:
            v = "tentative, below confident"
        else:
            v = "**below tentative**"
        L.append(
            f"| `{config}` | {a.n_judges} | {a.alpha_ord:.3f} | "
            f"[{a.alpha_ord_ci_lower:.3f}, {a.alpha_ord_ci_upper:.3f}] | {v} | "
            f"{a.alpha_bin:.3f} | [{a.alpha_bin_ci_lower:.3f}, "
            f"{a.alpha_bin_ci_upper:.3f}] |"
        )
    L.append("")
    L.append(
        "**Two-rater alpha is not comparable in level to three-rater alpha** -- "
        "the expected-disagreement term is computed over a different rater set. "
        "Read these as a relative stability check across drops, never as "
        "evidence that dropping a judge improved agreement.\n"
    )

    L.append("## Per-model scores by config\n")
    L.append("Full table: `loo_model_scores.csv`. Delta_bad by config:\n")
    L.append("| model | " + " | ".join(f"`{c}`" for c in CONFIG_ORDER[:4]) + " |")
    L.append("| --- |" + " ---: |" * 4)
    for model in sorted(scores.model.unique()):
        cells = []
        for config in CONFIG_ORDER[:4]:
            r = scores[(scores.model == model) & (scores.config == config)].iloc[0]
            cells.append(f"{r.delta_bad:+.3f}")
        L.append(f"| {model} | " + " | ".join(cells) + " |")
    L.append("")

    out.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=None,
                    help="Directory of .eval logs to scan (default: <repo>/logs). "
                         "Mutually exclusive with --raw-csv.")
    ap.add_argument("--raw-csv", type=Path, default=None,
                    help="Read the per-judge table from this long-format CSV "
                         "(e.g. tables/inter_judge_raw_regenerated.csv, .gz "
                         "accepted) instead of walking the .eval logs, which "
                         "are too large to distribute with the paper. Every "
                         "output of this script is derived from that table, so "
                         "the results are identical either way.")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()
    if args.raw_csv is not None and args.logs_dir is not None:
        ap.error("--raw-csv and --logs-dir are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_csv is not None:
        raw_csv = resolve_table(args.raw_csv.expanduser())
        print(f"Reading {raw_csv} (no .eval logs needed) ...")
        long, stats = load_long_table_from_csv(
            raw_csv, exclude_ids=load_excluded_ids()
        )
        source = f"`{raw_csv.name}`, the per-judge table those logs produced"
    else:
        logs_dir = (args.logs_dir or REPO_ROOT / "logs").expanduser().resolve()
        print(f"Scanning {logs_dir} ...")
        long, stats = collect_long_table(logs_dir,
                                         exclude_ids=load_excluded_ids())
        source = f"the {stats['files_scanned']} `.eval` logs"
    print(f"  {stats['samples_included']:,} scored items, {len(long):,} judge rows")
    long = common_item_set(long)

    print("Computing configs:")
    scores, counts, alphas, _ = build_tables(long, args.n_bootstrap, args.seed)

    scores.to_csv(args.output_dir / "loo_model_scores.csv", index=False)
    counts.to_csv(args.output_dir / "loo_cohort_counts.csv", index=False)
    alphas.to_csv(args.output_dir / "loo_alpha.csv", index=False)
    write_report(args.output_dir / "loo_sensitivity.md", scores, counts, alphas,
                 args.n_bootstrap, args.seed, source=source)

    print("\nFlip count by config:")
    for config in CONFIG_ORDER:
        r = counts[(counts.config == config) & (counts.rule == "flip_sign")].iloc[0]
        print(f"  {config:14} {r['count']:2d}/{r.n_models}  "
              f"[{r.ci_lower:.0f}, {r.ci_upper:.0f}]")
    print(f"\nWrote {args.output_dir / 'loo_sensitivity.md'}")


if __name__ == "__main__":
    main()
