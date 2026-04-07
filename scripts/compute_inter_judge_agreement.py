"""
Compute inter-judge agreement for the 3-judge ensemble used in HumaneBench.

Reads every .eval file under logs/{baseline,good_persona,bad_persona}/<model>/
and pulls each sample's per-judge severities from
Score.metadata["individual_scores"] (populated by humanebench.scorer.overseer).

Reports:
  - Krippendorff's alpha on the 4-point ordinal scale {-1.0, -0.5, 0.5, 1.0}
  - Krippendorff's alpha after binarizing to {positive, negative} via sev >= 0
    (matches the prosocial-flip definition in scripts/create_steerability_chart.py)
  - Pairwise Cohen's kappa (unweighted and quadratic-weighted) for each of the
    three judge pairs
  - 95% bootstrap CIs for the two pooled alphas
  - Stratified breakdowns by persona, principle, and evaluated model (CSV-only)

Outputs are written under tables/:
  inter_judge_agreement.md
  inter_judge_agreement.csv
  inter_judge_agreement_by_persona.csv
  inter_judge_agreement_by_principle.csv
  inter_judge_agreement_by_model.csv
  inter_judge_raw.csv
"""

import argparse
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PERSONAS = ["baseline", "good_persona", "bad_persona"]
PRINCIPLES = [
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
]
ORDINAL_LEVELS = [-1.0, -0.5, 0.5, 1.0]
ORDINAL_TO_CAT = {v: i for i, v in enumerate(ORDINAL_LEVELS)}


def _short_judge_name(model_id: str) -> str:
    """Map full provider/model id to a short label for the report."""
    name = model_id.split("/")[-1]
    if "claude" in name:
        return "claude-4.5-sonnet"
    if "gpt" in name:
        return "gpt-5.1"
    if "gemini" in name:
        return "gemini-2.5-pro"
    return name


def _iter_eval_samples(eval_path: Path) -> Iterable[dict]:
    """Yield each sample dict from an Inspect .eval file (which is a ZIP)."""
    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = [
            f for f in zf.namelist()
            if f.startswith("samples/") and f.endswith(".json")
        ]
        for sf in sample_files:
            with zf.open(sf) as fh:
                yield json.load(fh)


def collect_long_table(logs_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Walk logs/ and build a long-format DataFrame of per-judge scores.

    Returns the DataFrame and a stats dict tracking inclusion/exclusion counts.
    """
    rows = []
    stats = {
        "total_samples": 0,
        "samples_included": 0,
        "samples_excluded_no_individual_scores": 0,
        "samples_excluded_invalid_severity": 0,
        "files_scanned": 0,
    }

    for persona in PERSONAS:
        persona_dir = logs_dir / persona
        if not persona_dir.is_dir():
            print(f"[warn] missing persona dir: {persona_dir}")
            continue
        for model_dir in sorted(persona_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for eval_path in sorted(model_dir.glob("*.eval")):
                stats["files_scanned"] += 1
                for sample in _iter_eval_samples(eval_path):
                    stats["total_samples"] += 1
                    sample_id = sample.get("id")
                    target = sample.get("target")
                    overseer = (sample.get("scores") or {}).get("overseer")
                    if not overseer:
                        stats["samples_excluded_no_individual_scores"] += 1
                        continue
                    md = overseer.get("metadata") or {}
                    individual = md.get("individual_scores")
                    judges = md.get("ensemble_models")
                    if not individual or not judges or len(individual) != len(judges):
                        stats["samples_excluded_no_individual_scores"] += 1
                        continue
                    # Drop any sample where a judge severity isn't on the canonical scale
                    if any(s not in ORDINAL_LEVELS for s in individual):
                        stats["samples_excluded_invalid_severity"] += 1
                        continue
                    principle = target or overseer.get("answer")
                    sample_uid = f"{persona}|{model_name}|{sample_id}"
                    for sev, judge_id in zip(individual, judges):
                        rows.append({
                            "sample_uid": sample_uid,
                            "persona": persona,
                            "model": model_name,
                            "principle": principle,
                            "judge_name": _short_judge_name(judge_id),
                            "severity": float(sev),
                        })
                    stats["samples_included"] += 1

    df = pd.DataFrame(rows)
    return df, stats


def _build_reliability_matrix(long_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Pivot the long table to a (judges, samples) matrix on the ordinal scale."""
    wide = long_df.pivot_table(
        index="judge_name",
        columns="sample_uid",
        values="severity",
        aggfunc="first",
    )
    # Stable judge order so pairwise pairings are reproducible across slices
    judge_order = sorted(wide.index)
    wide = wide.loc[judge_order]
    return wide.values, judge_order


def _krippendorff_with_ci(matrix: np.ndarray, level: str, n_bootstrap: int) -> dict:
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)
    result = {"alpha": float(alpha), "ci_lower": None, "ci_upper": None,
              "n_bootstrap": 0}
    if n_bootstrap <= 0 or matrix.shape[1] == 0:
        return result
    rng = np.random.default_rng(seed=20260407)
    n_items = matrix.shape[1]
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_items, size=n_items)
        resampled = matrix[:, idx]
        try:
            boot = krippendorff.alpha(
                reliability_data=resampled,
                level_of_measurement=level,
            )
        except Exception:
            continue
        if not math.isnan(boot):
            boots.append(boot)
    if boots:
        result["ci_lower"] = float(np.percentile(boots, 2.5))
        result["ci_upper"] = float(np.percentile(boots, 97.5))
        result["n_bootstrap"] = len(boots)
    return result


def _pairwise_kappas(matrix: np.ndarray, judges: list[str]) -> list[dict]:
    """Compute Cohen's kappa (unweighted and quadratic-weighted) for each pair."""
    pairs = []
    n = len(judges)
    for i in range(n):
        for j in range(i + 1, n):
            row_i = matrix[i]
            row_j = matrix[j]
            mask = ~(np.isnan(row_i) | np.isnan(row_j))
            if not mask.any():
                pairs.append({
                    "judge_a": judges[i],
                    "judge_b": judges[j],
                    "n_items": 0,
                    "kappa_unweighted": None,
                    "kappa_quadratic": None,
                    "exact_agreement": None,
                })
                continue
            a = pd.Series(row_i[mask]).map(ORDINAL_TO_CAT).values
            b = pd.Series(row_j[mask]).map(ORDINAL_TO_CAT).values
            labels = list(range(len(ORDINAL_LEVELS)))
            try:
                k_un = cohen_kappa_score(a, b, labels=labels)
            except Exception:
                k_un = None
            try:
                k_q = cohen_kappa_score(a, b, labels=labels, weights="quadratic")
            except Exception:
                k_q = None
            exact = float((a == b).mean())
            pairs.append({
                "judge_a": judges[i],
                "judge_b": judges[j],
                "n_items": int(mask.sum()),
                "kappa_unweighted": None if k_un is None else float(k_un),
                "kappa_quadratic": None if k_q is None else float(k_q),
                "exact_agreement": exact,
            })
    return pairs


def _binarize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Map severities to 1 (positive) if >= 0 else 0 (negative). NaNs preserved."""
    out = np.full_like(matrix, np.nan, dtype=float)
    mask = ~np.isnan(matrix)
    out[mask] = (matrix[mask] >= 0).astype(float)
    return out


def compute_metrics(long_df: pd.DataFrame, n_bootstrap: int) -> dict:
    """Compute pooled metrics for a long-format slice of the data."""
    if long_df.empty:
        return {
            "n_items": 0,
            "judges": [],
            "alpha_ord": None,
            "alpha_bin": None,
            "pairwise": [],
            "judge_marginals": {},
        }
    matrix, judges = _build_reliability_matrix(long_df)
    alpha_ord = _krippendorff_with_ci(matrix, "ordinal", n_bootstrap)
    bin_matrix = _binarize_matrix(matrix)
    alpha_bin = _krippendorff_with_ci(bin_matrix, "nominal", n_bootstrap)
    pairwise = _pairwise_kappas(matrix, judges)

    marginals = {}
    for j_idx, judge in enumerate(judges):
        col = matrix[j_idx]
        col = col[~np.isnan(col)]
        counts = Counter(col.tolist())
        total = len(col) or 1
        marginals[judge] = {f"{lvl:+.1f}": counts.get(lvl, 0) / total
                            for lvl in ORDINAL_LEVELS}

    # global flip rate: items where judges disagree in sign
    n_items = matrix.shape[1]
    flip_count = 0
    for col_idx in range(n_items):
        col = matrix[:, col_idx]
        col = col[~np.isnan(col)]
        if col.size < 2:
            continue
        signs = (col >= 0).astype(int)
        if signs.min() != signs.max():
            flip_count += 1
    flip_rate = flip_count / n_items if n_items else None

    return {
        "n_items": int(n_items),
        "judges": judges,
        "alpha_ord": alpha_ord,
        "alpha_bin": alpha_bin,
        "pairwise": pairwise,
        "judge_marginals": marginals,
        "sign_disagreement_rate": flip_rate,
    }


def _stratified(long_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Compute pooled metrics per slice on column `by`. No bootstrap CIs."""
    rows = []
    for value, slice_df in long_df.groupby(by):
        m = compute_metrics(slice_df, n_bootstrap=0)
        row = {
            by: value,
            "n_items": m["n_items"],
            "alpha_ord": m["alpha_ord"]["alpha"] if m["alpha_ord"] else None,
            "alpha_bin": m["alpha_bin"]["alpha"] if m["alpha_bin"] else None,
            "sign_disagreement_rate": m["sign_disagreement_rate"],
        }
        for pair in m["pairwise"]:
            tag = f"{pair['judge_a']}__vs__{pair['judge_b']}"
            row[f"kappa_unweighted__{tag}"] = pair["kappa_unweighted"]
            row[f"kappa_quadratic__{tag}"] = pair["kappa_quadratic"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(by).reset_index(drop=True)
    return df


def _format_alpha(d: dict) -> str:
    if d is None or d.get("alpha") is None:
        return "n/a"
    a = d["alpha"]
    lo, hi = d.get("ci_lower"), d.get("ci_upper")
    if lo is not None and hi is not None:
        return f"{a:.3f} [95% CI: {lo:.3f}, {hi:.3f}]"
    return f"{a:.3f}"


def write_outputs(metrics: dict, stats: dict, long_df: pd.DataFrame,
                  tables_dir: Path, n_bootstrap: int) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pooled CSV (one row, headline numbers)
    pooled_row = {
        "n_items_included": metrics["n_items"],
        "n_samples_excluded_no_individual_scores":
            stats["samples_excluded_no_individual_scores"],
        "n_samples_excluded_invalid_severity":
            stats["samples_excluded_invalid_severity"],
        "n_samples_total_scanned": stats["total_samples"],
        "alpha_ord": metrics["alpha_ord"]["alpha"],
        "alpha_ord_ci_lower": metrics["alpha_ord"]["ci_lower"],
        "alpha_ord_ci_upper": metrics["alpha_ord"]["ci_upper"],
        "alpha_bin": metrics["alpha_bin"]["alpha"],
        "alpha_bin_ci_lower": metrics["alpha_bin"]["ci_lower"],
        "alpha_bin_ci_upper": metrics["alpha_bin"]["ci_upper"],
        "sign_disagreement_rate": metrics["sign_disagreement_rate"],
        "n_bootstrap": n_bootstrap,
    }
    for pair in metrics["pairwise"]:
        tag = f"{pair['judge_a']}__vs__{pair['judge_b']}"
        pooled_row[f"kappa_unweighted__{tag}"] = pair["kappa_unweighted"]
        pooled_row[f"kappa_quadratic__{tag}"] = pair["kappa_quadratic"]
        pooled_row[f"exact_agreement__{tag}"] = pair["exact_agreement"]
    pooled_df = pd.DataFrame([pooled_row])
    pooled_df.to_csv(tables_dir / "inter_judge_agreement.csv", index=False)

    # 2. Markdown report (paste-ready)
    md = ["# Inter-judge agreement",
          "",
          f"Computed across **{metrics['n_items']:,} scored items** "
          f"(samples scanned: {stats['total_samples']:,}; "
          f"excluded due to missing individual_scores: "
          f"{stats['samples_excluded_no_individual_scores']:,}; "
          f"excluded due to off-scale severities: "
          f"{stats['samples_excluded_invalid_severity']:,}).",
          "",
          f"Eval files scanned: {stats['files_scanned']}.",
          "",
          "## Pooled (paste these into the LaTeX placeholders)",
          "",
          f"- **Krippendorff's α (ordinal, 4-point):** "
          f"{_format_alpha(metrics['alpha_ord'])}",
          f"- **Krippendorff's α (binary, sev ≥ 0):** "
          f"{_format_alpha(metrics['alpha_bin'])}",
          f"- **Sign-disagreement rate (≥1 judge disagrees in sign):** "
          f"{metrics['sign_disagreement_rate']:.3%}",
          "",
          "## Pairwise Cohen's κ (Appendix app:judges)",
          "",
          "| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for pair in metrics["pairwise"]:
        md.append(
            f"| {pair['judge_a']} × {pair['judge_b']} | {pair['n_items']:,} | "
            f"{pair['kappa_unweighted']:.3f} | {pair['kappa_quadratic']:.3f} | "
            f"{pair['exact_agreement']:.3%} |"
        )
    md += ["",
           "## Per-judge marginal distributions (sanity check)",
           "",
           "| judge | -1.0 | -0.5 | +0.5 | +1.0 |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for judge, dist in metrics["judge_marginals"].items():
        md.append(
            f"| {judge} | {dist['-1.0']:.3f} | {dist['-0.5']:.3f} | "
            f"{dist['+0.5']:.3f} | {dist['+1.0']:.3f} |"
        )
    md.append("")
    (tables_dir / "inter_judge_agreement.md").write_text("\n".join(md))

    # 3. Stratified breakdowns
    _stratified(long_df, "persona").to_csv(
        tables_dir / "inter_judge_agreement_by_persona.csv", index=False)
    _stratified(long_df, "principle").to_csv(
        tables_dir / "inter_judge_agreement_by_principle.csv", index=False)
    _stratified(long_df, "model").to_csv(
        tables_dir / "inter_judge_agreement_by_model.csv", index=False)

    # 4. Raw long-format table for appendix reproducibility
    long_df.to_csv(tables_dir / "inter_judge_raw.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "logs",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for the pooled α CIs (0 to disable).",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()
    if not logs_dir.is_dir():
        raise SystemExit(f"logs dir not found: {logs_dir}")

    print(f"Scanning {logs_dir} ...")
    long_df, stats = collect_long_table(logs_dir)
    print(f"  files scanned:        {stats['files_scanned']}")
    print(f"  samples scanned:      {stats['total_samples']:,}")
    print(f"  samples included:     {stats['samples_included']:,}")
    print(f"  excluded (no scores): {stats['samples_excluded_no_individual_scores']:,}")
    print(f"  excluded (off-scale): {stats['samples_excluded_invalid_severity']:,}")

    if long_df.empty:
        raise SystemExit("No usable per-judge scores were found in the eval logs.")

    print(f"\nComputing pooled metrics (n_bootstrap={args.n_bootstrap}) ...")
    metrics = compute_metrics(long_df, n_bootstrap=args.n_bootstrap)

    print(f"\nPooled across {metrics['n_items']:,} items "
          f"(judges: {', '.join(metrics['judges'])}):")
    print(f"  alpha_ord = {_format_alpha(metrics['alpha_ord'])}")
    print(f"  alpha_bin = {_format_alpha(metrics['alpha_bin'])}")
    print(f"  sign-disagreement rate = "
          f"{metrics['sign_disagreement_rate']:.3%}")
    print("\nPairwise Cohen's kappa:")
    for pair in metrics["pairwise"]:
        print(f"  {pair['judge_a']} x {pair['judge_b']}: "
              f"unweighted={pair['kappa_unweighted']:.3f}, "
              f"quadratic={pair['kappa_quadratic']:.3f}, "
              f"exact_agreement={pair['exact_agreement']:.3%} "
              f"(n={pair['n_items']:,})")

    print("\nPer-judge marginal distribution over the 4 severity levels:")
    for judge, dist in metrics["judge_marginals"].items():
        parts = ", ".join(f"{k}={v:.3f}" for k, v in dist.items())
        total = sum(dist.values())
        print(f"  {judge}: {parts} (sum={total:.3f})")

    print(f"\nWriting outputs to {tables_dir} ...")
    write_outputs(metrics, stats, long_df, tables_dir, args.n_bootstrap)
    print("Done.")


if __name__ == "__main__":
    main()
