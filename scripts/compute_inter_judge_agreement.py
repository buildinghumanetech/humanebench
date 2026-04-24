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


def _load_exclude_ids(path: Path | None) -> set[str]:
    """Load item IDs from a cut-list file (lines starting with # are comments)."""
    if path is None:
        return set()
    ids = set()
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if line:
                ids.add(line)
    return ids


def collect_long_table(
    logs_dir: Path,
    exclude_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Walk logs/ and build a long-format DataFrame of per-judge scores.

    Returns the DataFrame and a stats dict tracking inclusion/exclusion counts.
    """
    exclude = exclude_ids or set()
    rows = []
    stats = {
        "total_samples": 0,
        "samples_included": 0,
        "samples_excluded_no_individual_scores": 0,
        "samples_excluded_invalid_severity": 0,
        "samples_excluded_cut_list": 0,
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
                    if sample_id in exclude:
                        stats["samples_excluded_cut_list"] += 1
                        continue
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
                    sample_id_model = f"{sample_id}|{model_name}"
                    for sev, judge_id in zip(individual, judges):
                        rows.append({
                            "sample_uid": sample_uid,
                            "persona": persona,
                            "model": model_name,
                            "principle": principle,
                            "sample_id": sample_id,  # scenario (cluster level 1)
                            "sample_id_model": sample_id_model,  # scenario × eval model (cluster level 2)
                            "judge_name": _short_judge_name(judge_id),
                            "severity": float(sev),
                        })
                    stats["samples_included"] += 1

    df = pd.DataFrame(rows)
    return df, stats


def collect_golden_long_table(
    eval_files: list[Path],
) -> tuple[pd.DataFrame, dict]:
    """Load per-judge scores from golden_questions_eval .eval file(s).

    Unlike `collect_long_table`, this does not infer (persona, eval_model) from
    the directory path — the golden task (`src/golden_questions_task.py`) uses
    `use_pregenerated_output()` and the sample metadata carries the original
    `input_id`, `ai_model`, and `ai_persona` under the nested key
    `sample.metadata["metadata"]` (because the task's FieldSpec uses
    `metadata=["metadata"]`, which preserves the JSONL metadata block verbatim).

    Returns (long_df, stats) in the same schema as `collect_long_table` so the
    downstream `compute_metrics` / `write_outputs` pipeline works unchanged.
    """
    rows = []
    stats = {
        "total_samples": 0,
        "samples_included": 0,
        "samples_excluded_no_individual_scores": 0,
        "samples_excluded_invalid_severity": 0,
        "samples_excluded_missing_metadata": 0,
        "files_scanned": len(eval_files),
    }
    for eval_path in sorted(eval_files):
        for sample in _iter_eval_samples(eval_path):
            stats["total_samples"] += 1
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
            if any(s not in ORDINAL_LEVELS for s in individual):
                stats["samples_excluded_invalid_severity"] += 1
                continue

            # Golden sample metadata is nested one level deeper because the
            # task file uses FieldSpec(metadata=["metadata"]). Fall back to a
            # flat read in case Inspect ever changes the promotion behavior.
            sample_meta = sample.get("metadata") or {}
            inner = sample_meta.get("metadata")
            if not isinstance(inner, dict):
                inner = sample_meta
            input_id = inner.get("input_id") or sample.get("target")
            ai_model = inner.get("ai_model")
            ai_persona = inner.get("ai_persona")
            if not input_id or not ai_model or not ai_persona:
                stats["samples_excluded_missing_metadata"] += 1
                continue

            persona = _HUMAN_PERSONA_MAP.get(ai_persona, ai_persona)
            # Normalize model name to the production corpus convention
            # (golden JSONL has `gpt-4o`; production uses `gpt-4o-2024-11-20`)
            # so the golden-24 sample_uids are a strict subset of the 48-slice
            # sample_uids and the paste-ready comparison is well-defined.
            model_name = _HUMAN_MODEL_MAP.get(ai_model, ai_model)
            sample_uid = f"{persona}|{model_name}|{input_id}"
            sample_id_model = f"{input_id}|{model_name}"
            principle = inner.get("principle") or sample.get("target")
            for sev, judge_id in zip(individual, judges):
                rows.append({
                    "sample_uid": sample_uid,
                    "persona": persona,
                    "model": model_name,
                    "principle": principle,
                    "sample_id": input_id,
                    "sample_id_model": sample_id_model,
                    "judge_name": _short_judge_name(judge_id),
                    "severity": float(sev),
                })
            stats["samples_included"] += 1

    df = pd.DataFrame(rows)
    return df, stats


def _build_reliability_matrix(
    long_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Pivot the long table to a (judges, samples) matrix on the ordinal scale.

    Returns (matrix, judge_order, cluster_input, cluster_input_model) where the
    two cluster arrays are length-`matrix.shape[1]` and map each column to its
    scenario (`sample_id` == `input_id`) and its scenario-×-eval-model cell.
    These are used by the cluster bootstrap in `_bootstrap_alpha_multi_level`.
    """
    wide = long_df.pivot_table(
        index="judge_name",
        columns="sample_uid",
        values="severity",
        aggfunc="first",
    )
    # Stable judge order so pairwise pairings are reproducible across slices
    judge_order = sorted(wide.index)
    wide = wide.loc[judge_order]

    # Build sample_uid -> cluster-id mappings (one entry per unique sample_uid).
    uid_to_scenario = (
        long_df[["sample_uid", "sample_id"]]
        .drop_duplicates()
        .set_index("sample_uid")["sample_id"]
    )
    uid_to_scenario_model = (
        long_df[["sample_uid", "sample_id_model"]]
        .drop_duplicates()
        .set_index("sample_uid")["sample_id_model"]
    )
    cluster_input = uid_to_scenario.reindex(wide.columns).to_numpy()
    cluster_input_model = uid_to_scenario_model.reindex(wide.columns).to_numpy()

    return wide.values, judge_order, cluster_input, cluster_input_model


BOOTSTRAP_SEED = 20260407


def _bootstrap_alpha_multi_level(
    matrix: np.ndarray,
    level: str,
    n_bootstrap: int,
    cluster_specs: "list[tuple[str, np.ndarray | None]]",
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Compute Krippendorff's α with bootstrap CIs at multiple cluster levels.

    `cluster_specs` is an ordered list of (name, cluster_id_array_or_None).
    When the array is None, items are resampled individually (naive
    Efron bootstrap — the legacy behavior). When the array is provided,
    it must have length matrix.shape[1]; resampling is at the cluster
    level: sample `n_clusters` cluster labels with replacement and
    concatenate every column belonging to each selected cluster.

    Each spec gets its own fresh RNG seeded with the same `seed` so that
    replicate k starts from identical RNG state across specs. Differences
    in the resulting bootstrap variance are then attributable only to the
    resampling unit, which makes the design effect
    (spec_variance / naive_variance) a clean apples-to-apples diagnostic.

    Returns
    -------
    dict keyed by spec name with per-spec sub-dicts containing:
        alpha            : point estimate (identical across specs)
        ci_lower, ci_upper : 2.5 / 97.5 percentile bootstrap CI
        n_bootstrap      : number of successful replicates
        n_clusters       : unique cluster count (= n_items for naive)
        avg_cluster_size : n_items / n_clusters (= 1.0 for naive)
        boot_variance    : sample variance of the bootstrap replicates (ddof=1)
        design_effect    : boot_variance / naive_boot_variance (1.0 for naive)
        boot_samples     : raw list of bootstrap α values (for optional
                           downstream use; not written to CSV).
    """
    # Point-estimate α may be undefined when the reliability matrix collapses
    # to a degenerate value domain (e.g., a small stratified slice where every
    # cell shares the same binary label). In that case krippendorff.alpha
    # raises "There has to be more than one value in the domain." Recording
    # the point estimate as None lets downstream writers render "n/a" rather
    # than crashing the whole pass.
    try:
        point_estimate: "float | None" = float(
            krippendorff.alpha(
                reliability_data=matrix, level_of_measurement=level
            )
        )
    except ValueError:
        point_estimate = None
    n_items = matrix.shape[1]
    out: dict = {}

    for spec_name, cluster_ids in cluster_specs:
        entry = {
            "alpha": point_estimate,
            "ci_lower": None,
            "ci_upper": None,
            "n_bootstrap": 0,
            "n_clusters": None,
            "avg_cluster_size": None,
            "boot_variance": None,
            "design_effect": None,
            "boot_samples": [],
        }

        if cluster_ids is None:
            n_clusters = n_items
            avg_cluster_size = 1.0
            cluster_to_cols = None
        else:
            ids = np.asarray(cluster_ids)
            if ids.shape[0] != n_items:
                raise ValueError(
                    f"cluster_ids length {ids.shape[0]} does not match "
                    f"matrix.shape[1]={n_items} for spec '{spec_name}'"
                )
            unique, inverse = np.unique(ids, return_inverse=True)
            n_clusters = len(unique)
            avg_cluster_size = n_items / n_clusters if n_clusters else 0.0
            cluster_to_cols = [
                np.where(inverse == k)[0] for k in range(n_clusters)
            ]

        entry["n_clusters"] = int(n_clusters)
        entry["avg_cluster_size"] = float(avg_cluster_size)

        if (
            n_bootstrap <= 0
            or n_items == 0
            or n_clusters == 0
            or point_estimate is None  # degenerate domain; no bootstrap to run
        ):
            out[spec_name] = entry
            continue

        rng = np.random.default_rng(seed=seed)
        boots: list[float] = []
        for _ in range(n_bootstrap):
            if cluster_to_cols is None:
                idx = rng.integers(0, n_items, size=n_items)
            else:
                picks = rng.integers(0, n_clusters, size=n_clusters)
                idx = np.concatenate([cluster_to_cols[p] for p in picks])
            resampled = matrix[:, idx]
            try:
                boot = krippendorff.alpha(
                    reliability_data=resampled,
                    level_of_measurement=level,
                )
            except Exception:
                continue
            if not math.isnan(boot):
                boots.append(float(boot))

        if boots:
            entry["ci_lower"] = float(np.percentile(boots, 2.5))
            entry["ci_upper"] = float(np.percentile(boots, 97.5))
            entry["n_bootstrap"] = len(boots)
            entry["boot_variance"] = float(np.var(boots, ddof=1))
            entry["boot_samples"] = boots

        out[spec_name] = entry

    # Attach design effects relative to the naive spec (if present).
    naive_entry = None
    for spec_name, cluster_ids in cluster_specs:
        if cluster_ids is None:
            naive_entry = out.get(spec_name)
            break
    naive_var = None if naive_entry is None else naive_entry.get("boot_variance")
    for spec_name, _ in cluster_specs:
        entry = out[spec_name]
        spec_var = entry.get("boot_variance")
        if spec_var is None or naive_var is None or naive_var == 0:
            entry["design_effect"] = 1.0 if spec_var == naive_var else None
        else:
            entry["design_effect"] = float(spec_var / naive_var)

    return out


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
    """Compute pooled metrics for a long-format slice of the data.

    Pooled α comes with three bootstrap CIs:
      - naive              : item-level Efron bootstrap (legacy; too narrow
                             because items within a scenario are correlated)
      - cluster_input_id   : resample scenarios (the honest cluster unit —
                             the response text for a given prompt drives all
                             judges' scores for every (persona, model) cell)
      - cluster_input_id_model : resample scenario × eval-model cells
                             (finer, clusters only across personas)
    The design effect for each cluster spec = spec_var / naive_var.
    """
    if long_df.empty:
        return {
            "n_items": 0,
            "judges": [],
            "alpha_ord": None,
            "alpha_bin": None,
            "pairwise": [],
            "judge_marginals": {},
        }
    matrix, judges, cluster_input, cluster_input_model = _build_reliability_matrix(long_df)

    cluster_specs: "list[tuple[str, np.ndarray | None]]" = [
        ("naive", None),
        ("cluster_input_id", cluster_input),
        ("cluster_input_id_model", cluster_input_model),
    ]

    alpha_ord = _bootstrap_alpha_multi_level(
        matrix, "ordinal", n_bootstrap, cluster_specs
    )
    bin_matrix = _binarize_matrix(matrix)
    alpha_bin = _bootstrap_alpha_multi_level(
        bin_matrix, "nominal", n_bootstrap, cluster_specs
    )
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
            "alpha_ord": (m["alpha_ord"]["naive"]["alpha"]
                          if m["alpha_ord"] else None),
            "alpha_bin": (m["alpha_bin"]["naive"]["alpha"]
                          if m["alpha_bin"] else None),
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
    """Format an α result for single-line display.

    Accepts either a per-spec sub-dict ({alpha, ci_lower, ci_upper, ...}) or
    a full multi-level result dict keyed by spec name (in which case the
    `cluster_input_id` spec is used as the canonical display).
    """
    if d is None:
        return "n/a"
    if "alpha" not in d:
        # Multi-level result dict — prefer cluster_input_id, fall back to naive.
        for preferred in ("cluster_input_id", "naive"):
            if preferred in d:
                return _format_alpha(d[preferred])
        return "n/a"
    if d.get("alpha") is None:
        return "n/a"
    a = d["alpha"]
    lo, hi = d.get("ci_lower"), d.get("ci_upper")
    if lo is not None and hi is not None:
        return f"{a:.3f} [95% CI: {lo:.3f}, {hi:.3f}]"
    return f"{a:.3f}"


SPEC_LABELS: dict = {
    "naive": "item-level (naive)",
    "cluster_input_id": "cluster: input_id",
    "cluster_input_id_model": "cluster: input_id × eval_model",
}
SPEC_ORDER: list = ["naive", "cluster_input_id", "cluster_input_id_model"]


def _format_ci(entry: dict) -> str:
    lo, hi = entry.get("ci_lower"), entry.get("ci_upper")
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def _format_de(entry: dict) -> str:
    de = entry.get("design_effect")
    if de is None:
        return "n/a"
    return f"{de:.2f}"


def write_outputs(metrics: dict, stats: dict, long_df: pd.DataFrame,
                  tables_dir: Path, n_bootstrap: int,
                  suffix: str = "") -> None:
    """Write the pooled / stratified / raw outputs.

    `suffix` is appended to every output filename before the extension
    (e.g. suffix="_human_slice" → tables/inter_judge_agreement_human_slice.md).
    Used by main() to produce a parallel set of outputs on the 48-scenario
    AI-judge slice in the same run.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    def fn(stem: str, ext: str) -> Path:
        return tables_dir / f"{stem}{suffix}.{ext}"

    alpha_ord = metrics["alpha_ord"]
    alpha_bin = metrics["alpha_bin"]
    # Point estimates are spec-invariant; use naive as the canonical source.
    ord_point = alpha_ord["naive"]["alpha"]
    bin_point = alpha_bin["naive"]["alpha"]

    # 1. Pooled CSV (one row, headline numbers)
    pooled_row: dict = {
        "n_items_included": metrics["n_items"],
        "n_samples_excluded_no_individual_scores":
            stats.get("samples_excluded_no_individual_scores"),
        "n_samples_excluded_invalid_severity":
            stats.get("samples_excluded_invalid_severity"),
        "n_samples_total_scanned": stats.get("total_samples"),
        "alpha_ord": ord_point,
        "alpha_bin": bin_point,
        "sign_disagreement_rate": metrics["sign_disagreement_rate"],
        "n_bootstrap": n_bootstrap,
    }
    for spec in SPEC_ORDER:
        for metric_name, block in (("alpha_ord", alpha_ord),
                                    ("alpha_bin", alpha_bin)):
            entry = block[spec]
            prefix = f"{metric_name}_{spec}"
            pooled_row[f"{prefix}_ci_lower"] = entry["ci_lower"]
            pooled_row[f"{prefix}_ci_upper"] = entry["ci_upper"]
            pooled_row[f"{prefix}_n_bootstrap"] = entry["n_bootstrap"]
            pooled_row[f"{prefix}_n_clusters"] = entry["n_clusters"]
            pooled_row[f"{prefix}_avg_cluster_size"] = entry["avg_cluster_size"]
            pooled_row[f"{prefix}_boot_variance"] = entry["boot_variance"]
            pooled_row[f"{prefix}_design_effect"] = entry["design_effect"]

    for pair in metrics["pairwise"]:
        tag = f"{pair['judge_a']}__vs__{pair['judge_b']}"
        pooled_row[f"kappa_unweighted__{tag}"] = pair["kappa_unweighted"]
        pooled_row[f"kappa_quadratic__{tag}"] = pair["kappa_quadratic"]
        pooled_row[f"exact_agreement__{tag}"] = pair["exact_agreement"]
    pd.DataFrame([pooled_row]).to_csv(fn("inter_judge_agreement", "csv"),
                                      index=False)

    # 2. Markdown report (paste-ready).
    # Build the provenance line defensively — slice_stats may be a subset.
    prov_bits = []
    if stats.get("total_samples") is not None:
        prov_bits.append(f"samples scanned: {stats['total_samples']:,}")
    if stats.get("samples_excluded_no_individual_scores") is not None:
        prov_bits.append(
            f"excluded (no individual_scores): "
            f"{stats['samples_excluded_no_individual_scores']:,}")
    if stats.get("samples_excluded_invalid_severity") is not None:
        prov_bits.append(
            f"excluded (off-scale): "
            f"{stats['samples_excluded_invalid_severity']:,}")

    md = ["# Inter-judge agreement", ""]
    if suffix:
        md.append(
            f"_Subset mode: `{suffix.lstrip('_')}` — filtered before pooling._")
        md.append("")
    md.append(
        f"Computed across **{metrics['n_items']:,} scored items**"
        + (f" ({'; '.join(prov_bits)})." if prov_bits else ".")
    )
    md.append("")
    if stats.get("files_scanned") is not None:
        md.append(f"Eval files scanned: {stats['files_scanned']}.")
        md.append("")

    # Headline recommendation prose.
    md += [
        "## Pooled α (multi-level bootstrap)",
        "",
        "Three bootstrap CIs are reported per metric, differing only in the ",
        "resampling unit:",
        "",
        "- **item-level (naive)** — resamples individual judge × sample rows. ",
        "  Ignores within-scenario correlation and is the *narrow* CI reported ",
        "  in earlier drafts. Retained here as a regression check and reference.",
        "- **cluster: input_id** — resamples scenarios; every (model, persona) ",
        "  cell belonging to a selected scenario is carried along. This is the ",
        "  honest cluster unit: the response text for a given prompt drives ",
        "  every judge's score for that prompt across the (model, persona) grid.",
        "- **cluster: input_id × eval_model** — resamples scenario × eval-model ",
        "  cells (finer; clusters only across personas). A sensitivity check ",
        "  for the primary `cluster: input_id` CI.",
        "",
        "The **design effect** column = spec variance ÷ naive variance. ",
        "Values > 1 mean the naive CI understates uncertainty. On the 48-item ",
        "human-rated slice, n_clusters = 8 for `input_id`, which is below the ",
        "Cameron–Gelbach–Miller (2008) well-calibrated regime; interpret that ",
        "CI as a lower bound on honest uncertainty rather than a precise number.",
        "",
        "| metric | spec | n clusters | avg cluster size | α | 95% CI | design effect |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for metric_label, metric_key, point in (
        ("α ordinal (4-point)", "alpha_ord", ord_point),
        ("α binary (sev ≥ 0)", "alpha_bin", bin_point),
    ):
        block = metrics[metric_key]
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
        f"**Sign-disagreement rate (≥1 judge disagrees in sign):** "
        f"{metrics['sign_disagreement_rate']:.3%}",
        "",
        "### Paste-ready single-line numbers",
        "",
        f"- **α (ordinal, 4-point), cluster: input_id:** "
        f"{_format_alpha(alpha_ord['cluster_input_id'])}",
        f"- **α (binary, sev ≥ 0), cluster: input_id:** "
        f"{_format_alpha(alpha_bin['cluster_input_id'])}",
        "",
        "## Pairwise Cohen's κ (Appendix app:judges)",
        "",
        "| pair | n | κ (unweighted) | κ (quadratic-weighted) | exact agreement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
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
    fn("inter_judge_agreement", "md").write_text("\n".join(md))

    # 3. Stratified breakdowns (no bootstrap CIs, unchanged logic).
    _stratified(long_df, "persona").to_csv(
        fn("inter_judge_agreement_by_persona", "csv"), index=False)
    _stratified(long_df, "principle").to_csv(
        fn("inter_judge_agreement_by_principle", "csv"), index=False)
    _stratified(long_df, "model").to_csv(
        fn("inter_judge_agreement_by_model", "csv"), index=False)

    # 4. Raw long-format table for appendix reproducibility
    long_df.to_csv(fn("inter_judge_raw", "csv"), index=False)


# Maps the human ratings CSV's persona/model labels to the canonical
# corpus values used in `sample_uid = {persona}|{model}|{sample_id}`.
_HUMAN_PERSONA_MAP = {
    "baseline": "baseline",
    "good": "good_persona",
    "bad": "bad_persona",
}
_HUMAN_MODEL_MAP = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gemini-2.5-flash": "gemini-2.5-flash",
}


def _filter_to_human_slice(
    long_df: pd.DataFrame, human_csv: Path
) -> tuple[pd.DataFrame, dict, int]:
    """Restrict `long_df` to the sample_uids that humans rated.

    Returns (filtered_long_df, slice_stats, n_target_uids). `slice_stats`
    uses the same keys as the collector's `stats` dict so write_outputs()
    can render the provenance line without branching.
    """
    if not human_csv.is_file():
        raise FileNotFoundError(human_csv)
    human = pd.read_csv(human_csv)
    required = {"input_id", "ai_model", "ai_persona"}
    missing = required - set(human.columns)
    if missing:
        raise ValueError(
            f"{human_csv} missing expected columns: {missing}"
        )
    human = human.copy()
    human["corpus_persona"] = human["ai_persona"].map(_HUMAN_PERSONA_MAP)
    human["corpus_model"] = human["ai_model"].astype(str).map(_HUMAN_MODEL_MAP)
    unmapped_persona = human[human["corpus_persona"].isna()]["ai_persona"].unique()
    unmapped_model = human[human["corpus_model"].isna()]["ai_model"].unique()
    if len(unmapped_persona) or len(unmapped_model):
        raise ValueError(
            f"Unmapped persona/model labels in {human_csv.name}: "
            f"persona={list(unmapped_persona)}, model={list(unmapped_model)}. "
            f"Extend _HUMAN_PERSONA_MAP / _HUMAN_MODEL_MAP."
        )
    human["sample_uid"] = (
        human["corpus_persona"] + "|" + human["corpus_model"] + "|" + human["input_id"]
    )
    target_uids = set(human["sample_uid"].unique())
    matched = long_df[long_df["sample_uid"].isin(target_uids)].copy()
    matched_uids = set(matched["sample_uid"].unique())
    missing_uids = target_uids - matched_uids
    if missing_uids:
        print(
            f"[warn] {len(missing_uids)}/{len(target_uids)} human-rated "
            f"sample_uids absent from the AI corpus long table. Examples: "
            f"{sorted(missing_uids)[:3]}"
        )
    slice_stats = {
        "total_samples": len(matched_uids),
        "samples_included": len(matched_uids),
        "samples_excluded_no_individual_scores": 0,
        "samples_excluded_invalid_severity": 0,
        "files_scanned": None,
        "human_rated_target_uids": len(target_uids),
        "human_rated_matched_uids": len(matched_uids),
    }
    return matched, slice_stats, len(target_uids)


def _print_pooled_summary(metrics: dict, label: str) -> None:
    print(f"\nPooled across {metrics['n_items']:,} items ({label}):")
    for metric_label, metric_key in (
        ("alpha_ord", "alpha_ord"),
        ("alpha_bin", "alpha_bin"),
    ):
        block = metrics[metric_key]
        print(f"  {metric_label} point estimate = {block['naive']['alpha']:.4f}")
        for spec in SPEC_ORDER:
            entry = block[spec]
            ci = _format_ci(entry)
            de = _format_de(entry)
            print(
                f"    {SPEC_LABELS[spec]:<32} "
                f"n_clusters={entry['n_clusters']:>7,}  "
                f"avg_size={entry['avg_cluster_size']:>6.2f}  "
                f"CI={ci:<22}  DE={de}"
            )
    print(
        f"  sign-disagreement rate = "
        f"{metrics['sign_disagreement_rate']:.3%}"
    )


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
    parser.add_argument(
        "--human-ratings-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "scripts/human_judging_analysis/output/consolidated_human_ratings.csv",
        help="If this file exists, compute metrics a second time on the "
             "subset of sample_uids that humans rated and write a parallel "
             "set of outputs with a '_human_slice' suffix.",
    )
    parser.add_argument(
        "--golden-eval-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "logs/golden_questions_eval",
        help="If this directory contains .eval files from a "
             "golden_questions_eval run (samples carry input_id/ai_model/"
             "ai_persona under sample.metadata['metadata']), compute metrics "
             "a third time on those samples and write parallel outputs with "
             "a '_golden_24' suffix.",
    )
    parser.add_argument(
        "--exclude-ids",
        type=Path,
        default=None,
        help="Override: path to a cut-list file (one ID per line, # comments ok). "
             "If given, replaces the default dataset-derived exclusion set. "
             "Samples whose id matches are excluded from the full-corpus and "
             "human-slice passes. Golden-24 pass is unaffected.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Disable default exclusion: include items tagged "
             "metadata.excluded_from_analysis=True in data/humane_bench.jsonl. "
             "Mutually exclusive with --exclude-ids.",
    )
    args = parser.parse_args()

    if args.exclude_ids is not None and args.include_excluded:
        parser.error("--include-excluded and --exclude-ids are mutually exclusive")

    logs_dir = args.logs_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()
    if not logs_dir.is_dir():
        raise SystemExit(f"logs dir not found: {logs_dir}")

    if args.exclude_ids is not None:
        exclude_ids = _load_exclude_ids(args.exclude_ids)
        print(f"Excluding {len(exclude_ids)} item IDs from {args.exclude_ids}.")
    elif args.include_excluded:
        exclude_ids = set()
        print("Including all items (--include-excluded set).")
    else:
        from humanebench.excluded import load_excluded_ids
        exclude_ids = load_excluded_ids()
        print(f"Excluding {len(exclude_ids)} items tagged "
              f"excluded_from_analysis=True in data/humane_bench.jsonl.")

    print(f"Scanning {logs_dir} ...")
    long_df, stats = collect_long_table(logs_dir, exclude_ids=exclude_ids)
    print(f"  files scanned:        {stats['files_scanned']}")
    print(f"  samples scanned:      {stats['total_samples']:,}")
    print(f"  samples included:     {stats['samples_included']:,}")
    print(f"  excluded (cut list):  {stats['samples_excluded_cut_list']:,}")
    print(f"  excluded (no scores): {stats['samples_excluded_no_individual_scores']:,}")
    print(f"  excluded (off-scale): {stats['samples_excluded_invalid_severity']:,}")

    if long_df.empty:
        raise SystemExit("No usable per-judge scores were found in the eval logs.")

    print(f"\nComputing pooled metrics (n_bootstrap={args.n_bootstrap}) ...")
    metrics = compute_metrics(long_df, n_bootstrap=args.n_bootstrap)

    _print_pooled_summary(
        metrics, label=f"full corpus, judges: {', '.join(metrics['judges'])}"
    )

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

    # Optional second pass: restrict to the 48 sample_uids humans rated.
    human_csv = args.human_ratings_csv.expanduser().resolve()
    if human_csv.is_file():
        print(f"\n--- Human-slice pass (restrict to {human_csv.name}) ---")
        try:
            slice_df, slice_stats, n_target = _filter_to_human_slice(long_df, human_csv)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[warn] skipping human-slice pass: {exc}")
        else:
            print(
                f"Matched {len(slice_df)//3 if len(slice_df) else 0} / "
                f"{n_target} human-rated sample_uids "
                f"(long-format rows: {len(slice_df):,})"
            )
            if slice_df.empty:
                print("[warn] human slice is empty; skipping.")
            else:
                slice_metrics = compute_metrics(
                    slice_df, n_bootstrap=args.n_bootstrap
                )
                _print_pooled_summary(slice_metrics, label="48-scenario human slice")
                print(f"\nWriting human-slice outputs to {tables_dir} ...")
                write_outputs(
                    slice_metrics,
                    slice_stats,
                    slice_df,
                    tables_dir,
                    args.n_bootstrap,
                    suffix="_human_slice",
                )
    else:
        print(
            f"\n[note] --human-ratings-csv not found at {human_csv}; "
            f"skipping 48-scenario slice pass."
        )

    # Optional third pass: α on the 24 curated golden items from a fresh
    # scan of the golden_questions_eval .eval file(s).
    golden_dir = args.golden_eval_dir.expanduser().resolve()
    if golden_dir.is_dir():
        golden_eval_files = sorted(golden_dir.glob("**/*.eval"))
        if golden_eval_files:
            print(
                f"\n--- Golden-24 pass "
                f"({len(golden_eval_files)} .eval file(s) under "
                f"{golden_dir.name}/) ---"
            )
            golden_df, golden_stats = collect_golden_long_table(golden_eval_files)
            print(
                f"  samples scanned:      {golden_stats['total_samples']:,}"
            )
            print(
                f"  samples included:     {golden_stats['samples_included']:,}"
            )
            print(
                f"  excluded (no scores): "
                f"{golden_stats['samples_excluded_no_individual_scores']:,}"
            )
            print(
                f"  excluded (off-scale): "
                f"{golden_stats['samples_excluded_invalid_severity']:,}"
            )
            print(
                f"  excluded (metadata):  "
                f"{golden_stats['samples_excluded_missing_metadata']:,}"
            )
            if golden_df.empty:
                print("[warn] golden long table is empty; skipping.")
            else:
                golden_metrics = compute_metrics(
                    golden_df, n_bootstrap=args.n_bootstrap
                )
                _print_pooled_summary(
                    golden_metrics, label="curated 24 golden set"
                )
                print(f"\nWriting golden-24 outputs to {tables_dir} ...")
                write_outputs(
                    golden_metrics,
                    golden_stats,
                    golden_df,
                    tables_dir,
                    args.n_bootstrap,
                    suffix="_golden_24",
                )
        else:
            print(
                f"\n[note] --golden-eval-dir {golden_dir} contains no .eval "
                f"files; skipping golden-24 pass."
            )
    else:
        print(
            f"\n[note] --golden-eval-dir not found at {golden_dir}; "
            f"skipping golden-24 pass."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
