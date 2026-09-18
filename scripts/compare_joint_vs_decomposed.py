"""Three-arm comparison: joint vs decomposed prompt structure (arm C analysis).

Arms, all scoring the SAME turns:
  A = partner's single judge, joint call (all 8 principles at once) — read from
      the orig_judgments dict the converter embedded in the arm-C log.
  B = 3-judge panel, one call per (turn, principle) — read from the arm-B
      .eval log (per-cell panel MEDIAN of the 3 judges).
  C = 3-judge panel, joint call — read from the arm-C .eval log (per-slug
      panel MEDIAN of the 3 judges' individual_scores).

C↔B isolates prompt structure (decomposed vs joint) with the judge held
constant; C↔A isolates the judge with the call structure held constant.

PRE-COMMITTED PRIMARY READOUTS (plan file, arm-C section — decided before any
results existed):
  1. Co-flag rate: share of turns with >= --coflag-threshold principles
     negative (severity <= -0.5) — the judge-validity audit's
     "multi-principle echo" statistic. Reported per arm with Wilson CIs;
     paired arm differences tested with an exact McNemar (binomial on
     discordant turns; the same turns appear in every arm).
  2. Aggregate inter-principle correlation: mean off-diagonal pairwise
     Spearman rho over the 8x8 principle matrix, per arm (pairwise-complete
     over non-NaN cells). Excess inter-correlation in a joint arm is the
     halo/echo signature.

Everything else is EXPLORATORY and labeled as such in the report:
  - first-eigenvalue share of the Spearman matrix (factor-structure view)
  - per-principle paired severity shift C - B (the decomposition effect at
    severity level, judge held constant)
  - position-bias check: per-cell severity (centered per principle) vs the
    principle's position in arm C's per-turn randomized order
  - full per-pair correlation matrices (CSV)

Usage:
    python scripts/compare_joint_vs_decomposed.py \
        --joint-log <C.eval> --decomposed-log <B.eval> \
        --out-md <results/joint_vs_decomposed.md> \
        [--out-csv-prefix <results/joint_corr>] [--coflag-threshold 4]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
# Shared with the per-principle analyzer so the two report identical CIs and
# agree on what counts as missing.
from compare_panel_vs_single_judge import (  # noqa: E402
    _is_nan_value as _is_nanish,
    load_cells,
    wilson_ci,
)

NEGATIVE_MAX = -0.5  # severity <= this counts as a negative flag


def _iter_samples(log_path: Path):
    with zipfile.ZipFile(log_path) as z:
        names = [n for n in z.namelist()
                 if n.startswith("samples/") and n.endswith(".json")]
        if not names:
            raise SystemExit(f"{log_path}: no samples/ entries -- not an eval log?")
        for name in names:
            with z.open(name) as f:
                yield json.load(f)


def load_joint_arms(log_path: Path) -> tuple[dict, dict, dict, list[str]]:
    """From the arm-C log: (C_scores, A_scores, orders, slugs).

    C_scores[turn][slug] = median of the panel judges' severities (NaN unless
    EVERY judge in the log's ensemble validly scored the slug — the strict
    per-slug rule, with the panel size taken from the log, not hardcoded).
    A_scores[turn][slug] = the partner judge's original severity.
    orders[turn] = the randomized principle order used in C's prompt.

    The slug set is the union over ALL samples (never just the first), repeat
    ids (__rep2/__rep3) are excluded to match the decomposed loader, and turn
    ids are coerced to str for the cross-log equality check.
    """
    raw: list[tuple[str, dict, dict]] = []  # (turn, score, sample_metadata)
    slug_set: set[str] = set()

    for s in _iter_samples(log_path):
        turn = str(s["id"])
        if turn.endswith(("__rep2", "__rep3")):
            continue  # repeat-slice copies belong to the reliability analysis
        sc = (s.get("scores") or {}).get("joint_overseer")
        if sc is None:
            raise SystemExit(
                f"{log_path}: sample {turn} has no joint_overseer score -- "
                "is this really the arm-C joint log?"
            )
        slug_set.update((sc.get("value") or {}).keys())
        raw.append((turn, sc, (s.get("metadata") or {}).get("metadata") or {}))

    if not slug_set:
        raise SystemExit(f"{log_path}: no principle keys in any sample's value")
    slugs = sorted(slug_set)

    c_scores: dict[str, dict[str, float]] = {}
    a_scores: dict[str, dict[str, float]] = {}
    orders: dict[str, list[str]] = {}
    for turn, sc, md in raw:
        smd = sc.get("metadata") or {}
        individual = smd.get("individual_scores") or []
        med: dict[str, float] = {}
        for slug in slugs:
            sevs = [j.get(slug) for j in individual]
            # Strict per-slug ensemble: every judge in the log's panel (however
            # many that is) must have validly scored the slug.
            if not individual or any(_is_nanish(x) for x in sevs):
                med[slug] = math.nan
            else:
                med[slug] = statistics.median(sevs)
        c_scores[turn] = med

        orig = md.get("orig_judgments") or {}
        a_scores[turn] = {
            slug: (orig.get(slug) or {}).get("severity", math.nan)
            for slug in slugs
        }
        orders[turn] = smd.get("principle_order") or []

    return c_scores, a_scores, orders, slugs


def load_decomposed_arm(log_path: Path, slugs: list[str]) -> dict:
    """From the arm-B log: B_scores[turn][slug] = panel median (NaN cells kept
    as NaN). Reuses the per-principle analyzer's loader for the cell parsing."""
    b_scores: dict[str, dict[str, float]] = defaultdict(
        lambda: {slug: math.nan for slug in slugs}
    )
    for c in load_cells(log_path):
        if c.repeat_kind is not None:
            continue
        if c.slug not in slugs:
            continue  # keep the two arms on the identical principle set
        b_scores[str(c.base_id)][c.slug] = (
            math.nan if c.is_nan else c.panel_median
        )
    return dict(b_scores)


# --------------------------------------------------------------------------- #
# Primary 1: co-flag rate
# --------------------------------------------------------------------------- #
def coflag_flags(scores: dict, slugs: list[str], threshold: int) -> dict[str, bool | None]:
    """Per turn: True if >= threshold principles are negative; None if any
    cell is NaN AND the non-NaN negatives alone cannot settle the question."""
    out: dict[str, bool | None] = {}
    for turn, per_slug in scores.items():
        neg = sum(1 for s in slugs if not _is_nanish(per_slug[s]) and per_slug[s] <= NEGATIVE_MAX)
        n_nan = sum(1 for s in slugs if _is_nanish(per_slug[s]))
        if neg >= threshold:
            out[turn] = True
        elif neg + n_nan < threshold:
            out[turn] = False
        else:
            out[turn] = None  # undecidable because of NaN cells
    return out


def mcnemar_exact(flags_x: dict, flags_y: dict) -> dict:
    """Exact McNemar on paired boolean flags (two-sided binomial on the
    discordant pairs). Turns undecidable (None) in either arm are excluded."""
    both = [t for t in flags_x if flags_x[t] is not None and flags_y.get(t) is not None]
    x_only = sum(1 for t in both if flags_x[t] and not flags_y[t])
    y_only = sum(1 for t in both if flags_y[t] and not flags_x[t])
    n_disc = x_only + y_only
    p = binomtest(x_only, n_disc, 0.5).pvalue if n_disc else float("nan")
    return {"n_pairs": len(both), "x_only": x_only, "y_only": y_only, "pvalue": p}


# --------------------------------------------------------------------------- #
# Primary 2: inter-principle correlation
# --------------------------------------------------------------------------- #
def severity_matrix(scores: dict, slugs: list[str], turn_order: list[str]) -> np.ndarray:
    return np.array(
        [[scores[t][slug] for slug in slugs] for t in turn_order], dtype=float
    )


def spearman_offdiag(mat: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Mean off-diagonal pairwise Spearman rho (pairwise-complete rows) and
    the full matrix. Returns (mean_rho, rho_matrix, n_pairs_used)."""
    k = mat.shape[1]
    rho = np.full((k, k), np.nan)
    vals = []
    for i in range(k):
        rho[i, i] = 1.0
        for j in range(i + 1, k):
            ok = ~(np.isnan(mat[:, i]) | np.isnan(mat[:, j]))
            if ok.sum() < 3:
                continue
            # A constant column is a skipped pair (handled below via NaN), not
            # an error worth a console warning on every run.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = spearmanr(mat[ok, i], mat[ok, j]).statistic
            # A constant column (zero variance) yields NaN — skip that pair.
            if not math.isnan(r):
                rho[i, j] = rho[j, i] = r
                vals.append(r)
    mean_rho = float(np.mean(vals)) if vals else float("nan")
    return mean_rho, rho, len(vals)


def first_eigen_share(rho: np.ndarray) -> float:
    """EXPLORATORY: share of variance on the first eigenvalue of the
    correlation matrix (NaN pairs imputed with the mean off-diagonal rho so
    the matrix is complete)."""
    k = rho.shape[0]
    if k == 0:
        return float("nan")
    filled = rho.copy()
    off = filled[~np.eye(k, dtype=bool)]
    fill_value = np.nanmean(off) if off.size and not np.all(np.isnan(off)) else 0.0
    filled[np.isnan(filled)] = fill_value
    eigvals = np.linalg.eigvalsh(filled)
    return float(eigvals[-1] / k)


# --------------------------------------------------------------------------- #
# Exploratory: paired shift and position bias
# --------------------------------------------------------------------------- #
def paired_shift(c_scores: dict, b_scores: dict, slugs: list[str]) -> dict:
    """Per principle: mean (C - B) severity over turns where both are scored,
    plus milder/equal/harsher counts. Judge held constant -> the decomposition
    effect at severity level."""
    out = {}
    for slug in slugs:
        deltas = []
        milder = harsher = equal = 0
        for turn, c_row in c_scores.items():
            b_row = b_scores.get(turn)
            if b_row is None:
                continue
            c, b = c_row[slug], b_row[slug]
            if _is_nanish(c) or _is_nanish(b):
                continue
            deltas.append(c - b)
            if c > b:
                milder += 1
            elif c < b:
                harsher += 1
            else:
                equal += 1
        out[slug] = {
            "n": len(deltas),
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else float("nan"),
            "joint_milder": milder,
            "equal": equal,
            "joint_harsher": harsher,
        }
    return out


def position_bias(c_scores: dict, orders: dict, slugs: list[str]) -> dict:
    """EXPLORATORY: per-cell severity centered per principle (removing
    principle identity), bucketed by the principle's position in that turn's
    randomized joint prompt. A trend across positions = criterion-order bias."""
    per_slug_vals: dict[str, list[float]] = defaultdict(list)
    for row in c_scores.values():
        for slug in slugs:
            if not _is_nanish(row[slug]):
                per_slug_vals[slug].append(row[slug])
    slug_mean = {
        s: (sum(v) / len(v)) if v else float("nan")
        for s, v in per_slug_vals.items()
    }

    by_pos: dict[int, list[float]] = defaultdict(list)
    xs, ys = [], []
    for turn, row in c_scores.items():
        order = orders.get(turn) or []
        for pos, slug in enumerate(order, start=1):
            v = row.get(slug)
            if slug in slug_mean and not _is_nanish(v):
                centered = v - slug_mean[slug]
                by_pos[pos].append(centered)
                xs.append(pos)
                ys.append(centered)

    trend = spearmanr(xs, ys) if len(xs) >= 3 else None
    return {
        "mean_centered_by_position": {
            pos: (sum(v) / len(v), len(v)) for pos, v in sorted(by_pos.items())
        },
        "spearman_position_vs_centered": (
            {"rho": float(trend.statistic), "p": float(trend.pvalue)}
            if trend else None
        ),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x) -> str:
    return "n/a" if _is_nanish(x) else f"{100*x:.1f}%"


def _ci(ci) -> str:
    lo, hi = ci
    return "n/a" if math.isnan(lo) else f"[{100*lo:.1f}%, {100*hi:.1f}%]"


def _p(p) -> str:
    return "n/a" if _is_nanish(p) else f"{p:.4g}"


def build_report(args, slugs, arms, coflags, mcnemars, corr, corr_all, eigen,
                 shifts, posbias, n_turns, n_common) -> str:
    k = len(slugs)
    n_pairs_max = k * (k - 1) // 2
    L = []
    A = L.append
    A("# Joint vs. decomposed prompt structure (three-arm comparison)")
    A("")
    A(f"> Same {n_turns} production turns in every arm. A = partner judge + "
      "joint call; B = panel + per-principle calls; C = panel + joint call. "
      "C↔B isolates prompt structure; C↔A isolates the judge. Primaries were "
      "pre-committed before results existed; everything else is exploratory.")
    A("")
    A("## Provenance")
    A(f"- Joint (C) log: `{args.joint_log}`")
    A(f"- Decomposed (B) log: `{args.decomposed_log}`")
    A(f"- Turns: {n_turns}; negative flag = severity <= {NEGATIVE_MAX}; "
      f"co-flag threshold = {args.coflag_threshold} of {k} principles")
    A("")

    A(f"## Primary 1 — co-flag rate (turns with >= {args.coflag_threshold} "
      "principles negative)")
    A("")
    A("| arm | co-flagged | rate | 95% CI |")
    A("|---|--:|--:|---|")
    undecidable_notes = []
    for arm in ("A", "B", "C"):
        fl = coflags[arm]
        decided = [v for v in fl.values() if v is not None]
        n_flag = sum(decided)
        A(f"| {arm} ({arms[arm]}) | {n_flag}/{len(decided)} | "
          f"{_pct(n_flag/len(decided) if decided else float('nan'))} | "
          f"{_ci(wilson_ci(n_flag, len(decided)))} |")
        n_und = sum(1 for v in fl.values() if v is None)
        if n_und:
            undecidable_notes.append(f"{arm}: {n_und}")
    A("")
    if undecidable_notes:
        A("- NaN-undecidable turns excluded from the rates above "
          f"({', '.join(undecidable_notes)}); the paired McNemar below already "
          "restricts to turns decidable in both arms.")
        A("")
    A("Paired contrasts (exact McNemar on discordant turns):")
    A("")
    A("| contrast | isolates | discordant (x-only / y-only) | p (two-sided) |")
    A("|---|---|---|--:|")
    for (x, y, isolates), m in mcnemars.items():
        A(f"| {x} vs {y} | {isolates} | {m['x_only']} / {m['y_only']} "
          f"(n={m['n_pairs']}) | {_p(m['pvalue'])} |")
    A("")

    A("## Primary 2 — mean off-diagonal inter-principle Spearman rho")
    A("")
    A("Excess inter-correlation in a joint arm is the halo / multi-principle "
      f"echo signature (one holistic impression bleeding into all {k} grades).")
    A("")
    A(f"Computed over the {n_common}/{n_turns} common complete-case turns (no "
      "NaN in any arm), so every arm's rho uses the identical row set and the "
      "cross-arm comparison cannot reflect differential NaN survival. The "
      "all-rows figure is a sensitivity check.")
    A("")
    A("| arm | rho (common rows) | rho (all rows, sens.) | pairs used | "
      "first-eigenvalue share (expl.) |")
    A("|---|--:|--:|--:|--:|")
    for arm in ("A", "B", "C"):
        mean_rho, _, n_pairs = corr[arm]
        A(f"| {arm} ({arms[arm]}) | {mean_rho:+.3f} | {corr_all[arm]:+.3f} | "
          f"{n_pairs}/{n_pairs_max} | {_pct(eigen[arm])} |")
    A("")
    A("- Note: rho is computed over this stratified subset (negative-enriched), "
      "not the raw production distribution; compare arms, not absolute levels.")
    A("")

    A("## Exploratory — decomposition effect at severity level (C − B, judge "
      "held constant)")
    A("")
    A("| principle | n | mean Δ (C−B) | joint milder / equal / harsher |")
    A("|---|--:|--:|---|")
    for slug in slugs:
        s = shifts[slug]
        A(f"| {slug} | {s['n']} | {s['mean_delta']:+.3f} | "
          f"{s['joint_milder']} / {s['equal']} / {s['joint_harsher']} |")
    A("")

    A("## Exploratory — position bias in the joint prompt (arm C)")
    A("")
    A("Severity centered per principle, bucketed by the principle's position "
      "in the turn's randomized order (order was randomized per turn and "
      "recorded precisely to enable this check).")
    A("")
    A("| position | mean centered severity | n |")
    A("|--:|--:|--:|")
    for pos, (m, n) in posbias["mean_centered_by_position"].items():
        A(f"| {pos} | {m:+.4f} | {n} |")
    tr = posbias["spearman_position_vs_centered"]
    if tr:
        A("")
        A(f"- Spearman(position, centered severity): rho={tr['rho']:+.4f}, "
          f"p={_p(tr['p'])}")
    A("")
    A("---")
    A("_Primaries were pre-committed (plan file, arm-C design gate). All "
      "correlations are Spearman over pairwise-complete non-NaN cells. The "
      "same turns appear in every arm; paired tests exploit that._")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--joint-log", required=True, type=Path)
    parser.add_argument("--decomposed-log", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-csv-prefix", type=Path,
                        help="write per-arm 8x8 Spearman matrices as CSVs")
    parser.add_argument("--coflag-threshold", type=int, default=4)
    args = parser.parse_args()

    c_scores, a_scores, orders, slugs = load_joint_arms(args.joint_log)
    b_scores = load_decomposed_arm(args.decomposed_log, slugs)

    turns_c = set(c_scores)
    turns_b = set(b_scores)
    if turns_c != turns_b:
        only_c, only_b = turns_c - turns_b, turns_b - turns_c
        raise SystemExit(
            "turn sets differ between logs (arms must score the SAME turns): "
            f"{len(only_c)} only in joint, {len(only_b)} only in decomposed"
        )
    turn_order = sorted(turns_c)

    arms = {"A": "partner judge, joint", "B": "panel, per-principle",
            "C": "panel, joint"}
    scores = {"A": a_scores, "B": b_scores, "C": c_scores}

    coflags = {
        arm: coflag_flags(scores[arm], slugs, args.coflag_threshold)
        for arm in arms
    }
    mcnemars = {
        ("C", "B", "prompt structure"): mcnemar_exact(coflags["C"], coflags["B"]),
        ("C", "A", "judge"): mcnemar_exact(coflags["C"], coflags["A"]),
        ("B", "A", "judge + structure"): mcnemar_exact(coflags["B"], coflags["A"]),
    }

    # Primary 2 is computed over the COMMON complete-case turn set (rows with
    # no NaN in ANY arm), so every arm's rho uses the identical row set and the
    # cross-arm comparison cannot be confounded by which turns survived each
    # arm's NaN rule. The all-rows figure is kept as a sensitivity check.
    mats = {arm: severity_matrix(scores[arm], slugs, turn_order) for arm in arms}
    common_ok = np.ones(len(turn_order), dtype=bool)
    for mat in mats.values():
        common_ok &= ~np.isnan(mat).any(axis=1)
    n_common = int(common_ok.sum())

    corr = {}
    corr_all = {}
    eigen = {}
    for arm in arms:
        mean_rho, rho, n_pairs = spearman_offdiag(mats[arm][common_ok])
        corr[arm] = (mean_rho, rho, n_pairs)
        eigen[arm] = first_eigen_share(rho)
        corr_all[arm] = spearman_offdiag(mats[arm])[0]
        if args.out_csv_prefix:
            out = args.out_csv_prefix.parent / (
                f"{args.out_csv_prefix.name}_{arm}.csv"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                f.write("," + ",".join(slugs) + "\n")
                for i, slug in enumerate(slugs):
                    f.write(slug + "," + ",".join(
                        "" if math.isnan(rho[i, j]) else f"{rho[i, j]:.4f}"
                        for j in range(len(slugs))
                    ) + "\n")

    shifts = paired_shift(c_scores, b_scores, slugs)
    posbias = position_bias(c_scores, orders, slugs)

    report = build_report(args, slugs, arms, coflags, mcnemars, corr, corr_all,
                          eigen, shifts, posbias, len(turn_order), n_common)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report)
    print(f"Wrote {args.out_md}")

    for arm in ("A", "B", "C"):
        fl = [v for v in coflags[arm].values() if v is not None]
        mean_rho, _, _ = corr[arm]
        pct = f"{100*sum(fl)/len(fl):.1f}%" if fl else "n/a (no decidable turns)"
        print(f"arm {arm}: co-flag {sum(fl)}/{len(fl)} ({pct}), "
              f"mean off-diag rho {mean_rho:+.3f} over {n_common} common turns")


if __name__ == "__main__":
    main()
