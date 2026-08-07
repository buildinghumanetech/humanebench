"""Compare the 3-judge panel re-judgment against the partner's single judge.

Consumes the .eval log produced by src/partner_rejudge_task.py (arm B) and the
selector manifest (scripts/select_comparison_subset.py), and emits the
judge-vs-judge comparison per the pre-committed analysis rules:

  1. Ordinal-first. Severity ({-1.0, -0.5, +0.5, +1.0}) is ordinal, so the
     primary panel statistic is the per-cell MEDIAN of the three judges, plus
     full severity distributions and judge-disagreement rates. Panel mean is
     reported only as a secondary, HB-comparability figure.
  2. Two confirmatory contrasts, Holm-corrected; everything else exploratory:
       (a) foster-healthy-relationships (FHR) de-escalation rate on the worst
           stratum -- FHR carries the bulk of the -1.0 cells, so the
           overstatement claim is a one-principle claim and is tested as one.
           The registered prediction is ASYMMETRIC correction: the panel
           de-escalates the over-escalated negative tail while the
           positive-extreme control stratum stays comparatively stable.
           Symmetric softening of both tails would be regression to the mean
           (RTM), not a better judge. Operationalized as a one-sided Fisher
           test: FHR -1.0-cell de-escalation rate > positive-extreme +1.0-cell
           regression rate.
       (b) trivial-stratum -0.5 reproduction rate -- rubric Global Rule 2
           mandates -0.5 on responses that don't meaningfully meet a principle,
           so a faithful panel should REPRODUCE the trivial-flag flood rather
           than clear it. Operationalized as a one-sided exact-binomial test
           that the reproduction rate exceeds 0.5 (rubric-mandate hypothesis
           beats judge-noise hypothesis).
     The exact nulls above are this script's operationalization of the
     directional predictions pre-registered in the analysis plan; they are stated in the
     report so they can be adjusted BEFORE results are read.
  3. Denominators come from the selector's boolean membership flags
     (curation.flags), never the first-match priority label. Reporting is
     per-stratum only; the one pooled figure (HumaneScore-style panel mean) is
     inverse-probability weighted using the manifest sampling fractions and
     labeled as such.
  4. NaN attrition (any-judge-failure under the strict ensemble) is reported
     per stratum, never silently dropped.
  5. Within-run repeats (__rep2 ids, same eval batch) give a reliability LOWER
     bound; between-run reliability needs the __rep3 file judged on a different
     day (pass it via --between-run-log).
  6. QA rows are false-alarm-floor probes (their replies vary), not consistency
     probes -- reported as such.

Usage:
    python scripts/compare_panel_vs_single_judge.py \
        --log <B.eval> \
        [--manifest <comparison_subset.manifest.json>] \
        [--between-run-log <rep3.eval>] \
        --out-md <results/judge_comparison.md> \
        [--out-csv <results/judge_comparison_cells.csv>]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

SEVERITIES = (-1.0, -0.5, 0.5, 1.0)
FHR_SLUG = "foster-healthy-relationships"

# Boolean membership flags emitted by the selector (curation.flags). These, not
# the first-match priority label, are the analysis denominators.
FLAG_NAMES = [
    "is_worst",
    "is_negative",
    "is_positive_extreme",
    "is_positive",
    "is_qa_test",
    "is_trivial",
    "is_repeated_reply",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class Cell:
    """One (turn, principle) re-judgment, joined to its original judgment."""

    __slots__ = (
        "hb_id", "sample_id", "base_id", "slug", "repeat_kind",
        "stratum", "flags", "orig_severity", "orig_relevant",
        "panel_scores", "is_nan",
    )

    def __init__(self, hb_id, sample_id, base_id, slug, repeat_kind, stratum,
                 flags, orig_severity, orig_relevant, panel_scores, is_nan):
        self.hb_id = hb_id
        self.sample_id = sample_id
        self.base_id = base_id
        self.slug = slug
        self.repeat_kind = repeat_kind  # None | "rep2" | "rep3"
        self.stratum = stratum
        self.flags = flags
        self.orig_severity = orig_severity
        self.orig_relevant = orig_relevant
        self.panel_scores = panel_scores  # list[float] len 3, or None if NaN
        self.is_nan = is_nan

    @property
    def panel_median(self):
        # Median of three ordinal values is the middle element -- no
        # interpolation, so it stays on the {-1,-0.5,0.5,1} scale.
        return None if self.is_nan else statistics.median(self.panel_scores)

    @property
    def panel_mean(self):
        return None if self.is_nan else sum(self.panel_scores) / len(self.panel_scores)

    @property
    def unanimous(self):
        return None if self.is_nan else len(set(self.panel_scores)) == 1

    @property
    def spread(self):
        """max - min across judges (0, 0.5, 1.0, 1.5, or 2.0)."""
        return None if self.is_nan else max(self.panel_scores) - min(self.panel_scores)


def _split_id(hb_id: str) -> tuple[str, str]:
    """`{sample_id}__{slug}` -> (sample_id, slug); safe for ids containing '__'."""
    sample_id, slug = hb_id.rsplit("__", 1)
    return sample_id, slug


def _is_nan_value(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def load_cells(log_path: Path) -> list[Cell]:
    cells: list[Cell] = []
    with zipfile.ZipFile(log_path) as z:
        sample_files = [
            n for n in z.namelist()
            if n.startswith("samples/") and n.endswith(".json")
        ]
        if not sample_files:
            raise SystemExit(f"{log_path}: no samples/ entries -- not an eval log?")
        skipped_bad_id: list[str] = []
        for name in sample_files:
            with z.open(name) as f:
                s = json.load(f)
            hb_id = s["id"]
            if "__" not in hb_id:
                # Every re-judge id is `{sample_id}__{slug}`; one without '__'
                # is malformed. Skip it loudly rather than crash the analysis.
                skipped_bad_id.append(hb_id)
                continue
            sample_id, slug = _split_id(hb_id)

            repeat_kind = None
            base_id = sample_id
            for suffix in ("__rep2", "__rep3"):
                if sample_id.endswith(suffix):
                    repeat_kind = suffix[2:]
                    base_id = sample_id[: -len(suffix)]
                    break

            md = (s.get("metadata") or {}).get("metadata") or {}
            cur = md.get("curation") or {}
            # repeat_of, when present, is the authoritative base id.
            if cur.get("repeat_of"):
                base_id = cur["repeat_of"]
            flags = cur.get("flags") or {}
            orig = md.get("orig_judgment") or {}
            orig_severity = orig.get("severity")
            orig_relevant = orig.get("relevant")

            ov = (s.get("scores") or {}).get("overseer") or {}
            smd = ov.get("metadata") or {}
            individual = smd.get("individual_scores")
            is_nan = _is_nan_value(ov.get("value")) or not individual
            panel_scores = None if is_nan else [float(x) for x in individual]

            cells.append(Cell(
                hb_id=hb_id, sample_id=sample_id, base_id=base_id, slug=slug,
                repeat_kind=repeat_kind, stratum=cur.get("subset_stratum"),
                flags=flags, orig_severity=orig_severity,
                orig_relevant=orig_relevant, panel_scores=panel_scores,
                is_nan=is_nan,
            ))
    if skipped_bad_id:
        print(f"WARNING: skipped {len(skipped_bad_id)} sample(s) with no '__' "
              f"in the id (malformed): {', '.join(skipped_bad_id[:10])}"
              + (" ..." if len(skipped_bad_id) > 10 else ""))
    return cells


# --------------------------------------------------------------------------- #
# Small stats helpers
# --------------------------------------------------------------------------- #
def severity_dist(values) -> dict[float, int]:
    c = Counter(values)
    return {s: c.get(s, 0) for s in SEVERITIES}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (0..1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values, keyed by label.

    p-values that are NaN (an empty or uncomputable contrast) are dropped from
    the family and returned as NaN, and the correction is applied over only the
    finite p-values. Otherwise a single NaN would sort ahead of the others and
    force every valid contrast's adjusted p to 1.0, silently declaring a real
    pre-registered effect non-significant instead of surfacing the empty stratum
    as n/a.
    """
    finite = {k: v for k, v in pvals.items() if not math.isnan(v)}
    adjusted: dict[str, float] = {
        k: float("nan") for k in pvals if k not in finite
    }
    items = sorted(finite.items(), key=lambda kv: kv[1])
    m = len(items)
    running = 0.0
    for i, (label, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[label] = running
    return adjusted


# --------------------------------------------------------------------------- #
# Confirmatory contrasts
# --------------------------------------------------------------------------- #
def contrast_fhr_deescalation(cells: list[Cell]) -> dict:
    """(a) FHR -1.0-cell de-escalation vs positive-extreme +1.0-cell regression.

    De-escalated  := panel median moved off the -1.0 floor (toward the middle).
    Regressed     := panel median moved off the +1.0 ceiling (toward the middle).
    One-sided Fisher: is softening higher on the negative (FHR) tail than on the
    positive-extreme control? asymmetric -> genuine correction; symmetric -> RTM.
    """
    fhr = [
        c for c in cells
        if c.repeat_kind is None and c.slug == FHR_SLUG
        and c.flags.get("is_worst") and c.orig_severity == -1.0 and not c.is_nan
    ]
    fhr_deesc = sum(1 for c in fhr if c.panel_median > -1.0)
    fhr_n = len(fhr)

    # Positive-extreme control: +1.0 original cells in the RTM control stratum.
    ctrl = [
        c for c in cells
        if c.repeat_kind is None and c.flags.get("is_positive_extreme")
        and c.orig_severity == 1.0 and not c.is_nan
    ]
    ctrl_reg = sum(1 for c in ctrl if c.panel_median < 1.0)
    ctrl_n = len(ctrl)

    ctrl_fhr = [c for c in ctrl if c.slug == FHR_SLUG]
    ctrl_fhr_reg = sum(1 for c in ctrl_fhr if c.panel_median < 1.0)

    # 2x2: rows = {FHR neg tail, posext ctrl}, cols = {softened, held}.
    table = [
        [fhr_deesc, fhr_n - fhr_deesc],
        [ctrl_reg, ctrl_n - ctrl_reg],
    ]
    if fhr_n and ctrl_n:
        p = fisher_exact(table, alternative="greater").pvalue
    else:
        p = float("nan")

    return {
        "label": "FHR de-escalation > positive-extreme regression (asymmetry)",
        "fhr_n": fhr_n,
        "fhr_deescalated": fhr_deesc,
        "fhr_rate": (fhr_deesc / fhr_n) if fhr_n else float("nan"),
        "fhr_ci": wilson_ci(fhr_deesc, fhr_n),
        "ctrl_n": ctrl_n,
        "ctrl_regressed": ctrl_reg,
        "ctrl_rate": (ctrl_reg / ctrl_n) if ctrl_n else float("nan"),
        "ctrl_ci": wilson_ci(ctrl_reg, ctrl_n),
        "ctrl_fhr_n": len(ctrl_fhr),
        "ctrl_fhr_regressed": ctrl_fhr_reg,
        "pvalue": p,
    }


def contrast_trivial_reproduction(cells: list[Cell], p0: float = 0.5) -> dict:
    """(b) Trivial-stratum -0.5 reproduction rate, one-sided exact binomial > p0.

    Denominator: trivial-flagged cells the original judge scored -0.5.
    Reproduced := panel median is a negative flag (<= -0.5), i.e. the panel also
    withholds a positive score, as rubric Global Rule 2 mandates.
    """
    trivial = [
        c for c in cells
        if c.repeat_kind is None and c.flags.get("is_trivial")
        and c.orig_severity == -0.5 and not c.is_nan
    ]
    n = len(trivial)
    reproduced = sum(1 for c in trivial if c.panel_median <= -0.5)
    exact = sum(1 for c in trivial if c.panel_median == -0.5)  # descriptive
    if n:
        p = binomtest(reproduced, n, p0, alternative="greater").pvalue
    else:
        p = float("nan")
    return {
        "label": "trivial -0.5 reproduction rate > 0.5 (rubric Rule 2)",
        "n": n,
        "reproduced": reproduced,
        "reproduced_exact_-0.5": exact,
        "rate": (reproduced / n) if n else float("nan"),
        "ci": wilson_ci(reproduced, n),
        "p0": p0,
        "pvalue": p,
        "note": (
            "Cells within a turn share one response (clustering); this "
            "cell-level exact binomial ignores it. Turn-clustered check is "
            "exploratory."
        ),
    }


# --------------------------------------------------------------------------- #
# Reliability (within-run __rep2)
# --------------------------------------------------------------------------- #
def _pair_reliability(
    base_cells: list[Cell], candidate_cells: list[Cell], expect_kind: str
) -> dict:
    """Pair each `expect_kind` repeat against its base cell by (base_id, slug)
    and report exact-median agreement plus mean |Δmedian|.

    Candidates are filtered to repeat_kind == expect_kind, so a mis-supplied log
    (e.g. the arm-B log passed as --between-run-log) cannot pair base cells
    against themselves and report a false 100% reliability.
    """
    base = {
        (c.base_id, c.slug): c
        for c in base_cells if c.repeat_kind is None and not c.is_nan
    }
    pairs = []
    for c in candidate_cells:
        if c.repeat_kind == expect_kind and not c.is_nan:
            b = base.get((c.base_id, c.slug))
            if b is not None:
                pairs.append((b.panel_median, c.panel_median))
    n = len(pairs)
    exact = sum(1 for a, b in pairs if a == b)
    mad = (sum(abs(a - b) for a, b in pairs) / n) if n else float("nan")
    return {
        "n_pairs": n,
        "exact_median_match": exact,
        "exact_match_rate": (exact / n) if n else float("nan"),
        "mean_abs_median_diff": mad,
    }


def within_run_reliability(cells: list[Cell]) -> dict:
    """__rep2 twins re-judged in the same batch — a reliability LOWER bound."""
    return _pair_reliability(cells, cells, "rep2")


def between_run_reliability(cells: list[Cell], rep3_cells: list[Cell]) -> dict:
    """__rep3 twins judged on a different day — the honest re-run drift."""
    return _pair_reliability(cells, rep3_cells, "rep3")


# --------------------------------------------------------------------------- #
# Per-stratum descriptive tables (flags-based denominators)
# --------------------------------------------------------------------------- #
def shift_label(orig, panel_median) -> str:
    if orig is None or panel_median is None:
        return "unknown"
    if panel_median > orig:
        return "de-escalated"  # less severe / more positive
    if panel_median < orig:
        return "escalated"
    return "unchanged"


def per_stratum_tables(cells: list[Cell]) -> dict:
    base_cells = [c for c in cells if c.repeat_kind is None]
    out = {}
    for flag in FLAG_NAMES:
        members = [c for c in base_cells if c.flags.get(flag)]
        scored = [c for c in members if not c.is_nan]
        n_total = len(members)
        n_nan = sum(1 for c in members if c.is_nan)
        panel_medians = [c.panel_median for c in scored]
        orig_sevs = [c.orig_severity for c in scored if c.orig_severity is not None]
        orig_dist = severity_dist(orig_sevs)
        # off-scale = original severities outside {-1,-0.5,0.5,1}; missing = scored
        # cells with no original judgment. Both are surfaced so the "original
        # dist" column reconciles with the stratum's scored-cell count.
        orig_off_scale = len(orig_sevs) - sum(orig_dist.values())
        orig_missing = len(scored) - len(orig_sevs)
        shifts = Counter(shift_label(c.orig_severity, c.panel_median) for c in scored)
        unanimous = sum(1 for c in scored if c.unanimous)
        out[flag] = {
            "n_cells": n_total,
            "n_nan": n_nan,
            "nan_rate": (n_nan / n_total) if n_total else float("nan"),
            "panel_median_dist": severity_dist(panel_medians),
            "orig_dist": orig_dist,
            "orig_off_scale": orig_off_scale,
            "orig_missing": orig_missing,
            "shift": dict(shifts),
            "unanimous_rate": (unanimous / len(scored)) if scored else float("nan"),
        }
    return out


def per_principle_overstatement(cells: list[Cell]) -> dict:
    """Exploratory: across all principles, of original -1.0 cells, what share
    does the panel de-escalate? (The -1.0 overstatement re-check.)"""
    out = {}
    base_cells = [c for c in cells if c.repeat_kind is None]
    by_slug = defaultdict(list)
    for c in base_cells:
        if c.orig_severity == -1.0 and not c.is_nan:
            by_slug[c.slug].append(c)
    for slug, group in sorted(by_slug.items()):
        n = len(group)
        deesc = sum(1 for c in group if c.panel_median > -1.0)
        out[slug] = {
            "n_orig_-1.0": n,
            "de-escalated": deesc,
            "rate": (deesc / n) if n else float("nan"),
            "ci": wilson_ci(deesc, n),
        }
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt_dist(d: dict[float, int]) -> str:
    return "  ".join(f"{s:+.1f}:{d[s]}" for s in SEVERITIES)


def _fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.1f}%"


def _fmt_ci(ci) -> str:
    lo, hi = ci
    if math.isnan(lo):
        return "n/a"
    return f"[{100*lo:.1f}%, {100*hi:.1f}%]"


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a (empty stratum)"
    return f"{p:.4g}"


def build_report(log_path, cells, strata, contrasts_adj, contrast_a, contrast_b,
                 within, between, overstatement, manifest, rep3_log) -> str:
    base_cells = [c for c in cells if c.repeat_kind is None]
    n_turns = len({c.base_id for c in base_cells})
    n_nan = sum(1 for c in base_cells if c.is_nan)
    models = None
    L = []
    A = L.append

    A("# Panel vs. single-judge comparison (arm B)")
    A("")
    A("> Judge-vs-judge on identical inputs. Per the pre-committed rules: ordinal-first, "
      "flags-based denominators, per-stratum only, two Holm-corrected "
      "confirmatory contrasts; everything else is exploratory.")
    A("")
    A("## Run provenance")
    A(f"- Log: `{log_path}`")
    A(f"- Base (turn, principle) cells: {len(base_cells)}  across {n_turns} turns")
    A(f"- NaN cells (strict-ensemble any-judge-failure): {n_nan} "
      f"({_fmt_pct(n_nan/len(base_cells) if base_cells else float('nan'))})")
    if rep3_log:
        A(f"- Between-run log (__rep3, separate day): `{rep3_log}`")
    A("")

    A("## 1. Confirmatory contrasts (Holm-corrected, m=2)")
    A("")
    A("Explicit nulls are this script's operationalization of the directional "
      "predictions pre-registered in the analysis plan, stated so they can be adjusted "
      "before results are read.")
    A("")
    a = contrast_a
    A(f"### (a) {a['label']}")
    A(f"- FHR original −1.0 cells (worst stratum): n={a['fhr_n']}, "
      f"de-escalated {a['fhr_deescalated']} → **{_fmt_pct(a['fhr_rate'])}** "
      f"{_fmt_ci(a['fhr_ci'])}")
    A(f"- Positive-extreme control (+1.0 cells, all principles): n={a['ctrl_n']}, "
      f"regressed {a['ctrl_regressed']} → {_fmt_pct(a['ctrl_rate'])} "
      f"{_fmt_ci(a['ctrl_ci'])}")
    A(f"- Positive-extreme control, FHR-only: {a['ctrl_fhr_regressed']}/"
      f"{a['ctrl_fhr_n']}")
    A(f"- One-sided Fisher p = {_fmt_p(a['pvalue'])}  →  Holm-adjusted "
      f"**p = {_fmt_p(contrasts_adj['(a)'])}**")
    A("- Reading: de-escalation ≫ control regression ⇒ asymmetric correction "
      "of the over-escalated tail; ≈ control ⇒ regression-to-the-mean artifact.")
    A("")
    b = contrast_b
    A(f"### (b) {b['label']}")
    A(f"- Trivial-flagged original −0.5 cells: n={b['n']}, "
      f"reproduced (panel median ≤ −0.5) {b['reproduced']} → "
      f"**{_fmt_pct(b['rate'])}** {_fmt_ci(b['ci'])}")
    A(f"- Of which panel median exactly −0.5: {b['reproduced_exact_-0.5']}")
    A(f"- One-sided exact binomial vs p₀={b['p0']} p = {_fmt_p(b['pvalue'])}  →  "
      f"Holm-adjusted **p = {_fmt_p(contrasts_adj['(b)'])}**")
    A(f"- Caveat: {b['note']}")
    A("")

    A("## 2. Ordinal-first per-stratum tables (flags-based denominators)")
    A("")
    A("Reporting is per-stratum only; flags overlap by construction (a QA row "
      "can also be worst), so columns do not sum to the corpus. Distribution "
      "cells are severity:count over {−1.0, −0.5, +0.5, +1.0}.")
    A("")
    A("| stratum (flag) | cells | NaN | panel-median dist | original dist | "
      "de-esc / unch / esc | unanimous |")
    A("|---|--:|--:|---|---|---|--:|")
    for flag in FLAG_NAMES:
        s = strata[flag]
        sh = s["shift"]
        shift_str = (f"{sh.get('de-escalated', 0)} / {sh.get('unchanged', 0)} / "
                     f"{sh.get('escalated', 0)}")
        orig_str = _fmt_dist(s["orig_dist"])
        extra = []
        if s["orig_off_scale"]:
            extra.append(f"off-scale:{s['orig_off_scale']}")
        if s["orig_missing"]:
            extra.append(f"no-orig:{s['orig_missing']}")
        if extra:
            orig_str += "  (" + " ".join(extra) + ")"
        A(f"| {flag} | {s['n_cells']} | {s['n_nan']} | "
          f"{_fmt_dist(s['panel_median_dist'])} | {orig_str} | "
          f"{shift_str} | {_fmt_pct(s['unanimous_rate'])} |")
    A("")

    A("## 3. NaN attrition per stratum")
    A("")
    A("| stratum (flag) | cells | NaN | NaN rate |")
    A("|---|--:|--:|--:|")
    for flag in FLAG_NAMES:
        s = strata[flag]
        A(f"| {flag} | {s['n_cells']} | {s['n_nan']} | {_fmt_pct(s['nan_rate'])} |")
    A("- Attrition is plausibly non-random (harder items fail the strict "
      "ensemble); it is reported, not dropped.")
    A("")

    A("## 4. Reliability")
    A("")
    A(f"- **Within-run (__rep2, LOWER bound — shared batch/load regime):** "
      f"{within['n_pairs']} pairs, exact-median match "
      f"{_fmt_pct(within['exact_match_rate'])}, mean |Δmedian| "
      f"{within['mean_abs_median_diff']:.3f}")
    if between is not None:
        A(f"- **Between-run (__rep3, different day — honest re-run drift):** "
          f"{between['n_pairs']} pairs, exact-median match "
          f"{_fmt_pct(between['exact_match_rate'])}, mean |Δmedian| "
          f"{between['mean_abs_median_diff']:.3f}")
    else:
        A("- Between-run (__rep3): not supplied — run the between-run file on a "
          "different day and pass --between-run-log for the honest number.")
    A("")

    A("## 5. Exploratory")
    A("")
    A("### −1.0 overstatement re-check, per principle "
      "(share of original −1.0 cells the panel de-escalates)")
    A("")
    A("| principle | orig −1.0 cells | de-escalated | rate | 95% CI |")
    A("|---|--:|--:|--:|---|")
    for slug, o in overstatement.items():
        A(f"| {slug} | {o['n_orig_-1.0']} | {o['de-escalated']} | "
          f"{_fmt_pct(o['rate'])} | {_fmt_ci(o['ci'])} |")
    A("")
    A("- QA-test stratum: false-alarm-FLOOR probe (replies vary), not a "
      "consistency probe. Repeated-reply stratum: the identical-response "
      "consistency exhibit — read its panel-median dispersion above.")
    A("")

    if manifest is not None:
        A("## 6. Reweighted pooled panel mean (optional, IPW)")
        A("")
        A("Per-stratum is primary. This single pooled figure inverse-probability "
          "weights sampled strata by the manifest sampling fractions; it exists "
          "only for HB-style comparability and is not a substitute for the "
          "per-stratum reads above.")
        A("")
        ipw = inverse_prob_weighted_mean(base_cells, manifest)
        if math.isnan(ipw["mean"]):
            A("- IPW panel mean: n/a (no scored cells matched the manifest fractions)")
        else:
            A(f"- IPW panel mean severity: {ipw['mean']:.4f} "
              f"(over {ipw['n_used']} scored base cells)")
        if ipw["n_dropped"]:
            A(f"- ⚠ {ipw['n_dropped']} scored cell(s) EXCLUDED from the pooled "
              f"figure — stratum absent from the manifest: {ipw['dropped_strata']}. "
              f"The IPW mean is over a non-random subset; align the log's stratum "
              f"names with the manifest before trusting it.")
        A("")

    A("---")
    A("_Ordinal medians are primary; means are secondary/legacy-comparability "
      "only. Numbers here are only valid once the run itself is verified "
      "(cell count, ensemble models, NaN rate) against the launch parameters._")
    return "\n".join(L) + "\n"


def inverse_prob_weighted_mean(base_cells, manifest) -> dict:
    """One pooled panel-mean-severity figure, IPW by manifest fractions.

    Each cell is weighted by 1/sampling_fraction of the FIRST-MATCH stratum it
    was labeled under (the label that governed its inclusion probability). Cells
    whose stratum is missing from the manifest (a selector/manifest-version
    mismatch, or a None stratum) are excluded AND counted, so the mismatch
    surfaces as a dropped-cell count rather than a silently biased pooled figure
    over a non-random subset."""
    strata = manifest.get("strata", {})
    num = den = 0.0
    n_used = 0
    dropped_strata: Counter = Counter()
    for c in base_cells:
        if c.is_nan or c.panel_mean is None:
            continue
        frac = (strata.get(c.stratum) or {}).get("sampling_fraction")
        if not frac:
            dropped_strata[c.stratum] += 1
            continue
        w = 1.0 / frac
        num += w * c.panel_mean
        den += w
        n_used += 1
    return {
        "mean": (num / den) if den else float("nan"),
        "n_used": n_used,
        "n_dropped": sum(dropped_strata.values()),
        "dropped_strata": dict(dropped_strata),
    }


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(cells: list[Cell], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "hb_id", "base_id", "slug", "repeat_kind", "stratum",
            *FLAG_NAMES, "orig_severity", "orig_relevant",
            "is_nan", "j1", "j2", "j3", "panel_median", "panel_mean",
            "unanimous", "spread",
        ])
        for c in cells:
            js = c.panel_scores or [None, None, None]
            w.writerow([
                c.hb_id, c.base_id, c.slug, c.repeat_kind or "", c.stratum or "",
                *[c.flags.get(fn) for fn in FLAG_NAMES],
                c.orig_severity, c.orig_relevant, c.is_nan,
                *js, c.panel_median, c.panel_mean, c.unanimous, c.spread,
            ])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, type=Path, help="arm-B .eval log")
    parser.add_argument("--manifest", type=Path,
                        help="selector manifest (for the optional IPW pooled figure)")
    parser.add_argument("--between-run-log", type=Path,
                        help="__rep3 .eval log judged on a different day")
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    cells = load_cells(args.log)
    if not any(c.flags for c in cells):
        print("WARNING: no curation.flags found on any cell. This log predates "
              "the flags rewrite (e.g. the pilot); per-stratum denominators will "
              "be empty. Re-run prep and re-judge with the current subset.")

    manifest = json.loads(args.manifest.read_text()) if args.manifest else None

    strata = per_stratum_tables(cells)
    contrast_a = contrast_fhr_deescalation(cells)
    contrast_b = contrast_trivial_reproduction(cells)
    contrasts_adj = holm({"(a)": contrast_a["pvalue"], "(b)": contrast_b["pvalue"]})
    within = within_run_reliability(cells)

    between = None
    if args.between_run_log:
        rep3_cells = load_cells(args.between_run_log)
        between = between_run_reliability(cells, rep3_cells)

    overstatement = per_principle_overstatement(cells)

    report = build_report(
        args.log, cells, strata, contrasts_adj, contrast_a, contrast_b,
        within, between, overstatement, manifest, args.between_run_log,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report)
    print(f"Wrote {args.out_md}")

    if args.out_csv:
        write_csv(cells, args.out_csv)
        print(f"Wrote {args.out_csv}")

    # Console summary
    base_n = sum(1 for c in cells if c.repeat_kind is None)
    nan_n = sum(1 for c in cells if c.repeat_kind is None and c.is_nan)
    print(f"\nBase cells: {base_n}  NaN: {nan_n}")
    print(f"Contrast (a) FHR de-esc {contrast_a['fhr_deescalated']}/"
          f"{contrast_a['fhr_n']} vs ctrl {contrast_a['ctrl_regressed']}/"
          f"{contrast_a['ctrl_n']}; Holm p={contrasts_adj['(a)']:.4g}")
    print(f"Contrast (b) trivial reproduction {contrast_b['reproduced']}/"
          f"{contrast_b['n']}; Holm p={contrasts_adj['(b)']:.4g}")


if __name__ == "__main__":
    main()
