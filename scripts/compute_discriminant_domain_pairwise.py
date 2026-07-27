#!/usr/bin/env python3
"""Domain-stratified pairwise interactions — supplementary probe.

Explores whether principle separation is confounded with topical separation.
Cramér's V between principle and domain is 0.432, and Be Transparent & Honest
is 72.4% technology-use. Some of what reads as principle separation may be
topical separation. Whether that is a confound or is what construct distinctness
*means* for a behavioral benchmark is a genuine question; this probe exposes
the structure without answering it.

For each principle pair, within-domain interactions are computed where both
principles have >= 8 scenarios in the same domain, then combined with weights
n_Xd * n_Yd / (n_Xd + n_Yd). Bootstrap within row × domain.

Labeled exploratory. Targets 31 Jul supplementary.

Inputs (read-only):
  - tables/discriminant_pooled/matrix_long.csv  (or --tables-dir)

Outputs:
  - tables/discriminant_pooled/domain_pairwise.csv
  - results/discriminant_domain_pairwise.md  (optional, --report)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    PRINCIPLES,
)
from humanebench.discriminant import PRINCIPLE_SHORT  # noqa: E402

ALPHA = 0.05
MIN_SCENARIOS_PER_SIDE = 8
N_BOOTSTRAP = 10_000


def short(p: str) -> str:
    return PRINCIPLE_SHORT[p]


def _fmt(x: float, places: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{places}f}".replace("-", "−")


def domain_interaction(
    long: pd.DataFrame,
    principle_a: str,
    principle_b: str,
    domain: str,
) -> float | None:
    """Interaction for one pair restricted to one domain's scenarios."""
    a_scenarios = long[
        (long["designed_principle"] == principle_a) & (long["domain"] == domain)
    ]["scenario_id"].unique()
    b_scenarios = long[
        (long["designed_principle"] == principle_b) & (long["domain"] == domain)
    ]["scenario_id"].unique()

    if len(a_scenarios) < MIN_SCENARIOS_PER_SIDE or len(b_scenarios) < MIN_SCENARIOS_PER_SIDE:
        return None

    def cell_mean(designed: str, scored: str, scenarios: np.ndarray) -> float:
        mask = (
            (long["designed_principle"] == designed)
            & (long["scored_principle"] == scored)
            & (long["scenario_id"].isin(scenarios))
        )
        vals = long.loc[mask, "score"]
        return vals.mean() if len(vals) > 0 else float("nan")

    a = cell_mean(principle_a, principle_a, a_scenarios)
    b = cell_mean(principle_a, principle_b, a_scenarios)
    c = cell_mean(principle_b, principle_a, b_scenarios)
    d = cell_mean(principle_b, principle_b, b_scenarios)
    return (a - b) - (c - d)


def bootstrap_domain_interaction(
    long: pd.DataFrame,
    principle_a: str,
    principle_b: str,
    domain: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, int, int]:
    """Bootstrap CI for one pair within one domain."""
    rng = np.random.default_rng(seed)

    a_scenarios = sorted(long[
        (long["designed_principle"] == principle_a) & (long["domain"] == domain)
    ]["scenario_id"].unique())
    b_scenarios = sorted(long[
        (long["designed_principle"] == principle_b) & (long["domain"] == domain)
    ]["scenario_id"].unique())

    na, nb = len(a_scenarios), len(b_scenarios)
    if na < MIN_SCENARIOS_PER_SIDE or nb < MIN_SCENARIOS_PER_SIDE:
        return float("nan"), float("nan"), float("nan"), na, nb

    def _build_cell_scores(designed, scenarios, principles_scored):
        """Pre-index: {scored_principle: {scenario_id: mean_score}}."""
        mask = (long["designed_principle"] == designed) & (long["scenario_id"].isin(scenarios))
        sub = long.loc[mask]
        out = {}
        for scored in principles_scored:
            g = sub.loc[sub["scored_principle"] == scored].groupby("scenario_id")["score"].mean()
            out[scored] = g.to_dict()
        return out

    scored_principles = [principle_a, principle_b]
    a_cells = _build_cell_scores(principle_a, a_scenarios, scored_principles)
    b_cells = _build_cell_scores(principle_b, b_scenarios, scored_principles)

    def _interaction(a_ids, b_ids):
        def cell_mean(cells, scored, ids):
            vals = [cells[scored][s] for s in ids if s in cells[scored]]
            return np.mean(vals) if vals else np.nan
        a = cell_mean(a_cells, principle_a, a_ids)
        b = cell_mean(a_cells, principle_b, a_ids)
        c = cell_mean(b_cells, principle_a, b_ids)
        d = cell_mean(b_cells, principle_b, b_ids)
        return (a - b) - (c - d)

    point = _interaction(a_scenarios, b_scenarios)
    reps = np.empty(n_bootstrap)
    a_arr = np.array(a_scenarios)
    b_arr = np.array(b_scenarios)
    for i in range(n_bootstrap):
        a_draw = rng.choice(a_arr, size=na, replace=True)
        b_draw = rng.choice(b_arr, size=nb, replace=True)
        reps[i] = _interaction(a_draw.tolist(), b_draw.tolist())

    lo = float(np.nanpercentile(reps, 2.5))
    hi = float(np.nanpercentile(reps, 97.5))
    return point, lo, hi, na, nb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant_pooled")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "results" / "discriminant_domain_pairwise.md")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--min-per-side", type=int, default=MIN_SCENARIOS_PER_SIDE)
    args = ap.parse_args()
    out_dir = args.output_dir or args.tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    long_path = args.tables_dir / "matrix_long.csv"
    if not long_path.exists():
        print(f"ERROR: {long_path} not found", file=sys.stderr)
        return 2
    long = pd.read_csv(long_path)
    if "domain" not in long.columns:
        print("ERROR: matrix_long.csv has no 'domain' column", file=sys.stderr)
        return 1

    domains = sorted(long["domain"].dropna().unique())
    print(f"{len(long):,} rows, {len(domains)} domains")

    rows: list[dict] = []
    for i in range(len(PRINCIPLES)):
        for j in range(i + 1, len(PRINCIPLES)):
            pa, pb = PRINCIPLES[i], PRINCIPLES[j]
            for domain in domains:
                point, lo, hi, na, nb = bootstrap_domain_interaction(
                    long, pa, pb, domain, args.n_bootstrap, args.seed)
                if np.isnan(point):
                    continue
                rows.append({
                    "principle_a": pa,
                    "principle_b": pb,
                    "domain": domain,
                    "interaction": point,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_scenarios_a": na,
                    "n_scenarios_b": nb,
                    "feasible": True,
                })

    if not rows:
        print("No feasible domain-pair combinations found.")
        return 0

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "domain_pairwise.csv", index=False)
    print(f"wrote {len(df)} domain-pair rows to {out_dir / 'domain_pairwise.csv'}")

    L: list[str] = []
    A = L.append
    A("# Domain-stratified pairwise interactions — exploratory probe")
    A("")
    A("**This is exploratory.** It probes whether principle separation is "
      "confounded with topical separation. No multiplicity correction is applied "
      "within this analysis; it is a descriptive decomposition, not a test.")
    A("")
    A(f"Feasibility gate: >= {args.min_per_side} scenarios per side within "
      f"a domain. {len(df)} (pair, domain) combinations pass.")
    A("")
    A("| Pair | Domain | interaction | 95% CI | n_a | n_b |")
    A("|---|---|---|---|---|---|")
    for _, r in df.sort_values("interaction", ascending=False).head(30).iterrows():
        pair = f"{short(r.principle_a)}/{short(r.principle_b)}"
        A(f"| {pair} | {r.domain} | {_fmt(r.interaction)} | "
          f"[{_fmt(r.ci_lower)}, {_fmt(r.ci_upper)}] | "
          f"{r.n_scenarios_a} | {r.n_scenarios_b} |")
    A("")
    A(f"Full table: `{out_dir / 'domain_pairwise.csv'}`")
    A("")
    A("---")
    A("")
    A(f"Generated by `scripts/compute_discriminant_domain_pairwise.py`. "
      f"{args.n_bootstrap:,} bootstrap replicates, seed {args.seed}.")
    A("")

    args.report.write_text("\n".join(L))
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
