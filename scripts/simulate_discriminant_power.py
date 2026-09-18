#!/usr/bin/env python3
"""Power of the pairwise discriminant test at larger per-principle sample sizes.

WHAT THIS ANSWERS
-----------------
`scripts/compute_discriminant_pairwise.py` distinguishes 23 of 28 principle
pairs at 12 scenarios per principle. Five pairs fail, and §7 of
`results/discriminant_pairwise.md` names the sample as the one limitation more
data would fix. Before spending an expansion run to find out, this simulates
what that run would buy.

DESIGN
------
A two-level bootstrap. The **outer** level draws a hypothetical larger scenario
set by resampling, with replacement, from the 12 scenarios each principle
already has; the drawn scenario carries its full 24 judged calls (8 scored
principles x 3 source models), so a scenario is resampled as one cluster, which
is the unit `bootstrap_designed_measured_matrix` treats it as. The **inner**
level is the published analysis run unchanged on that hypothetical set:
`bootstrap_designed_measured_matrix` -> `pairwise_interactions`, Holm across all
28 pairs, alpha = 0.05.

Two target sizes, modelling two different things:

- **n = 36** — the 12 observed scenarios held FIXED plus 24 newly drawn. This
  is what pooling an expansion run onto the existing data looks like: the
  current scenarios do not go away.
- **n = 24** — a pure 24-scenario draw, no originals retained. This is what a
  standalone replication wave looks like.

Drawn scenarios are relabelled `{scenario_id}#{k}`. Two draws of the same
scenario must be two clusters, not one: without the relabel the duplicate-key
guard in `bootstrap_designed_measured_matrix` would (correctly) refuse the frame.

WHY B = 2,000 IS ENOUGH FOR THE INNER BOOTSTRAP
-----------------------------------------------
The bootstrap p has a floor of 2/(B+1), and Holm's first step compares the
smallest of 28 p-values against 0.05/28 = 0.0017857. At B = 2,000 the floor is
0.0009995, which clears it. (At the repo default B = 1,000 the floor is 0.0020
and nothing can pass — see the replicate-count section of
`compute_discriminant_pairwise.py`.) The published run uses B = 10,000 for
tighter CI bounds; 2,000 is used here because the inner loop runs `--outer`
times per size and the quantity reported is a pass/fail rate, not a CI bound.

WHAT THIS IS OPTIMISTIC ABOUT
-----------------------------
Resampling from 12 observed scenarios treats those 12 as the population. A real
expansion draws genuinely new scenarios, which will include effects the observed
12 never expressed. The simulated tails are therefore too thin and these
probabilities should be read as an upper bound, not a forecast.

No API calls. Reads the matrix the scored run already produced. Writes only its
own two outputs.

Run from repo root:
    python scripts/simulate_discriminant_power.py
    python scripts/simulate_discriminant_power.py --outer 3 --inner 200  # smoke
"""
# Paper: produces tables/discriminant/power_simulation.csv -- the per-pair
# probability of passing Holm at larger per-principle samples, which is what
# motivated drawing the second separability sample (main paper, "Principle
# Separability").
# Paper: implements a two-level bootstrap: an outer resample of whole scenario
# clusters up to 24 and 36 scenarios per principle, and inside each hypothetical
# set the published pairwise interaction analysis with Holm correction across the
# 28-pair family.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    PRINCIPLES,
    bootstrap_designed_measured_matrix,
    pairwise_interactions,
)
from humanebench.discriminant import PRINCIPLE_SHORT  # noqa: E402

ALPHA = 0.05
N_PAIRS = len(PRINCIPLES) * (len(PRINCIPLES) - 1) // 2
N_BOOTSTRAP_INNER = 2_000
N_OUTER = 500
SIZES = (24, 36)

# The size at which the 12 observed scenarios are carried through unchanged and
# only the remainder is drawn. Any other size is simulated as a fresh draw.
POOLED_SIZE = 36


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def short(principle: str) -> str:
    return PRINCIPLE_SHORT[principle]


def _fmt(x: float, places: int = 3) -> str:
    """Signed fixed-point with a real minus sign, matching the sibling report."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{places}f}".replace("-", "−")


def _pair_label(row: pd.Series) -> str:
    return f"{short(row.principle_a)} / {short(row.principle_b)}"


def build_index(long: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    """Scenario ids per designed principle, and each scenario's row positions."""
    by_principle: dict[str, list[str]] = {}
    for principle in PRINCIPLES:
        sub = long[long["designed_principle"] == principle]
        by_principle[principle] = sorted(sub["scenario_id"].unique())

    positions: dict[str, np.ndarray] = {}
    for sid, idx in long.groupby("scenario_id").indices.items():
        positions[sid] = np.asarray(idx, dtype=int)
    return by_principle, positions


def resample_frame(
    long: pd.DataFrame,
    by_principle: dict[str, list[str]],
    positions: dict[str, np.ndarray],
    size: int,
    rng: np.random.Generator,
    keep_originals: bool,
) -> pd.DataFrame:
    """One hypothetical scenario set of `size` scenarios per designed principle.

    Each drawn scenario carries all of its rows, so the resampling unit is the
    scenario cluster rather than the individual judged call. Draws are relabelled
    so a scenario drawn twice contributes two distinct clusters.
    """
    take: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for principle in PRINCIPLES:
        sids = by_principle[principle]
        n = len(sids)
        n_draw = size
        if keep_originals:
            for sid in sids:
                pos = positions[sid]
                take.append(pos)
                labels.append(np.full(pos.size, sid, dtype=object))
            n_draw = size - n
        for k, pick in enumerate(rng.integers(0, n, size=n_draw)):
            sid = sids[pick]
            pos = positions[sid]
            take.append(pos)
            labels.append(np.full(pos.size, f"{sid}#{k}", dtype=object))

    out = long.iloc[np.concatenate(take)].copy()
    out["scenario_id"] = np.concatenate(labels)
    return out.reset_index(drop=True)


def simulate(
    long: pd.DataFrame,
    sizes: list[int],
    n_outer: int,
    n_inner: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Holm-pass rate per (pair, size), plus the per-replicate pass counts."""
    by_principle, positions = build_index(long)
    n_observed = min(len(s) for s in by_principle.values())

    pair_keys: list[tuple[str, str]] | None = None
    passes = {size: np.zeros((n_outer, N_PAIRS), dtype=bool) for size in sizes}
    started = time.time()

    for rep, child in enumerate(np.random.SeedSequence(seed).spawn(n_outer)):
        scen_ss, inner_ss = child.spawn(2)
        rng = np.random.default_rng(scen_ss)
        inner_rng = np.random.default_rng(inner_ss)

        for size in sizes:
            frame = resample_frame(
                long, by_principle, positions, size, rng,
                keep_originals=(size == POOLED_SIZE and size > n_observed),
            )
            matrix = bootstrap_designed_measured_matrix(
                frame, n_bootstrap=n_inner,
                seed=int(inner_rng.integers(0, 2**63)),
            )
            pw = pairwise_interactions(matrix)
            if pair_keys is None:
                pair_keys = list(zip(pw["principle_a"], pw["principle_b"]))
            passes[size][rep] = (pw["p_holm"] < ALPHA).to_numpy()

        if (rep + 1) % 25 == 0 or rep + 1 == n_outer:
            rate = (time.time() - started) / (rep + 1)
            print(f"  replicate {rep + 1}/{n_outer} "
                  f"({rate:.2f}s each, {rate * (n_outer - rep - 1) / 60:.1f} min left)",
                  flush=True)

    assert pair_keys is not None
    rows = [{
        "principle_a": a,
        "principle_b": b,
        "size": size,
        "holm_pass_probability": float(passes[size][:, i].mean()),
    } for size in sizes for i, (a, b) in enumerate(pair_keys)]

    counts = {size: passes[size].sum(axis=1) for size in sizes}
    return pd.DataFrame(rows), counts


def write_report(
    power: pd.DataFrame,
    counts: dict[int, np.ndarray],
    observed: pd.DataFrame,
    sizes: list[int],
    n_outer: int,
    n_inner: int,
    seed: int,
    n_observed: int,
    path: Path,
    csv_path: Path,
) -> None:
    floor = 2.0 / (n_inner + 1)
    threshold = ALPHA / N_PAIRS
    failing = observed[~observed["significant"]]
    wide = power.pivot_table(
        index=["principle_a", "principle_b"], columns="size",
        values="holm_pass_probability",
    )
    obs_sig = observed.set_index(["principle_a", "principle_b"])["significant"]

    L: list[str] = []
    A = L.append

    A("# Power of the pairwise discriminant test at larger samples")
    A("")
    A(f"At the observed {n_observed} scenarios per principle, "
      f"{int(observed['significant'].sum())} of {N_PAIRS} pairs are "
      "distinguishable after Holm. Simulated expectation:")
    A("")
    A("| Scenarios per principle | expected pairs passing | median | 90% range |")
    A("|---|---|---|---|")
    for size in sizes:
        exp = power.loc[power["size"] == size, "holm_pass_probability"].sum()
        c = counts[size]
        A(f"| {size} | {exp:.1f} / {N_PAIRS} | {int(np.median(c))} | "
          f"{int(np.percentile(c, 5))}–{int(np.percentile(c, 95))} |")
    A("")
    A("The expectation is the sum of the per-pair pass probabilities; the median "
      "and range are percentiles of the pass count actually realised in each "
      "outer replicate, which is the more honest summary because the 28 tests "
      "move together (they are built from 8 overlapping matrix rows, and Holm "
      "makes each pair's verdict depend on the other 27).")
    A("")

    # ---- the pairs that currently fail --------------------------------------
    A(f"## The {len(failing)} pairs that currently fail")
    A("")
    A("These are what an expansion run is for. A pair whose probability stays "
      "low at 36 will not be rescued by this much more data.")
    A("")
    A("| Pair | interaction | current 95% CI | "
      + " | ".join(f"P(pass) at n={s}" for s in sizes) + " |")
    A("|---|---|---|" + "---|" * len(sizes))
    for _, r in failing.sort_values("interaction", ascending=False).iterrows():
        probs = " | ".join(
            f"{wide.loc[(r.principle_a, r.principle_b), s]:.0%}" for s in sizes)
        A(f"| {_pair_label(r)} | {_fmt(r.interaction)} | "
          f"[{_fmt(r.ci_lower)}, {_fmt(r.ci_upper)}] | {probs} |")
    A("")

    # ---- everything else -----------------------------------------------------
    A(f"## All {N_PAIRS} pairs")
    A("")
    A("| Pair | currently significant | "
      + " | ".join(f"P(pass) at n={s}" for s in sizes) + " |")
    A("|---|---|" + "---|" * len(sizes))
    ordered = wide.sort_values(max(sizes), ascending=False)
    for (a, b), row in ordered.iterrows():
        probs = " | ".join(f"{row[s]:.0%}" for s in sizes)
        mark = "yes" if bool(obs_sig.loc[(a, b)]) else "**no**"
        A(f"| {short(a)} / {short(b)} | {mark} | {probs} |")
    A("")
    A(f"(`{_rel(csv_path)}`; short codes: "
      + ", ".join(f"{short(p)} = {p}" for p in PRINCIPLES) + ".)")
    A("")

    # ---- method --------------------------------------------------------------
    A("## Method")
    A("")
    A(f"- **Outer level.** {n_outer:,} replicates. For each, and for each "
      "designed principle, scenario ids are resampled with replacement from the "
      f"{n_observed} that principle already has. A drawn scenario carries all "
      "24 of its judged calls (8 scored principles × 3 source models), so the "
      "resampling unit is the scenario cluster.")
    A(f"- **n = {POOLED_SIZE}.** The {n_observed} observed scenarios are held "
      f"fixed and {POOLED_SIZE - n_observed} more are drawn — what pooling an "
      "expansion run onto the existing data looks like. Any other size is a "
      "fresh draw of that many scenarios, modelling a standalone replication "
      "wave.")
    A("- **Relabelling.** Draws are renamed `{scenario_id}#{k}` so a scenario "
      "drawn twice forms two clusters rather than colliding into one. Without "
      "it the duplicate-key guard in `bootstrap_designed_measured_matrix` "
      "rejects the frame.")
    A("- **Inner level.** The published analysis, unchanged: "
      f"`bootstrap_designed_measured_matrix` at B = {n_inner:,}, then "
      "`pairwise_interactions`, then Holm across all 28 pairs at "
      f"α = {ALPHA}.")
    A(f"- **Seeds.** `numpy.random.SeedSequence({seed}).spawn({n_outer})`, one "
      "child per outer replicate, split again into an independent stream for "
      "the scenario draw and for the inner bootstrap seed. The whole run is "
      "reproducible from the one seed.")
    A("")
    A(f"### Why B = {n_inner:,} inner replicates")
    A("")
    A(f"The bootstrap p has a floor of 2/(B+1) = {floor:.7f}. Holm's first step "
      f"compares the smallest of {N_PAIRS} p-values against "
      f"{ALPHA}/{N_PAIRS} = {threshold:.7f}.")
    if floor < threshold:
        A("The floor clears the threshold, so the inner resolution does not "
          "cap the result.")
    else:
        A("**The floor is above the threshold, so no pair can pass whatever "
          "the data say.** Every probability below is an artifact of the "
          "replicate count, not a finding. Raise `--inner`.")
    A("")

    # ---- limits --------------------------------------------------------------
    A("## What this is optimistic about")
    A("")
    A(f"Resampling from the {n_observed} observed scenarios treats those "
      f"{n_observed} as the population. A real expansion run draws genuinely "
      "new scenarios, which will express variation the observed set never did — "
      "including scenarios that engage a pair's two rubrics more similarly than "
      "any of the current ones. The simulated tails are thinner than the real "
      "ones, so **these probabilities are an upper bound on what an expansion "
      "run would deliver, not a forecast of it.**")
    A("")
    A("Two further caveats that follow from the same construction:")
    A("")
    A("- The point estimates are anchored to the observed matrix. A pair whose "
      f"true interaction is smaller than what these {n_observed} scenarios "
      "happened to show will pass here and fail in the real run.")
    A(f"- At n = {POOLED_SIZE} the {n_observed} originals are held fixed, so "
      "every replicate shares them. That is faithful to how pooling works, and "
      "it also means the replicates are correlated through those scenarios — "
      "the spread of the pass count is narrower than it would be for "
      "independent runs of the same size.")
    A("")
    A("---")
    A("")
    A("Generated by `scripts/simulate_discriminant_power.py`. No API calls: "
      "re-analyses the judged calls already in "
      "`tables/discriminant/matrix_long.csv`. "
      f"{n_outer:,} outer × {n_inner:,} inner replicates, seed {seed}.")
    A("")

    path.write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant" / "matrix_long.csv")
    ap.add_argument("--observed", type=Path, default=None,
                    help="published pairwise_interactions.csv, for the "
                         "currently-failing pairs (default: beside --matrix)")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES),
                    help="target scenarios per principle")
    ap.add_argument("--outer", type=int, default=N_OUTER,
                    help="hypothetical scenario sets drawn per size")
    ap.add_argument("--inner", type=int, default=N_BOOTSTRAP_INNER,
                    help="bootstrap replicates inside each hypothetical set")
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant" / "power_simulation.csv")
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "results" / "discriminant_power_simulation.md")
    args = ap.parse_args()

    if not args.matrix.exists():
        print(f"ERROR: {args.matrix} not found; run "
              "scripts/compute_discriminant_validity.py first", file=sys.stderr)
        return 2
    long = pd.read_csv(args.matrix)

    observed_path = args.observed or args.matrix.parent / "pairwise_interactions.csv"
    if not observed_path.exists():
        print(f"ERROR: {observed_path} not found; run "
              "scripts/compute_discriminant_pairwise.py first", file=sys.stderr)
        return 2
    observed = pd.read_csv(observed_path)

    n_observed = long.groupby("designed_principle")["scenario_id"].nunique()
    if n_observed.nunique() != 1:
        print(f"ERROR: unequal scenarios per principle: {n_observed.to_dict()}. "
              "The pooled-size arithmetic assumes a balanced design.",
              file=sys.stderr)
        return 1
    n_per_principle = int(n_observed.iloc[0])

    sizes = sorted(set(args.sizes))
    too_small = [s for s in sizes if s < n_per_principle]
    if too_small:
        print(f"ERROR: --sizes {too_small} are below the observed "
              f"{n_per_principle} per principle.", file=sys.stderr)
        return 1

    floor = 2.0 / (args.inner + 1)
    threshold = ALPHA / N_PAIRS
    if floor >= threshold:
        print(f"WARNING: inner B = {args.inner:,} gives a p floor of "
              f"{floor:.6f}, at or above the Holm threshold {threshold:.6f}. "
              "No pair can pass; every probability will be 0. Use --inner "
              f"{N_BOOTSTRAP_INNER:,} for a real run.", file=sys.stderr)

    print(f"{len(long):,} judged calls, {long.scenario_id.nunique()} scenarios, "
          f"{long.source_model.nunique()} models, {n_per_principle} per principle")
    print(f"simulating sizes {sizes} — {args.outer:,} outer x {args.inner:,} "
          f"inner replicates, seed {args.seed}")

    power, counts = simulate(long, sizes, args.outer, args.inner, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    power.to_csv(args.output, index=False)
    write_report(power, counts, observed, sizes, args.outer, args.inner,
                 args.seed, n_per_principle, args.report, args.output)

    for size in sizes:
        exp = power.loc[power["size"] == size, "holm_pass_probability"].sum()
        print(f"n={size}: expected {exp:.1f}/{N_PAIRS} pairs pass "
              f"(median {int(np.median(counts[size]))})")
    print(f"  wrote {_rel(args.output)}")
    print(f"  wrote {_rel(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
