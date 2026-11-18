#!/usr/bin/env python3
"""Pool the original and expansion discriminant runs into one 36-per-principle table.

The pre-registered discriminant run scored 12 scenarios per principle; the
expansion run scored a further 24, drawn the same way from the same frame and
scored with the same as-deployed rubric, the same single judge, the same three
source models and the same archived baseline responses. Together they are one
sample of 36 per principle -- the size Run A of the follow-up spec argues is
needed for the underpowered pairwise failures to resolve.

This script does the merge and nothing else. It runs no new judge calls and
recomputes no published number. Its output is the input that
`scripts/compute_discriminant_pairwise.py --tables-dir tables/discriminant_pooled`
expects: a `matrix_long.csv` and the `fhr_pltw.csv` that script reconciles its
headline interaction against.

Two runs are only poolable if they are the same measurement, so every way they
could fail to be is checked before anything is written: identical schema,
disjoint scenario sets (a scenario scored twice would enter the row mean twice
and silently shrink its own bootstrap variance), the expected scenario count per
row, and no duplicate (scenario, model, scored principle) key.

Run from repo root, after both runs' tables exist:
    python scripts/build_discriminant_pooled_tables.py
"""
# Paper: produces tables/discriminant_pooled/matrix_long.csv and fhr_pltw.csv --
# the pooled 36-per-principle designed x scored sample and the Foster Healthy
# Relationships / Prioritize Long-term Wellbeing directional contrasts reported
# for it (main paper, "Principle Separability").
# Paper: implements the pooling checks and the scenario-cluster bootstrap CIs at
# 10,000 replicates, seed 20260407.
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
    bootstrap_designed_measured_matrix,
)
from humanebench.discriminant import PRINCIPLE_SHORT  # noqa: E402

FHR = "foster-healthy-relationships"
PLTW = "prioritize-long-term-wellbeing"

# The pairwise family is 28 tests, so Holm demands 0.05 / 28 = 0.0018 of the
# most significant. The bootstrap p floor is 2 / (B + 1), which at B = 1,000 is
# 0.0020 -- above that threshold, making the family unresolvable by
# construction. B = 10,000 puts the floor at 0.0002.
N_BOOTSTRAP_PAIRWISE = 10_000
SCENARIOS_PER_PRINCIPLE = 36


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def load_and_check(
    original_path: Path, expansion_path: Path, per_principle: int
) -> pd.DataFrame:
    original = pd.read_csv(original_path)
    expansion = pd.read_csv(expansion_path)

    if list(original.columns) != list(expansion.columns):
        raise SystemExit(
            f"schema mismatch: {_rel(original_path)} has {list(original.columns)}, "
            f"{_rel(expansion_path)} has {list(expansion.columns)}. The two runs "
            "must be the same measurement to pool."
        )

    overlap = set(original["scenario_id"]) & set(expansion["scenario_id"])
    if overlap:
        raise SystemExit(
            f"{len(overlap)} scenario(s) appear in both runs, e.g. "
            f"{sorted(overlap)[:3]}. The expansion must be a disjoint draw."
        )

    merged = pd.concat([original, expansion], ignore_index=True)

    per_row = merged.groupby("designed_principle")["scenario_id"].nunique()
    missing = set(PRINCIPLES) - set(per_row.index)
    if missing:
        raise SystemExit(f"no scenarios for designed principle(s): {sorted(missing)}")
    wrong = per_row[per_row != per_principle]
    if not wrong.empty:
        raise SystemExit(
            f"expected {per_principle} scenarios per designed principle, got "
            f"{wrong.to_dict()}. Rows of unequal size are poolable but the pooled "
            "analysis is not the one that was powered; pass "
            "--scenarios-per-principle if this is deliberate."
        )

    models = sorted(merged["source_model"].unique())
    expected_rows = per_principle * len(PRINCIPLES) * len(PRINCIPLES) * len(models)
    if len(merged) != expected_rows:
        raise SystemExit(
            f"{len(merged):,} rows, expected {expected_rows:,} "
            f"({per_principle} scenarios x {len(PRINCIPLES)} principles x "
            f"{len(PRINCIPLES)} rubrics x {len(models)} models). A partial matrix "
            "is not pooled."
        )

    key = ["scenario_id", "source_model", "scored_principle"]
    dupes = merged.duplicated(subset=key, keep=False)
    if dupes.any():
        example = merged.loc[dupes, key].drop_duplicates().head(3).to_dict("records")
        raise SystemExit(
            f"{int(dupes.sum())} duplicate {tuple(key)} rows; e.g. {example}. "
            "Each judged call must appear exactly once."
        )

    return merged


def build_fhr_pltw(merged: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    matrix = bootstrap_designed_measured_matrix(
        merged, n_bootstrap=n_bootstrap, seed=seed)
    rows = []
    for designed, other in ((FHR, PLTW), (PLTW, FHR)):
        d, lo, hi = matrix.cell_difference(designed, designed, other)
        estimable = bool(np.isfinite(d) and np.isfinite(lo) and np.isfinite(hi))
        rows.append({
            "comparison": f"{PRINCIPLE_SHORT[designed]}-designed: "
                          f"{PRINCIPLE_SHORT[designed]} minus {PRINCIPLE_SHORT[other]}",
            "designed_principle": designed,
            "scored_a": designed,
            "scored_b": other,
            "difference": d,
            "ci_lower": lo,
            "ci_upper": hi,
            "estimable": estimable,
            "excludes_zero": bool(estimable and (hi < 0 or lo > 0)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant" / "matrix_long.csv")
    ap.add_argument("--expansion", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant_expansion"
                            / "matrix_long.csv")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant_pooled")
    ap.add_argument("--scenarios-per-principle", type=int,
                    default=SCENARIOS_PER_PRINCIPLE)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_PAIRWISE)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    for path in (args.original, args.expansion):
        if not path.exists():
            print(f"ERROR: {_rel(path)} not found", file=sys.stderr)
            return 2

    merged = load_and_check(args.original, args.expansion,
                            args.scenarios_per_principle)
    print(f"{len(merged):,} judged calls, {merged.scenario_id.nunique()} scenarios, "
          f"{merged.source_model.nunique()} models")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "matrix_long.csv", index=False)

    fhr_pltw = build_fhr_pltw(merged, args.n_bootstrap, args.seed)
    fhr_pltw.to_csv(args.output_dir / "fhr_pltw.csv", index=False)
    for _, r in fhr_pltw.iterrows():
        print(f"{r.comparison}: {r.difference:+.3f} "
              f"[{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]"
              f"  {'excludes 0' if r.excludes_zero else 'includes 0'}")

    for name in ("matrix_long.csv", "fhr_pltw.csv"):
        print(f"  wrote {_rel(args.output_dir / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
