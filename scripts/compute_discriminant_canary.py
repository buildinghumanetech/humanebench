#!/usr/bin/env python3
"""Judge-drift gate: has the unpinned gpt-5.1 slug moved since the matrix was run?

`scripts/build_discriminant_canary.py` re-issues 96 of the 2,304 calls that
produced `tables/discriminant/matrix_long.csv`, with the same template, the same
scenario prompts and the same archived responses. The bytes sent to the judge are
identical to the published run's, so a score that comes back different did not
come back different because of the design. It came back different because
`openrouter/openai/gpt-5.1` is not a pinned model.

That matters because the follow-up conditions are meant to be *pooled* with the
published matrix. Pooling across a judge change would put drift inside a number
the paper attributes to the scenarios, and nothing downstream would show it.

THE GATE
--------
Exit 0 only if the re-judged slice reproduces the published scores closely
enough to pool:

    exact agreement >= 0.75   and   |mean shift| <= 0.10

Two thresholds because they fail differently. Exact agreement catches a judge
that has become noisier without moving on average -- the case a mean would hide.
The mean shift catches a judge that has become systematically harsher or more
lenient while still agreeing at the usual rate on the easy items, which is the
case that would bias a pooled estimate. Both are on the 4-point severity scale,
where one level is 0.5, so a 0.10 mean shift is a fifth of the smallest step the
scale can express.

Anything else exits 1. The canary is a stop sign, not a correction factor: there
is no adjustment that makes two judges into one, so a failure is a decision for
the authors, not for this script.

No API calls. Reads the canary run's logs and the published matrix.

Inputs (read-only):
  - logs/discriminant_canary/<model>/*.eval
  - tables/discriminant/matrix_long.csv

Outputs (written to --output-dir, default tables/discriminant_canary/):
  - canary_comparison.csv    one row per joined cell, both scores and the diff

Run from repo root, after the canary run finishes:
    python scripts/compute_discriminant_canary.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from humanebench import provenance as prov  # noqa: E402
from humanebench.discriminant import CONDITIONS  # noqa: E402

from compute_discriminant_validity import ORDINAL_LEVELS, select_eval  # noqa: E402

CONDITION = "discriminant_canary"
MIN_JOINED = 90
MIN_EXACT_AGREEMENT = 0.75
MAX_ABS_MEAN_SHIFT = 0.10
STOP = "STOP — consult Andalib before pooling."


def _rel(path: Path, logs_dir: Path | None = None) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        if logs_dir is not None:
            try:
                return str(Path("logs") / path.relative_to(logs_dir))
            except ValueError:
                pass
        return path.name


def load_condition_scores(logs_dir: Path, condition: str,
                          models: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Long table of one condition's run, one row per judged call.

    Admission is the rule `compute_discriminant_validity.load_run_scores` uses --
    a sample counts only if its severity is on the canonical 4-point scale --
    because the comparison is against the table that rule produced. A looser rule
    here would let cells into the canary that the published matrix excluded, and
    the agreement figure would then be measuring the admission rule.
    """
    rows: list[dict] = []
    stats = {"samples_seen": 0, "admitted": 0, "no_score": 0, "off_scale": 0,
             "files": []}

    root = logs_dir / condition
    dirs = ([root / m for m in models] if models
            else sorted(p for p in root.glob("*") if p.is_dir()))
    for model_dir in dirs:
        paths = sorted(model_dir.glob("*.eval")) if model_dir.is_dir() else []
        if not paths:
            continue
        path = select_eval(paths)
        stats["files"].append(_rel(path, logs_dir))
        for sample in prov.iter_eval_samples(path):
            stats["samples_seen"] += 1
            meta = (sample.get("metadata") or {}).get("metadata") or {}
            overseer = (sample.get("scores") or {}).get("overseer")
            if not overseer:
                stats["no_score"] += 1
                continue
            smeta = overseer.get("metadata") or {}
            individual = smeta.get("individual_scores")
            judges = smeta.get("ensemble_models")
            if not individual or not judges or len(individual) != len(judges):
                stats["no_score"] += 1
                continue
            if any(s not in ORDINAL_LEVELS for s in individual):
                stats["off_scale"] += 1
                continue
            rows.append({
                "scenario_id": meta.get("scenario_id"),
                "source_model": meta.get("source_model") or model_dir.name,
                "designed_principle": meta.get("designed_principle"),
                "scored_principle": meta.get("scored_principle") or sample.get("target"),
                "canary_score": float(individual[0]),
            })
            stats["admitted"] += 1

    df = pd.DataFrame(rows)
    if not df.empty and df[["scenario_id", "source_model",
                            "scored_principle"]].isna().any().any():
        raise SystemExit("canary logs are missing the metadata the join keys on")
    return df, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--tables-dir", type=Path,
                    default=REPO_ROOT / "tables" / "discriminant",
                    help="where the published matrix_long.csv lives")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: tables/<condition>/")
    ap.add_argument("--condition", default=CONDITION, choices=sorted(CONDITIONS),
                    help="which condition's logs hold the re-judged slice")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or REPO_ROOT / "tables" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = args.tables_dir / "matrix_long.csv"
    if not long_path.exists():
        print(f"ERROR: {_rel(long_path)} not found; the canary has nothing to "
              "compare against.", file=sys.stderr)
        return 2
    published = pd.read_csv(long_path).rename(columns={"score": "original_score"})

    canary, stats = load_condition_scores(args.logs_dir, args.condition, args.models)
    if canary.empty:
        print(f"ERROR: no scored samples under "
              f"{_rel(args.logs_dir / args.condition, args.logs_dir)}.",
              file=sys.stderr)
        return 2

    keys = ["scenario_id", "source_model", "scored_principle"]
    joined = canary.merge(
        published[keys + ["designed_principle", "original_score"]],
        on=keys, how="inner", suffixes=("", "_published"))
    if joined[keys].duplicated().any():
        # A duplicate key means one of the two tables holds two verdicts for the
        # same call, and the agreement figure would then be averaging a cell
        # against itself. That is a broken input, not a drift result.
        raise SystemExit("duplicate (scenario, model, principle) keys in the join")
    # The two tables agree on which principle a scenario was designed for, or the
    # frame moved under the canary and the comparison is not like-for-like.
    if (joined["designed_principle"] != joined["designed_principle_published"]).any():
        raise SystemExit("canary and published tables disagree on designed_principle")

    joined["difference"] = joined["canary_score"] - joined["original_score"]
    joined["agree"] = joined["canary_score"] == joined["original_score"]
    joined = joined.sort_values(keys).reset_index(drop=True)

    out_path = out_dir / "canary_comparison.csv"
    joined[["scenario_id", "source_model", "designed_principle", "scored_principle",
            "canary_score", "original_score", "difference", "agree"]].to_csv(
        out_path, index=False)

    n_joined = len(joined)
    unjoined = stats["admitted"] - n_joined
    print(f"canary run: {stats['admitted']} of {stats['samples_seen']} samples "
          f"admitted (no score {stats['no_score']}, off-scale {stats['off_scale']})")
    print("logs: " + ", ".join(stats["files"]))
    if unjoined:
        print(f"note: {unjoined} admitted canary call(s) have no counterpart in "
              f"{_rel(long_path)}")
    print(f"wrote {_rel(out_path)}")

    if n_joined < MIN_JOINED:
        print(f"\n{n_joined} joined cells, need at least {MIN_JOINED}. The gate "
              "cannot be evaluated on this many.", file=sys.stderr)
        print(STOP, file=sys.stderr)
        return 1

    exact = float(joined["agree"].mean())
    mean_shift = float(joined["difference"].mean())
    print(f"\njoined cells:    {n_joined}")
    print(f"exact agreement: {exact:.3f}  (threshold >= {MIN_EXACT_AGREEMENT:.2f})")
    print(f"mean shift:      {mean_shift:+.3f}  "
          f"(threshold |shift| <= {MAX_ABS_MEAN_SHIFT:.2f})")

    passed = (exact >= MIN_EXACT_AGREEMENT
              and math.isfinite(mean_shift)
              and abs(mean_shift) <= MAX_ABS_MEAN_SHIFT)
    if not passed:
        print(f"\n{STOP}", file=sys.stderr)
        return 1
    print("\nPASS: the judge reproduces the published scores closely enough to pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
