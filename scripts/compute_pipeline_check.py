#!/usr/bin/env python3
"""Pipeline check: reproduce the reported item counts straight from the logs.

Every number in the paper rests on the claim that 35,416 items were successfully
scored and 44 were dropped to judge-failure cascades. This script rebuilds that
accounting from the 45 `.eval` files with no intermediate artifact in the way,
and reports the full chain:

    800 prompts x 45 runs      = 36,000 samples on disk
      - 12 excluded prompts x 45 runs = 540 excluded by dataset flag
      = 35,460 items in analysis scope (788 x 45)
      - 44 judge-failure cascades
      - 0 off-scale severities
      = 35,416 successfully scored items  (x 3 judges = 106,248 rows)

`tables/inter_judge_agreement.md` reports the 36,000 and the 44 but omits the
540 line, which makes its arithmetic look like it is off by 540;
`scripts/compute_binarized_robustness_gap.py`'s docstring independently claims
35,956 rows. This script establishes which reading is correct.

It also emits the two structural facts every cohort-level analysis needs:

  - a 45-cell census of how many scenarios actually survived per
    (persona, model) cell -- cells are ragged (e.g. grok-4 bad, gemini-2.5-flash
    good), so "788" is not uniformly true;
  - the complete-case scenario set: scenarios present in all 45 cells, which is
    the only set over which a shared-scenario cluster bootstrap can carry one
    resample across the whole grid without silently reweighting cells.

Inputs (read-only):
  - logs/{baseline,good_persona,bad_persona}/<model>/*.eval   (45 files)
  - data/humane_bench.jsonl                (metadata.excluded_from_analysis)
  - tables/inter_judge_raw.csv             (committed artifact, for comparison)

Outputs (written to --output-dir, default tables/):
  - inter_judge_raw_regenerated.csv   fresh per-judge long table
  - cell_census.csv                   per-(persona, model) scenario counts
  - complete_case_scenarios.txt       scenario ids present in all 45 cells
  - pipeline_check.md                 the accounting chain + verdicts

Run from repo root:
    python scripts/compute_pipeline_check.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

from compute_inter_judge_agreement import collect_long_table  # noqa: E402
from humanebench.excluded import load_excluded_ids  # noqa: E402

# Reported values the paper and its .tex currently claim.
EXPECTED = {
    "n_prompts_total": 800,
    "n_prompts_excluded": 12,
    "n_prompts_analysis": 788,
    "n_runs": 45,
    "samples_on_disk": 36_000,
    "excluded_by_flag": 540,
    "in_scope": 35_460,
    "judge_failures": 44,
    "off_scale": 0,
    "scored": 35_416,
    "judge_rows": 106_248,
}

PERSONAS = ["baseline", "good_persona", "bad_persona"]


def _verdict(actual: int, expected: int) -> str:
    return "MATCH" if actual == expected else f"MISMATCH (expected {expected:,})"


def build_census(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (persona, model) with the count of distinct scenarios kept."""
    census = (
        long.groupby(["persona", "model"])["sample_id"]
        .nunique()
        .reset_index(name="n_scenarios")
    )
    census["shortfall_vs_788"] = EXPECTED["n_prompts_analysis"] - census["n_scenarios"]
    return census.sort_values(["persona", "model"]).reset_index(drop=True)


def complete_case_scenarios(long: pd.DataFrame) -> list[str]:
    """Scenario ids present in every (persona, model) cell.

    A shared-scenario cluster bootstrap draws one set of scenario ids and reuses
    it across the whole 15 x 3 grid. If a drawn scenario is absent from some
    cell, that cell's replicate silently reweights toward the scenarios it does
    have, which is not the estimator we intend. Restricting to the complete-case
    set removes that failure mode; the cost is reported explicitly.
    """
    n_cells = long.groupby("sample_id")[["persona", "model"]].apply(
        lambda g: len(set(map(tuple, g.to_numpy())))
    )
    expected_cells = long[["persona", "model"]].drop_duplicates().shape[0]
    return sorted(n_cells[n_cells == expected_cells].index)


def compare_to_committed(fresh: pd.DataFrame, committed_path: Path) -> dict:
    """Compare the fresh scan to the committed inter_judge_raw.csv, content-wise."""
    if not committed_path.is_file():
        return {"status": "ABSENT", "detail": f"{committed_path} not found"}

    committed = pd.read_csv(committed_path)
    key = ["persona", "model", "sample_id", "judge_name"]
    cols = key + ["severity", "principle"]

    missing = [c for c in cols if c not in committed.columns]
    if missing:
        return {"status": "SCHEMA", "detail": f"committed file missing {missing}"}

    a = fresh[cols].sort_values(key).reset_index(drop=True)
    b = committed[cols].sort_values(key).reset_index(drop=True)

    if len(a) != len(b):
        return {
            "status": "ROW COUNT",
            "detail": f"fresh {len(a):,} rows vs committed {len(b):,} rows",
            "n_fresh": len(a),
            "n_committed": len(b),
        }

    merged = a.merge(b, on=key, how="outer", suffixes=("_fresh", "_committed"),
                     indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    sev_diff = merged["severity_fresh"] != merged["severity_committed"]
    prin_diff = merged["principle_fresh"] != merged["principle_committed"]
    n_sev = int(sev_diff.fillna(True).sum())
    n_prin = int(prin_diff.fillna(True).sum())

    if unmatched == 0 and n_sev == 0 and n_prin == 0:
        return {"status": "IDENTICAL", "detail": f"{len(a):,} rows match on {key}"}
    return {
        "status": "DIFFERS",
        "detail": (f"{unmatched:,} unmatched keys, {n_sev:,} severity diffs, "
                   f"{n_prin:,} principle diffs"),
    }


def write_report(
    out: Path,
    stats: dict,
    long: pd.DataFrame,
    census: pd.DataFrame,
    complete: list[str],
    comparison: dict,
) -> None:
    on_disk = stats["total_samples"]
    excluded_flag = stats["samples_excluded_cut_list"]
    judge_fail = stats["samples_excluded_no_individual_scores"]
    off_scale = stats["samples_excluded_invalid_severity"]
    scored = stats["samples_included"]
    in_scope = on_disk - excluded_flag
    rows = len(long)

    L: list[str] = []
    L.append("# Pipeline check\n")
    L.append(
        "Rebuilt from the 45 `.eval` files with no intermediate artifact in the "
        "way. Every count below is read off the logs; nothing is carried "
        "forward from the `.tex` or from a derived table.\n"
    )

    L.append("## The accounting chain\n")
    L.append("| step | count | expected | verdict |")
    L.append("| --- | ---: | ---: | --- |")
    L.append(f"| `.eval` files scanned | {stats['files_scanned']} | "
             f"{EXPECTED['n_runs']} | {_verdict(stats['files_scanned'], EXPECTED['n_runs'])} |")
    L.append(f"| samples on disk (800 x 45) | {on_disk:,} | "
             f"{EXPECTED['samples_on_disk']:,} | {_verdict(on_disk, EXPECTED['samples_on_disk'])} |")
    L.append(f"| less: excluded by dataset flag (12 x 45) | -{excluded_flag:,} | "
             f"{EXPECTED['excluded_by_flag']:,} | "
             f"{_verdict(excluded_flag, EXPECTED['excluded_by_flag'])} |")
    L.append(f"| **items in analysis scope (788 x 45)** | **{in_scope:,}** | "
             f"{EXPECTED['in_scope']:,} | {_verdict(in_scope, EXPECTED['in_scope'])} |")
    L.append(f"| less: judge-failure cascades | -{judge_fail:,} | "
             f"{EXPECTED['judge_failures']} | "
             f"{_verdict(judge_fail, EXPECTED['judge_failures'])} |")
    L.append(f"| less: off-scale severities | -{off_scale:,} | "
             f"{EXPECTED['off_scale']} | {_verdict(off_scale, EXPECTED['off_scale'])} |")
    L.append(f"| **successfully scored items** | **{scored:,}** | "
             f"{EXPECTED['scored']:,} | {_verdict(scored, EXPECTED['scored'])} |")
    L.append(f"| per-judge rows (scored x 3) | {rows:,} | "
             f"{EXPECTED['judge_rows']:,} | {_verdict(rows, EXPECTED['judge_rows'])} |")
    L.append("")

    pct = 100.0 * judge_fail / in_scope if in_scope else float("nan")
    L.append(f"Judge-failure rate: **{judge_fail} / {in_scope:,} = {pct:.3f}%** of "
             "the analysis scope.\n")

    L.append("### Resolving the reported discrepancy\n")
    L.append(
        f"`tables/inter_judge_agreement.md` reports \"{scored:,} scored items "
        f"(samples scanned: {on_disk:,}; excluded (no individual_scores): "
        f"{judge_fail})\", which reads as though {on_disk:,} - {judge_fail} = "
        f"{on_disk - judge_fail:,} should be the scored total. It omits the "
        f"{excluded_flag:,}-item dataset-exclusion line "
        f"({EXPECTED['n_prompts_excluded']} excluded prompts x "
        f"{EXPECTED['n_runs']} runs). With that line restored the chain closes "
        f"exactly. The {on_disk - judge_fail:,} figure quoted in "
        "`scripts/compute_binarized_robustness_gap.py`'s docstring is the "
        "pre-exclusion count and is not the analysis denominator.\n"
    )
    L.append(
        f"**The paper's denominator for the judge-failure rate should be "
        f"{in_scope:,} (the 788-scenario analysis scope), not {on_disk:,}.**\n"
    )

    L.append("## Committed artifact comparison\n")
    L.append(f"`tables/inter_judge_raw.csv` vs this fresh scan: "
             f"**{comparison['status']}** -- {comparison['detail']}\n")

    L.append("## Cell census\n")
    ragged = census[census["shortfall_vs_788"] != 0]
    L.append(f"{len(census)} cells; **{len(ragged)} are short of "
             f"{EXPECTED['n_prompts_analysis']} scenarios**. Cells are ragged, "
             "so `n = 788` is not uniformly true and any analysis that assumes "
             "it will silently mis-weight.\n")
    if len(ragged):
        L.append("| persona | model | n scenarios | shortfall |")
        L.append("| --- | --- | ---: | ---: |")
        for _, r in ragged.iterrows():
            L.append(f"| {r['persona']} | {r['model']} | {r['n_scenarios']} | "
                     f"{r['shortfall_vs_788']} |")
        L.append("")
    L.append(f"Full census: `cell_census.csv`.\n")

    L.append("## Complete-case scenario set\n")
    n_c = len(complete)
    lost = EXPECTED["n_prompts_analysis"] - n_c
    L.append(
        f"Scenarios present in **all {len(census)} cells**: **{n_c:,}** of "
        f"{EXPECTED['n_prompts_analysis']} ({lost} lost, "
        f"{100.0 * lost / EXPECTED['n_prompts_analysis']:.2f}%).\n"
    )
    L.append(
        "This is the resampling frame for every cohort-level statistic (flip "
        "count, robust-set size) computed with the shared-scenario cluster "
        "bootstrap. Per-cell marginals still use the full per-cell scenario set; "
        "the two frames differ and the difference is reported wherever it "
        "matters. Ids: `complete_case_scenarios.txt`.\n"
    )

    out.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--committed", type=Path,
                    default=REPO_ROOT / "tables" / "inter_judge_raw.csv")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exclude_ids = load_excluded_ids()
    print(f"Dataset exclusion flags: {len(exclude_ids)} prompts")

    print(f"Scanning {args.logs_dir} ...")
    long, stats = collect_long_table(args.logs_dir, exclude_ids=exclude_ids)
    print(f"  files scanned          : {stats['files_scanned']}")
    print(f"  samples on disk        : {stats['total_samples']:,}")
    print(f"  excluded by flag       : {stats['samples_excluded_cut_list']:,}")
    print(f"  judge-failure cascades : {stats['samples_excluded_no_individual_scores']:,}")
    print(f"  off-scale severities   : {stats['samples_excluded_invalid_severity']:,}")
    print(f"  scored items           : {stats['samples_included']:,}")
    print(f"  per-judge rows         : {len(long):,}")

    census = build_census(long)
    complete = complete_case_scenarios(long)
    print(f"  complete-case scenarios: {len(complete):,}")

    comparison = compare_to_committed(long, args.committed)
    print(f"  vs committed raw csv   : {comparison['status']} -- {comparison['detail']}")

    long.to_csv(args.output_dir / "inter_judge_raw_regenerated.csv", index=False)
    census.to_csv(args.output_dir / "cell_census.csv", index=False)
    (args.output_dir / "complete_case_scenarios.txt").write_text(
        "\n".join(complete) + "\n"
    )
    write_report(args.output_dir / "pipeline_check.md", stats, long, census,
                 complete, comparison)
    print(f"\nWrote {args.output_dir / 'pipeline_check.md'}")


if __name__ == "__main__":
    main()
