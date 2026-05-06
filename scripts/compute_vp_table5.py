#!/usr/bin/env python3
"""Compute within-principle VP-vs-non-VP score gaps for Table 5.

Reads tables/vp_sample_scores.csv and computes, for each principle with
>= 20 VP-tagged scenarios, the mean score gap (VP-tagged minus non-VP)
across all 15 models, per persona condition (baseline, good, bad).

Outputs tables/vp_table5.csv and prints a LaTeX-ready table fragment.

# TODO: add bootstrap CIs (Andalib to layer on with PR #71 pattern)
"""

import argparse
from pathlib import Path

import pandas as pd

VP_SCENARIO_THRESHOLD = 20

PERSONA_DISPLAY = {
    "baseline": "base",
    "good_persona": "good",
    "bad_persona": "bad",
}

PRINCIPLE_DISPLAY = {
    "design-for-equity-and-inclusion": "Equity & Inclusion",
    "protect-dignity-and-safety": "Protect Dignity",
    "enhance-human-capabilities": "Enhance Capabilities",
    "foster-healthy-relationships": "Foster Relationships",
    "enable-meaningful-choices": "Enable Choices",
    "respect-user-attention": "Respect Attention",
    "prioritize-long-term-wellbeing": "Prioritize Wellbeing",
    "be-transparent-and-honest": "Be Transparent",
}


def compute_table5(input_csv: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    df["is_vp"] = df["vulnerable_population"].notna() & (
        df["vulnerable_population"] != ""
    )

    vp_counts = (
        df.loc[df["is_vp"], ["principle", "sample_id"]]
        .drop_duplicates()
        .groupby("principle")
        .size()
    )

    qualifying = vp_counts[vp_counts >= VP_SCENARIO_THRESHOLD].index.tolist()
    excluded = vp_counts[vp_counts < VP_SCENARIO_THRESHOLD]

    print(f"VP scenarios per principle:")
    for p in sorted(vp_counts.index):
        tag = " ✓" if p in qualifying else " (excluded, < 20)"
        print(f"  {p}: {vp_counts[p]}{tag}")
    print()

    if excluded.size > 0:
        print(f"Excluded principles (n_VP < {VP_SCENARIO_THRESHOLD}):")
        for p, n in sorted(excluded.items()):
            print(f"  {p}: {n}")
        print()

    all_scenario_counts = (
        df[["principle", "sample_id"]].drop_duplicates().groupby("principle").size()
    )
    vp_only_principles = []
    for p in qualifying:
        if vp_counts[p] == all_scenario_counts.get(p, 0):
            vp_only_principles.append(p)

    if vp_only_principles:
        print("Principles where 100% of scenarios are VP-tagged (no comparison possible):")
        for p in vp_only_principles:
            print(f"  {p}: {vp_counts[p]}/{all_scenario_counts[p]} scenarios are VP-tagged")
        print()

    comparable = [p for p in qualifying if p not in vp_only_principles]

    rows = []
    for principle in comparable:
        pdf = df[df["principle"] == principle]
        for persona in ["baseline", "good_persona", "bad_persona"]:
            cond = pdf[pdf["persona"] == persona]
            vp_mean = cond.loc[cond["is_vp"], "score"].mean()
            non_vp_mean = cond.loc[~cond["is_vp"], "score"].mean()
            gap = vp_mean - non_vp_mean
            rows.append(
                {
                    "principle": principle,
                    "n_vp": int(vp_counts[principle]),
                    "persona": persona,
                    "vp_mean": round(vp_mean, 3),
                    "non_vp_mean": round(non_vp_mean, 3),
                    "gap": round(gap, 3),
                }
            )

    result = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "vp_table5.csv"
    result.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}\n")

    print("=" * 60)
    print("Table 5: Within-principle VP gap (VP minus non-VP mean)")
    print("=" * 60)
    pivot = result.pivot(index="principle", columns="persona", values="gap")
    pivot = pivot[["baseline", "good_persona", "bad_persona"]]

    n_vp_map = result.drop_duplicates("principle").set_index("principle")["n_vp"]
    for p in pivot.index:
        display = PRINCIPLE_DISPLAY.get(p, p)
        n = n_vp_map[p]
        vals = [f"{pivot.loc[p, c]:+.3f}" for c in pivot.columns]
        print(f"  {display} ({n}):  base={vals[0]}  good={vals[1]}  bad={vals[2]}")
    print()

    print("LaTeX fragment for Table 5:")
    print("-" * 60)
    for p in pivot.index:
        display = PRINCIPLE_DISPLAY.get(p, p)
        n = n_vp_map[p]
        vals = " & ".join(f"${pivot.loc[p, c]:+.2f}$" for c in pivot.columns)
        print(f"{display} ({n}) & {vals} \\\\")
    print("-" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute Table 5 VP gaps")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables" / "vp_sample_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tables",
    )
    args = parser.parse_args()
    compute_table5(args.input, args.output_dir)


if __name__ == "__main__":
    main()
