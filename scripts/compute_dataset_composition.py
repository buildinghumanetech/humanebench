#!/usr/bin/env python3
"""Domain x principle composition, and how badly the two are confounded.

Section 4.4 reports per-principle HumaneScores and ranks principles by how far
they fall under the bad persona. That ranking is only a statement about
*principles* if principles are not systematically bound to particular topical
domains. They are: Be Transparent and Honest is overwhelmingly technology-use,
Enhance Human Capabilities is overwhelmingly education, and so on. Because each
scenario carries exactly one principle and one domain, and the pairing is far
from uniform, principle effects and domain effects are not separable in this
design at all -- there is no cell structure that would let a model estimate
them independently.

This script quantifies that rather than asserting it:

  - the full 8 x 12 crosstab, on the 788-scenario analysis set and on all 800;
  - chi-square and Cramer's V for the principle-domain association;
  - per-principle domain concentration (top-1 share and a normalized
    Herfindahl index), which is the number that says how much of a
    "principle effect" is carried by a single domain;
  - domains too thin to support any domain-level claim;
  - the VP x principle crosstab and the exact VP-tagged count on the 788 set,
    which section 4.5 currently reports as 268.

Inputs (read-only):
  - data/humane_bench.jsonl

Outputs (written to --output-dir, default tables/):
  - domain_principle_crosstab_788.csv
  - domain_principle_crosstab_800.csv
  - vp_principle_crosstab_788.csv
  - principle_domain_concentration.csv
  - dataset_composition.md

Run from repo root:
    python scripts/compute_dataset_composition.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import PRINCIPLES  # noqa: E402

# Below this a domain cell cannot support a domain-level claim at all.
THIN_DOMAIN = 10


def load_dataset(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            md = r.get("metadata") or {}
            rows.append({
                "id": r["id"],
                "principle": r.get("target") or md.get("principle"),
                "domain": md.get("domain") or "(unset)",
                "vp": md.get("vulnerable-population") or "",
                "excluded": bool(md.get("excluded_from_analysis")),
                "input": r["input"],
            })
    return pd.DataFrame(rows)


def crosstab(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ct = pd.crosstab(df["principle"], df[col])
    order = [p for p in PRINCIPLES if p in ct.index]
    ct = ct.loc[order]
    ct["TOTAL"] = ct.sum(axis=1)
    ct.loc["TOTAL"] = ct.sum(axis=0)
    return ct


def cramers_v(ct: pd.DataFrame) -> tuple[float, float, float, int]:
    """Cramer's V with the chi-square it comes from. `ct` must exclude totals."""
    chi2, p, dof, _ = sp.chi2_contingency(ct.to_numpy())
    n = ct.to_numpy().sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n and k else float("nan")
    return float(chi2), float(p), v, int(dof)


def concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Per-principle domain concentration.

    `top1_share` is the fraction of a principle's scenarios sitting in its
    single most common domain. `hhi_normalized` is the Herfindahl index of the
    domain shares rescaled so 0 = perfectly even across the domains present and
    1 = every scenario in one domain; it summarizes the whole distribution
    rather than just its mode.
    """
    rows = []
    n_domains = df["domain"].nunique()
    for principle, sub in df.groupby("principle"):
        shares = sub["domain"].value_counts(normalize=True)
        hhi = float((shares**2).sum())
        floor = 1.0 / n_domains
        rows.append({
            "principle": principle,
            "n": len(sub),
            "n_domains_present": int(shares.size),
            "top_domain": shares.index[0],
            "top1_share": float(shares.iloc[0]),
            "top1_n": int(sub["domain"].value_counts().iloc[0]),
            "hhi": hhi,
            "hhi_normalized": float((hhi - floor) / (1 - floor)),
        })
    out = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(PRINCIPLES)}
    return out.sort_values("principle", key=lambda s: s.map(order)).reset_index(drop=True)


def write_report(out: Path, df788: pd.DataFrame, df800: pd.DataFrame,
                 ct788: pd.DataFrame, conc: pd.DataFrame, vp_ct: pd.DataFrame) -> None:
    core = ct788.drop(index="TOTAL", columns="TOTAL")
    chi2, p, v, dof = cramers_v(core)

    L = ["# Dataset composition: domain x principle\n"]
    L.append(
        f"Computed on the **{len(df788)}-scenario analysis set** "
        f"(all {len(df800)} reported separately). Each scenario carries exactly "
        "one principle and one domain.\n"
    )

    L.append("## Principle-domain association\n")
    L.append("| statistic | value |")
    L.append("| --- | ---: |")
    L.append(f"| chi-square ({dof} df) | {chi2:.1f} |")
    L.append(f"| p | {'< 1e-300' if p == 0 else f'{p:.3g}'} |")
    L.append(f"| Cramer's V | {v:.3f} |")
    L.append("")
    L.append(
        f"Cramer's V = **{v:.3f}** on a 0-1 scale. Principle and domain are "
        "strongly associated, which is by construction: the generation pipeline "
        "steered each principle toward the domains where it naturally arises.\n"
    )

    L.append("## Per-principle domain concentration\n")
    L.append("| principle | n | domains | top domain | top-1 share | normalized HHI |")
    L.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for _, r in conc.iterrows():
        L.append(f"| {r.principle} | {r.n} | {r.n_domains_present} | "
                 f"{r.top_domain} ({r.top1_n}) | {r.top1_share:.1%} | "
                 f"{r.hhi_normalized:.3f} |")
    L.append("")

    worst = conc.sort_values("top1_share", ascending=False).head(4)
    bullet = "; ".join(
        f"{r.principle} is {r.top1_share:.0%} {r.top_domain}" for _, r in worst.iterrows()
    )
    L.append(f"The four most concentrated principles: {bullet}.\n")

    L.append("## Thin domains\n")
    dom_tot = df788["domain"].value_counts()
    thin = dom_tot[dom_tot < THIN_DOMAIN]
    L.append(f"| domain | n |")
    L.append("| --- | ---: |")
    for d, n in dom_tot.items():
        mark = " **(thin)**" if n < THIN_DOMAIN else ""
        L.append(f"| {d}{mark} | {n} |")
    L.append("")
    if len(thin):
        L.append(f"**{len(thin)} domains carry fewer than {THIN_DOMAIN} scenarios** "
                 f"({', '.join(f'{d} = {n}' for d, n in thin.items())}) and cannot "
                 "support a domain-level claim of any kind.\n")

    L.append("## What this means for the per-principle results\n")
    L.append(
        "Principle and domain are not separable in this design. Every scenario "
        "has exactly one of each, the assignment is far from uniform "
        f"(Cramer's V = {v:.3f}), and several principles draw the majority of "
        "their scenarios from a single domain. There is no cell structure that "
        "would let a model estimate a principle effect holding domain fixed.\n"
    )
    L.append(
        "The consequence for section 4.4 is specific and should be stated rather "
        "than hedged: **per-principle scores describe principle-domain bundles, "
        "not principles.** A claim that a given principle is the weakest is a "
        "claim about that principle *as instantiated in its domain mix*, and a "
        "different domain mix could reorder the table. This is a limitation of "
        "the construction, not an error in the numbers, and no reweighting "
        "available post hoc can remove it -- the confound is structural, not "
        "statistical.\n"
    )

    L.append("## Vulnerable-population tags\n")
    n_vp788 = int((df788["vp"] != "").sum())
    n_vp800 = int((df800["vp"] != "").sum())
    n_groups = df788.loc[df788["vp"] != "", "vp"].nunique()
    L.append(f"| set | VP-tagged | total | groups |")
    L.append("| --- | ---: | ---: | ---: |")
    L.append(f"| 788 analysis set | **{n_vp788}** | {len(df788)} | {n_groups} |")
    L.append(f"| all 800 | {n_vp800} | {len(df800)} | "
             f"{df800.loc[df800['vp'] != '', 'vp'].nunique()} |")
    L.append("")
    L.append(
        f"Section 4.5 states \"268 of the 788 scenarios in our dataset carry VP "
        f"tags spanning 17 population groups\". The analysis set actually has "
        f"**{n_vp788}** VP-tagged scenarios across **{n_groups}** groups.\n"
    )
    excl_vp = int(((df800["vp"] != "") & df800["excluded"]).sum())
    L.append(
        f"The 800-set count is {n_vp800}; {excl_vp} of the 12 excluded scenarios "
        f"carry a VP tag, giving {n_vp800} - {excl_vp} = {n_vp800 - excl_vp} on "
        "the analysis set.\n"
    )
    L.append("Full VP x principle crosstab: `vp_principle_crosstab_788.csv`.\n")

    L.append("## Crosstab (788 analysis set)\n")
    cols = list(core.columns)
    L.append("| principle | " + " | ".join(cols) + " | total |")
    L.append("| --- |" + " ---: |" * (len(cols) + 1))
    for pr in core.index:
        L.append(f"| {pr} | " + " | ".join(str(core.loc[pr, c]) for c in cols)
                 + f" | {int(core.loc[pr].sum())} |")
    L.append("| **total** | " + " | ".join(str(int(core[c].sum())) for c in cols)
             + f" | {int(core.to_numpy().sum())} |")
    L.append("")

    out.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path,
                    default=REPO_ROOT / "data" / "humane_bench.jsonl")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df800 = load_dataset(args.dataset)
    df788 = df800[~df800["excluded"]].reset_index(drop=True)
    print(f"Loaded {len(df800)} scenarios; {len(df788)} in the analysis set.")

    ct788 = crosstab(df788, "domain")
    ct800 = crosstab(df800, "domain")
    conc = concentration(df788)
    vp = df788[df788["vp"] != ""]
    vp_ct = pd.crosstab(vp["principle"], vp["vp"])

    core = ct788.drop(index="TOTAL", columns="TOTAL")
    chi2, p, v, dof = cramers_v(core)
    print(f"  chi2({dof}) = {chi2:.1f}, p = {p:.3g}, Cramer's V = {v:.3f}")
    print(f"  VP-tagged on 788 set: {(df788['vp'] != '').sum()} "
          f"across {vp['vp'].nunique()} groups")
    print("  most concentrated principles:")
    for _, r in conc.sort_values("top1_share", ascending=False).head(4).iterrows():
        print(f"    {r.principle:32} {r.top1_share:5.1%} {r.top_domain}")

    ct788.to_csv(args.output_dir / "domain_principle_crosstab_788.csv")
    ct800.to_csv(args.output_dir / "domain_principle_crosstab_800.csv")
    vp_ct.to_csv(args.output_dir / "vp_principle_crosstab_788.csv")
    conc.to_csv(args.output_dir / "principle_domain_concentration.csv", index=False)
    write_report(args.output_dir / "dataset_composition.md", df788, df800,
                 ct788, conc, vp_ct)
    print(f"\nWrote {args.output_dir / 'dataset_composition.md'}")


if __name__ == "__main__":
    main()
