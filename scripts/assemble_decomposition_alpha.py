#!/usr/bin/env python3
"""Assemble the per-condition inter-judge agreement table for the decomposition.

`compute_inter_judge_agreement.py` writes fixed filenames, so each decomposition
condition is scanned into its own directory:

    python scripts/compute_inter_judge_agreement.py \\
        --personas decomp_b_xml_objective \\
        --tables-dir tables/decomposition/alpha_decomp_b_xml_objective

This script collects those directories into one table. It exists instead of a
hand-written markdown file for one reason: every number in it is read from the
CSV that produced it, so the prose cannot drift from the data the way a
transcribed table can.

**What this table is not.** Each condition's alpha is conditioned on that
condition alone. The four are never pooled with each other, and none of them is
pooled with the published figures -- not the alpha, not the design effects, not
the 35,416-item count, not the judge self-preference DiD. The published pooled
alpha (0.706) is computed across baseline + good_persona + bad_persona together
and is *arithmetically* higher than any single condition's alpha for a reason
that has nothing to do with judge quality: pooling conditions widens the score
range, which enlarges alpha's expected-disagreement denominator. The only
like-for-like reference is therefore the published *per-persona* breakdown,
which is reproduced here from `tables/inter_judge_agreement_by_persona.csv` and
is still not a controlled comparison -- it covers fifteen models where the
decomposition covers eleven.

Run from repo root:
    python scripts/assemble_decomposition_alpha.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench import decomposition as dc  # noqa: E402

# The published headline CI: scenarios are the honest cluster unit, because one
# prompt's response text drives every judge's score for that prompt.
PRIMARY_SPEC = "cluster_input_id"

# Coverage below this is flagged as an incomplete scan. Matches the launcher's
# and the manifest's admission gate, so one number governs everywhere.
COVERAGE_OK = 0.98


def read_condition(directory: Path, expected_items: int) -> dict | None:
    """Read one condition's agreement CSV, or None if it is not there.

    `expected_items` is what a complete scan of that condition would produce
    (models x analysable scenarios). It is carried through to the output rather
    than checked here, because a short scan is a caveat to publish, not an error
    to raise -- but the caveat must be published. Treating "the file exists" as
    "the condition is complete" would let an interrupted scan appear in the
    table as that condition's alpha with nothing marking it.
    """
    path = directory / "inter_judge_agreement.csv"
    if not path.is_file():
        return None
    row = pd.read_csv(path).iloc[0]
    n_items = int(row["n_items_included"])
    return {
        "expected_items": expected_items,
        "coverage": n_items / expected_items if expected_items else float("nan"),
        "n_items": int(row["n_items_included"]),
        "alpha_ord": float(row["alpha_ord"]),
        "alpha_ord_lo": float(row[f"alpha_ord_{PRIMARY_SPEC}_ci_lower"]),
        "alpha_ord_hi": float(row[f"alpha_ord_{PRIMARY_SPEC}_ci_upper"]),
        "alpha_bin": float(row["alpha_bin"]),
        "alpha_bin_lo": float(row[f"alpha_bin_{PRIMARY_SPEC}_ci_lower"]),
        "alpha_bin_hi": float(row[f"alpha_bin_{PRIMARY_SPEC}_ci_upper"]),
        "sign_disagreement": float(row["sign_disagreement_rate"]),
        "n_clusters": int(row[f"alpha_ord_{PRIMARY_SPEC}_n_clusters"]),
        "design_effect": float(row[f"alpha_ord_{PRIMARY_SPEC}_design_effect"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha-root", type=Path,
                    default=REPO_ROOT / "tables" / "decomposition")
    ap.add_argument("--by-persona-csv", type=Path,
                    default=REPO_ROOT / "tables"
                    / "inter_judge_agreement_by_persona.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "tables" / "decomposition"
                    / "alpha_by_condition.md")
    args = ap.parse_args()

    n_models = len(dc.MODELS)
    found, missing = {}, []
    for cond in dc.CONDITIONS:
        stats = read_condition(
            args.alpha_root / f"alpha_{cond.task_type}",
            expected_items=n_models * cond.expected_analysis_samples(),
        )
        if stats is None:
            missing.append(cond.label)
        else:
            found[cond.label] = stats
    if not found:
        raise SystemExit(
            f"no per-condition agreement directories under {args.alpha_root}; "
            "run compute_inter_judge_agreement.py --personas <condition> first"
        )

    L: list[str] = ["# Inter-judge agreement, by decomposition condition\n"]
    L.append(
        "Generated by `scripts/assemble_decomposition_alpha.py` from one "
        "`compute_inter_judge_agreement.py` run per condition. CIs are the "
        "primary published spec: cluster bootstrap on `input_id` (scenarios), "
        "1000 replicates.\n"
    )
    L.append(
        "**Each row stands alone.** These alphas are not pooled with each "
        "other and not pooled with any published figure. The published pooled "
        "α, design effects, 35,416-item count and judge self-preference DiD "
        "remain conditioned on baseline + good_persona + bad_persona and are "
        "unchanged by anything here.\n"
    )
    if missing:
        L.append(
            f"**Conditions absent from this table:** {', '.join(missing)} — no "
            "agreement directory on disk, so they are excluded rather than "
            "estimated.\n"
        )

    L.append("| Condition | n scored items | of expected | scenarios "
             "| α ordinal [95% CI] | α binary [95% CI] | sign-disagreement "
             "| design effect |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|")
    for label, s in found.items():
        L.append(
            f"| {label} | {s['n_items']:,} | {s['coverage']:.1%} | "
            f"{s['n_clusters']:,} | "
            f"{s['alpha_ord']:.3f} [{s['alpha_ord_lo']:.3f}, {s['alpha_ord_hi']:.3f}] | "
            f"{s['alpha_bin']:.3f} [{s['alpha_bin_lo']:.3f}, {s['alpha_bin_hi']:.3f}] | "
            f"{s['sign_disagreement']:.1%} | {s['design_effect']:.2f} |"
        )
    L.append("")
    L.append(
        "`of expected` is the scan's coverage of "
        f"{n_models} models × the condition's analysable scenarios. It is "
        "printed because the assembler reads whatever CSV it finds: without it, "
        "an interrupted scan would appear here as that condition's α with "
        "nothing to distinguish it from a complete one.\n"
    )
    short = [(k, s) for k, s in found.items() if s["coverage"] < COVERAGE_OK]
    if short:
        L.append(
            "**Incomplete scans — read these rows as provisional:** "
            + "; ".join(f"{k} at {s['coverage']:.1%}" for k, s in short)
            + f". Anything below {COVERAGE_OK:.0%} of expected is not a "
            "condition-level α.\n"
        )

    if args.by_persona_csv.is_file():
        ref = pd.read_csv(args.by_persona_csv)
        L.append("## Reference: the published per-persona breakdown\n")
        L.append(
            "Reproduced unchanged from `tables/inter_judge_agreement_by_persona"
            ".csv`. It is the only like-for-like reference, because it is also "
            "one condition at a time. It is **not** a controlled comparison: it "
            "covers the published fifteen models, the decomposition covers "
            "eleven, and no CIs were computed for the per-persona split.\n"
        )
        L.append("| Persona | n scored items | α ordinal | α binary | sign-disagreement |")
        L.append("|---|---:|---:|---:|---:|")
        for r in ref.to_dict("records"):
            L.append(
                f"| {r['persona']} | {int(r['n_items']):,} | "
                f"{r['alpha_ord']:.3f} | {r['alpha_bin']:.3f} | "
                f"{r['sign_disagreement_rate']:.1%} |"
            )
        L.append("")

        # No test is available here and the file must not imply one. The
        # reference rows carry no CI, and they cover fifteen models where these
        # cover eleven, so the only honest move is to place each condition's
        # INTERVAL against the reference points and say what that does and does
        # not settle. An earlier version compared point estimates alone and
        # concluded "between the two" while every interval overlapped baseline.
        lo = min(s["alpha_ord"] for s in found.values())
        hi = max(s["alpha_ord"] for s in found.values())
        by_persona = dict(zip(ref["persona"], ref["alpha_ord"]))
        base = by_persona.get("baseline")
        anchor = by_persona.get(dc.ANCHOR_PERSONA)
        if base is not None and anchor is not None:
            overlaps_base = [k for k, s in found.items()
                             if s["alpha_ord_lo"] <= base <= s["alpha_ord_hi"]]
            below_anchor = [k for k, s in found.items()
                            if s["alpha_ord_hi"] < anchor]
            L.append(
                f"The decomposition conditions run α_ordinal {lo:.3f}–{hi:.3f} "
                f"by point estimate, against {base:.3f} for baseline and "
                f"{anchor:.3f} for the adversarial persona.\n"
            )
            L.append(
                f"- **Against the adversarial persona:** "
                f"{len(below_anchor)} of {len(found)} conditions have an upper "
                f"CI bound below {anchor:.3f}. Judges agree markedly less on "
                "the objective-only arms than on the arm whose responses are "
                "flagrantly bad."
            )
            L.append(
                f"- **Against baseline:** {len(overlaps_base)} of {len(found)} "
                f"conditions have a CI that *contains* {base:.3f}. **These "
                "arms are not distinguishable from baseline on agreement**, so "
                "the tidy reading — that they sit strictly between the two — is "
                "not supported. What the intervals support is the weaker claim: "
                "at or near baseline agreement, and clearly below the "
                "adversarial arm."
            )
            L.append("")
            L.append(
                "Neither bullet is a test. The reference rows carry no interval "
                "at all and cover fifteen models against these eleven, so they "
                "are a backdrop, not a comparator. Read all of this as a "
                "description of the same attenuation the score contrasts "
                "measure, seen through a second channel — not as independent "
                "evidence for it.\n"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print(f"Wrote {args.out} ({len(found)} condition(s)"
          + (f", missing: {', '.join(missing)}" if missing else "") + ")")


if __name__ == "__main__":
    main()
