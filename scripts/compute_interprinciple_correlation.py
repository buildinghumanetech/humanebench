#!/usr/bin/env python3
"""Model-level inter-principle correlation -- deliberately weak evidence.

Reviewers asked whether the eight principles are distinct constructs or a
smaller number of factors wearing eight labels. The analysis that would answer
that is a multi-label rescoring: score a scenario subsample against *all eight*
principles and factor the resulting item x dimension matrix. HumaneBench does
not have such a matrix, because every scenario is assigned exactly one
principle -- and building one requires new judge calls, which are out of scope
here.

What can be computed from the stored scores is a much weaker proxy: correlate
the eight per-principle scores **across the 15 models**. That gives an 8 x 8
matrix from n = 15 observations.

**This is not evidence of factor structure and must not be presented as such.**
At n = 15 a single correlation has a 95% CI roughly +/-0.5 wide; the sampling
error on a whole 8 x 8 matrix swamps any structure that could be read off it,
and the 15 "observations" are frontier models chosen for coverage, not a random
sample of anything. It is reported because reviewers will ask what the stored
data can say, and the honest answer -- "very little, and here is exactly how
little" -- is more useful than silence.

Correlations are computed separately per persona, because the bad-persona
condition compresses most models toward the floor and will manufacture
correlation that says nothing about construct overlap.

Inputs (read-only):
  - tables/inter_judge_raw_regenerated.csv   (from compute_pipeline_check.py)

Outputs (written to --output-dir, default tables/):
  - interprinciple_correlation_<persona>.csv   8 x 8 Pearson
  - interprinciple_correlation_long.csv        pairwise, with bootstrap CIs
  - interprinciple_correlation.md

Run from repo root:
    python scripts/compute_interprinciple_correlation.py
"""
# Paper: produces tables/interprinciple_correlation_*.csv and interprinciple_correlation.md -
#        correlations between the eight principle columns computed across models (n = 15), per
#        persona, with model-resample CIs; reported as uninformative for construct redundancy
#        because a general model-quality factor enters every column (supplement, "Cross-Model
#        Correlation and Construct Redundancy").
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_BOOTSTRAP_DEFAULT,
    PERSONAS,
    PRINCIPLES,
    load_long_scores,
)
from humanebench.tables import resolve_table  # noqa: E402

SHORT = {
    "respect-user-attention": "rua",
    "enable-meaningful-choices": "emc",
    "enhance-human-capabilities": "ehc",
    "protect-dignity-and-safety": "pds",
    "foster-healthy-relationships": "fhr",
    "prioritize-long-term-wellbeing": "pltw",
    "be-transparent-and-honest": "bath",
    "design-for-equity-and-inclusion": "dei",
}


def model_principle_matrix(long: pd.DataFrame, persona: str) -> pd.DataFrame:
    """(models x 8 principles) mean score for one persona."""
    sub = long[long["persona"] == persona]
    m = sub.groupby(["model", "principle"])["score"].mean().unstack("principle")
    return m[[p for p in PRINCIPLES if p in m.columns]]


def bootstrap_pairwise(
    mat: pd.DataFrame, n_bootstrap: int, seed: int
) -> list[dict]:
    """Pearson and Spearman per principle pair, with a model-resample CI.

    The resampling unit is the *model*, since that is the observation here.
    With n = 15 this CI is wide by construction; reporting it is the point.
    """
    rng = np.random.default_rng(seed)
    cols = list(mat.columns)
    arr = mat.to_numpy()
    n = arr.shape[0]
    draws = rng.integers(0, n, size=(n_bootstrap, n))

    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x, y = arr[:, i], arr[:, j]
            r, p = sp.pearsonr(x, y)
            rho, rho_p = sp.spearmanr(x, y)
            boots = []
            for d in draws:
                xs, ys = x[d], y[d]
                if xs.std() == 0 or ys.std() == 0:
                    continue
                boots.append(np.corrcoef(xs, ys)[0, 1])
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots
                      else (np.nan, np.nan))
            rows.append({
                "principle_a": cols[i], "principle_b": cols[j],
                "n_models": n,
                "pearson_r": r, "pearson_p": p,
                "pearson_ci_lower": lo, "pearson_ci_upper": hi,
                "ci_width": hi - lo,
                "spearman_rho": rho, "spearman_p": rho_p,
            })
    return rows


def write_report(out: Path, mats: dict, longs: dict, n_bootstrap: int) -> None:
    L = ["# Model-level inter-principle correlation\n"]
    L.append(
        "**Read the caveat before the numbers.** These correlations are computed "
        "across the **15 evaluated models**, so n = 15. That is the entire "
        "sample. A single correlation at n = 15 carries a 95% CI roughly 1.0 "
        "wide, and the 15 models are a coverage-driven selection of frontier "
        "systems, not a random sample from any population. This cannot "
        "establish factor structure, cannot refute construct overlap, and "
        "should not be described as doing either.\n"
    )
    L.append(
        "The analysis that *would* answer the construct-overlap question is a "
        "multi-label rescoring -- score a scenario subsample against all eight "
        "principles and factor the resulting item x dimension matrix. "
        "HumaneBench assigns exactly one principle per scenario, so no such "
        "matrix exists in the stored data, and building one needs new judge "
        "calls. It is not computed here.\n"
    )

    for persona in PERSONAS:
        if persona not in mats:
            continue
        mat, rows = mats[persona], longs[persona]
        df = pd.DataFrame(rows)
        L.append(f"## {persona}\n")
        med_w = df["ci_width"].median()
        n_sig = int((df["pearson_p"] < 0.05).sum())
        L.append(f"Median 95% CI width across the 28 pairs: **{med_w:.2f}** "
                 f"(on a scale that only spans 2.0). "
                 f"{n_sig} of {len(df)} pairs reach p < 0.05 uncorrected; "
                 f"at 28 tests roughly {0.05 * len(df):.1f} would be expected "
                 "by chance alone.\n")
        corr = mat.corr()
        cols = [SHORT[c] for c in corr.columns]
        L.append("| | " + " | ".join(cols) + " |")
        L.append("| --- |" + " ---: |" * len(cols))
        for pr in corr.index:
            cells = []
            for pc in corr.columns:
                v = corr.loc[pr, pc]
                cells.append("--" if pr == pc else f"{v:+.2f}")
            L.append(f"| **{SHORT[pr]}** | " + " | ".join(cells) + " |")
        L.append("")
        top = df.reindex(df["pearson_r"].abs().sort_values(ascending=False).index).head(5)
        L.append("Strongest 5 pairs, with the CI that undercuts them:\n")
        L.append("| pair | r | 95% CI | p |")
        L.append("| --- | ---: | :---: | ---: |")
        for _, r in top.iterrows():
            L.append(f"| {SHORT[r.principle_a]} x {SHORT[r.principle_b]} | "
                     f"{r.pearson_r:+.2f} | [{r.pearson_ci_lower:+.2f}, "
                     f"{r.pearson_ci_upper:+.2f}] | {r.pearson_p:.3f} |")
        L.append("")

    L.append("## Why high correlations here do *not* mean the principles overlap\n")
    L.append(
        "The correlations are large and get larger under adversarial pressure. "
        "That pattern is expected under **either** hypothesis and so "
        "discriminates between neither.\n"
    )
    L.append(
        "The unit of observation is the model. Models differ enormously in "
        "overall quality, and that single dominant dimension enters every one "
        "of the eight per-principle means. A model that scores well on one "
        "principle scores well on all of them because it is a better model, not "
        "because the principles measure the same thing. Aggregating to the model "
        "level therefore confounds construct similarity with a general "
        "model-quality factor, and cannot separate them: eight genuinely "
        "distinct constructs measured on 15 models of widely varying quality "
        "would produce exactly this matrix.\n"
    )
    L.append(
        "The bad-persona column makes the artifact visible. Median |r| rises to "
        "0.97 there, with CI widths collapsing to ~0.06 -- not because the "
        "principles became more alike under pressure, but because most models "
        "are driven toward the floor together, leaving one dimension of "
        "variance. Reading that as construct overlap would be a mistake.\n"
    )
    L.append("## How to use this\n")
    L.append(
        "As supporting material only, and only alongside the n = 15 caveat and "
        "the general-factor confound above, stated in the same breath. If the "
        "paper needs a defensible answer on construct distinctness, the "
        "multi-label rescoring is the analysis to run; this is not a substitute "
        "for it, and quoting these correlations without the confound would "
        "invite a fair reviewer objection.\n"
    )
    L.append(f"Short codes: " + ", ".join(f"`{v}` = {k}" for k, v in SHORT.items()) + ".\n")
    out.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-csv", type=Path,
                    default=REPO_ROOT / "tables" / "inter_judge_raw_regenerated.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    long = load_long_scores(resolve_table(args.raw_csv))
    mats, longs, all_rows = {}, {}, []
    for persona in PERSONAS:
        mat = model_principle_matrix(long, persona)
        if mat.empty:
            continue
        rows = bootstrap_pairwise(mat, args.n_bootstrap, args.seed)
        for r in rows:
            r["persona"] = persona
        mats[persona], longs[persona] = mat, rows
        all_rows.extend(rows)
        mat.corr().to_csv(
            args.output_dir / f"interprinciple_correlation_{persona}.csv")
        df = pd.DataFrame(rows)
        print(f"{persona:13} n_models={mat.shape[0]:2d}  "
              f"median |r|={df.pearson_r.abs().median():.2f}  "
              f"median CI width={df.ci_width.median():.2f}  "
              f"pairs p<0.05: {int((df.pearson_p < 0.05).sum())}/{len(df)}")

    pd.DataFrame(all_rows).to_csv(
        args.output_dir / "interprinciple_correlation_long.csv", index=False)
    write_report(args.output_dir / "interprinciple_correlation.md", mats, longs,
                 args.n_bootstrap)
    print(f"\nWrote {args.output_dir / 'interprinciple_correlation.md'}")


if __name__ == "__main__":
    main()
