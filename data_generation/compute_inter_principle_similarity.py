#!/usr/bin/env python3
"""Compute the 8x8 inter-principle cosine similarity matrix for HumaneBench.

Extends the intra-principle analysis in qa_analysis.ipynb with cross-principle
stats, using identical methodology:
    - text-embedding-3-large
    - raw `input` field from data/humane_bench.jsonl
    - sklearn.metrics.pairwise.cosine_similarity on API-returned embeddings
    - intra = upper-triangle mean (self-pairs excluded)
    - inter = mean of full 100x100 cross block between two principles

On first run, generates embeddings via the OpenAI API and caches them to
data_generation/cache/principle_embeddings.npz. On subsequent runs the cache
is reused (no network required).

Outputs:
    data_generation/inter_principle_similarity.json
    data_generation/inter_principle_similarity.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATASET_PATH = REPO_ROOT / "data" / "humane_bench.jsonl"
CACHE_PATH = HERE / "cache" / "principle_embeddings.npz"
JSON_OUT = HERE / "inter_principle_similarity.json"
MD_OUT = HERE / "inter_principle_similarity.md"

# Same order as config.HUMANE_PRINCIPLES (intra loop order in the notebook).
# Short codes kept alongside for matrix headers.
PRINCIPLES: List[str] = [
    "respect-user-attention",
    "enable-meaningful-choices",
    "enhance-human-capabilities",
    "protect-dignity-and-safety",
    "foster-healthy-relationships",
    "prioritize-long-term-wellbeing",
    "be-transparent-and-honest",
    "design-for-equity-and-inclusion",
]
SHORT_CODES: Dict[str, str] = {
    "be-transparent-and-honest": "bath",
    "design-for-equity-and-inclusion": "dei",
    "enable-meaningful-choices": "emc",
    "enhance-human-capabilities": "ehc",
    "foster-healthy-relationships": "fhr",
    "prioritize-long-term-wellbeing": "pltw",
    "protect-dignity-and-safety": "pds",
    "respect-user-attention": "rua",
}
# Display order requested by user (alphabetical by short code).
DISPLAY_ORDER: List[str] = [
    "be-transparent-and-honest",      # bath
    "design-for-equity-and-inclusion", # dei
    "enable-meaningful-choices",       # emc
    "enhance-human-capabilities",      # ehc
    "foster-healthy-relationships",    # fhr
    "prioritize-long-term-wellbeing",  # pltw
    "protect-dignity-and-safety",      # pds
    "respect-user-attention",          # rua
]

# Notebook reference values (qa_analysis.ipynb cell 12 output), for drift check.
NOTEBOOK_INTRA: Dict[str, float] = {
    "respect-user-attention": 0.274,
    "foster-healthy-relationships": 0.267,
    "prioritize-long-term-wellbeing": 0.262,
    "be-transparent-and-honest": 0.255,
    "enhance-human-capabilities": 0.234,
    "protect-dignity-and-safety": 0.233,
    "enable-meaningful-choices": 0.203,
    "design-for-equity-and-inclusion": 0.177,
}
DRIFT_STOP_THRESHOLD = 0.005
DRIFT_WARN_THRESHOLD = 0.001


def load_grouped_inputs() -> Dict[str, List[str]]:
    """Load humane_bench.jsonl and group `input` strings by principle slug.

    Matches the notebook's sampling: 100 per principle, random_state=42. Since
    the dataset has exactly 100 items per principle, sampling 100 returns all
    of them (only the order differs). Order within a principle doesn't affect
    the mean similarity, so we skip the shuffle and use the natural file order.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    by_principle: Dict[str, List[str]] = {p: [] for p in PRINCIPLES}
    with DATASET_PATH.open("r") as f:
        for line in f:
            item = json.loads(line)
            target = item.get("target") or item.get("metadata", {}).get("principle")
            if target not in by_principle:
                raise ValueError(f"Unexpected principle slug: {target!r}")
            text = item.get("input")
            if not text:
                raise ValueError(f"Item missing 'input' field: id={item.get('id')}")
            by_principle[target].append(text)

    for p, texts in by_principle.items():
        if len(texts) != 100:
            raise ValueError(f"Principle {p} has {len(texts)} items, expected 100")
    return by_principle


def fetch_embeddings(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Embed a list of texts via OpenAI text-embedding-3-large.

    Matches the notebook's get_embeddings function exactly: batch_size=100,
    0.5s sleep between batches.
    """
    # Imports deferred so the script can still run against a cached .npz
    # without needing the openai package or the API key set.
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env or export it before running."
        )
    client = OpenAI(api_key=api_key)

    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model="text-embedding-3-large", input=batch)
        all_vecs.extend(item.embedding for item in resp.data)
        if i + batch_size < len(texts):
            time.sleep(0.5)
    return np.array(all_vecs)


def load_or_build_cache(by_principle: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Return {principle: (100, 3072) embedding array}. Uses cache if present."""
    if CACHE_PATH.exists():
        print(f"[cache] loading {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        missing = [p for p in PRINCIPLES if p not in data.files]
        if missing:
            raise RuntimeError(f"Cache missing principles: {missing}")
        return {p: data[p] for p in PRINCIPLES}

    print(f"[cache] not found — generating embeddings via OpenAI API")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    embeds: Dict[str, np.ndarray] = {}
    for p in PRINCIPLES:
        print(f"  embedding {p} ({len(by_principle[p])} items)")
        embeds[p] = fetch_embeddings(by_principle[p])
    np.savez(CACHE_PATH, **embeds)
    print(f"[cache] wrote {CACHE_PATH}")
    return embeds


def intra_mean(emb: np.ndarray) -> float:
    """Upper-triangle mean cosine similarity (self-pairs excluded)."""
    sim = cosine_similarity(emb)
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def inter_mean(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Mean of the full 100x100 cross-similarity block."""
    return float(cosine_similarity(emb_a, emb_b).mean())


def build_matrix(embeds: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(DISPLAY_ORDER)
    mat = np.zeros((n, n), dtype=float)
    for i, p_i in enumerate(DISPLAY_ORDER):
        mat[i, i] = intra_mean(embeds[p_i])
        for j in range(i + 1, n):
            p_j = DISPLAY_ORDER[j]
            m = inter_mean(embeds[p_i], embeds[p_j])
            mat[i, j] = m
            mat[j, i] = m
    return mat


def check_diagonal_drift(mat: np.ndarray) -> List[str]:
    """Compare diagonal (computed intra) to notebook values. Return warnings."""
    warnings: List[str] = []
    for i, p in enumerate(DISPLAY_ORDER):
        observed = mat[i, i]
        expected = NOTEBOOK_INTRA[p]
        delta = abs(observed - expected)
        if delta > DRIFT_STOP_THRESHOLD:
            raise RuntimeError(
                f"Diagonal drift for {p}: observed {observed:.4f} vs notebook "
                f"{expected:.3f} (delta {delta:.4f} > {DRIFT_STOP_THRESHOLD})"
            )
        if delta > DRIFT_WARN_THRESHOLD:
            warnings.append(
                f"  {p}: observed {observed:.4f} vs notebook {expected:.3f} "
                f"(delta {delta:.4f})"
            )
    return warnings


def summarize(mat: np.ndarray) -> dict:
    n = len(DISPLAY_ORDER)
    diag = np.diag(mat)
    mean_intra = float(diag.mean())

    # Unique off-diagonal pairs (upper triangle, k=1)
    iu = np.triu_indices(n, k=1)
    off_vals = mat[iu]
    mean_inter = float(off_vals.mean())
    ratio = mean_intra / mean_inter if mean_inter else float("nan")

    pairs = []
    for (i, j), v in zip(zip(iu[0], iu[1]), off_vals):
        pairs.append(
            {
                "a": DISPLAY_ORDER[i],
                "b": DISPLAY_ORDER[j],
                "a_code": SHORT_CODES[DISPLAY_ORDER[i]],
                "b_code": SHORT_CODES[DISPLAY_ORDER[j]],
                "similarity": float(v),
            }
        )
    pairs_sorted = sorted(pairs, key=lambda x: x["similarity"], reverse=True)
    top3 = pairs_sorted[:3]
    bottom3 = pairs_sorted[-3:][::-1]  # lowest first

    # Nearest / farthest for each principle
    nearest: Dict[str, dict] = {}
    farthest: Dict[str, dict] = {}
    for i, p in enumerate(DISPLAY_ORDER):
        others = [(j, mat[i, j]) for j in range(n) if j != i]
        j_near, v_near = max(others, key=lambda t: t[1])
        j_far, v_far = min(others, key=lambda t: t[1])
        nearest[p] = {
            "other": DISPLAY_ORDER[j_near],
            "other_code": SHORT_CODES[DISPLAY_ORDER[j_near]],
            "similarity": float(v_near),
        }
        farthest[p] = {
            "other": DISPLAY_ORDER[j_far],
            "other_code": SHORT_CODES[DISPLAY_ORDER[j_far]],
            "similarity": float(v_far),
        }

    # bath / rua / ehc triangle
    def pair(a: str, b: str) -> float:
        return float(mat[DISPLAY_ORDER.index(a), DISPLAY_ORDER.index(b)])

    triangle = {
        "bath_rua": pair("be-transparent-and-honest", "respect-user-attention"),
        "bath_ehc": pair("be-transparent-and-honest", "enhance-human-capabilities"),
        "rua_ehc": pair("respect-user-attention", "enhance-human-capabilities"),
        "mean_inter_all": mean_inter,
        "bath_intra": float(mat[DISPLAY_ORDER.index("be-transparent-and-honest")][
            DISPLAY_ORDER.index("be-transparent-and-honest")
        ]),
        "rua_intra": float(mat[DISPLAY_ORDER.index("respect-user-attention")][
            DISPLAY_ORDER.index("respect-user-attention")
        ]),
        "ehc_intra": float(mat[DISPLAY_ORDER.index("enhance-human-capabilities")][
            DISPLAY_ORDER.index("enhance-human-capabilities")
        ]),
    }

    return {
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
        "intra_inter_ratio": ratio,
        "top3_pairs": top3,
        "bottom3_pairs": bottom3,
        "nearest_by_principle": nearest,
        "farthest_by_principle": farthest,
        "bath_rua_ehc_triangle": triangle,
    }


def write_json(mat: np.ndarray, summary: dict) -> None:
    payload = {
        "embedding_model": "text-embedding-3-large",
        "methodology": (
            "Diagonal = upper-triangle mean cosine similarity within each "
            "principle (self-pairs excluded), matching qa_analysis.ipynb. "
            "Off-diagonal = mean of the full 100x100 cross-similarity block "
            "between two principles. Embeddings from OpenAI "
            "text-embedding-3-large on the raw `input` field of "
            "data/humane_bench.jsonl (100 items per principle)."
        ),
        "principle_order": [SHORT_CODES[p] for p in DISPLAY_ORDER],
        "principle_order_full": list(DISPLAY_ORDER),
        "matrix": [[round(float(v), 4) for v in row] for row in mat],
        "summary": summary,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[write] {JSON_OUT}")


def write_markdown(mat: np.ndarray, summary: dict, warnings: List[str]) -> None:
    codes = [SHORT_CODES[p] for p in DISPLAY_ORDER]
    header = "| | " + " | ".join(codes) + " |"
    sep = "|---|" + "|".join(["---"] * len(codes)) + "|"
    rows = []
    for i, code in enumerate(codes):
        cells = [f"{mat[i, j]:.3f}" for j in range(len(codes))]
        # Bold the diagonal
        cells[i] = f"**{cells[i]}**"
        rows.append(f"| **{code}** | " + " | ".join(cells) + " |")

    top3 = summary["top3_pairs"]
    bot3 = summary["bottom3_pairs"]
    tri = summary["bath_rua_ehc_triangle"]

    def pair_line(p):
        return f"{p['a_code']} ↔ {p['b_code']}: {p['similarity']:.3f}"

    near_lines = []
    far_lines = []
    for p in DISPLAY_ORDER:
        code = SHORT_CODES[p]
        n = summary["nearest_by_principle"][p]
        f = summary["farthest_by_principle"][p]
        near_lines.append(f"- **{code}** nearest: {n['other_code']} ({n['similarity']:.3f})")
        far_lines.append(f"- **{code}** farthest: {f['other_code']} ({f['similarity']:.3f})")

    drift_section = ""
    if warnings:
        drift_section = (
            "\n### Diagonal drift vs. notebook intra values\n\n"
            + "\n".join(warnings)
            + f"\n\n(Drift within ±{DRIFT_STOP_THRESHOLD} tolerance; "
            "likely due to non-determinism in OpenAI embeddings.)\n"
        )

    content = f"""# Inter-principle cosine similarity matrix

**Embedding model:** `text-embedding-3-large` (OpenAI)
**Source:** `data/humane_bench.jsonl` (100 items per principle, 800 total)
**Methodology:** diagonal = upper-triangle mean cosine similarity within each
principle (self-pairs excluded, matches `qa_analysis.ipynb`); off-diagonal =
mean of the full 100×100 cross-similarity block between two principles.

## 8×8 matrix

{header}
{sep}
{chr(10).join(rows)}

Short codes: `bath` = be-transparent-and-honest, `dei` = design-for-equity-and-inclusion,
`emc` = enable-meaningful-choices, `ehc` = enhance-human-capabilities,
`fhr` = foster-healthy-relationships, `pltw` = prioritize-long-term-wellbeing,
`pds` = protect-dignity-and-safety, `rua` = respect-user-attention.

## Summary stats

- **Overall mean intra** (diagonal): {summary['mean_intra']:.3f}
- **Overall mean inter** (off-diagonal, 28 unique pairs): {summary['mean_inter']:.3f}
- **Intra/inter ratio:** {summary['intra_inter_ratio']:.2f}

### Top 3 highest inter-principle pairs
1. {pair_line(top3[0])}
2. {pair_line(top3[1])}
3. {pair_line(top3[2])}

### Bottom 3 lowest inter-principle pairs
1. {pair_line(bot3[0])}
2. {pair_line(bot3[1])}
3. {pair_line(bot3[2])}

### Nearest other principle (per principle)

{chr(10).join(near_lines)}

### Farthest other principle (per principle)

{chr(10).join(far_lines)}

## bath / rua / ehc triangle

- sim(bath, rua) = **{tri['bath_rua']:.3f}**
- sim(bath, ehc) = **{tri['bath_ehc']:.3f}**
- sim(rua, ehc) = **{tri['rua_ehc']:.3f}**
- Corpus mean inter (all 28 pairs) = {tri['mean_inter_all']:.3f}
- Intra: bath = {tri['bath_intra']:.3f}, rua = {tri['rua_intra']:.3f}, ehc = {tri['ehc_intra']:.3f}
{drift_section}"""
    MD_OUT.write_text(content)
    print(f"[write] {MD_OUT}")


def main() -> int:
    print(f"[load] {DATASET_PATH}")
    by_principle = load_grouped_inputs()
    embeds = load_or_build_cache(by_principle)

    print("[compute] building 8x8 matrix")
    mat = build_matrix(embeds)

    print("[check] diagonal drift vs notebook intra values")
    warnings = check_diagonal_drift(mat)
    if warnings:
        print("  drift within tolerance:")
        for w in warnings:
            print(w)
    else:
        print("  all diagonal values within ±{:.3f} of notebook".format(DRIFT_WARN_THRESHOLD))

    summary = summarize(mat)

    # Stdout report
    codes = [SHORT_CODES[p] for p in DISPLAY_ORDER]
    print("\n=== MATRIX ===")
    print("       " + "  ".join(f"{c:>6}" for c in codes))
    for i, code in enumerate(codes):
        cells = "  ".join(f"{mat[i, j]:6.3f}" for j in range(len(codes)))
        print(f"{code:>6} {cells}")

    print(f"\nmean_intra = {summary['mean_intra']:.3f}")
    print(f"mean_inter = {summary['mean_inter']:.3f}")
    print(f"ratio      = {summary['intra_inter_ratio']:.2f}")

    print("\nTop 3 inter pairs:")
    for p in summary["top3_pairs"]:
        print(f"  {p['a_code']:>4} ↔ {p['b_code']:<4} {p['similarity']:.3f}")
    print("Bottom 3 inter pairs:")
    for p in summary["bottom3_pairs"]:
        print(f"  {p['a_code']:>4} ↔ {p['b_code']:<4} {p['similarity']:.3f}")

    tri = summary["bath_rua_ehc_triangle"]
    print("\nbath/rua/ehc triangle:")
    print(f"  bath ↔ rua = {tri['bath_rua']:.3f}")
    print(f"  bath ↔ ehc = {tri['bath_ehc']:.3f}")
    print(f"  rua  ↔ ehc = {tri['rua_ehc']:.3f}")
    print(f"  corpus mean inter = {tri['mean_inter_all']:.3f}")

    write_json(mat, summary)
    write_markdown(mat, summary, warnings)
    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
