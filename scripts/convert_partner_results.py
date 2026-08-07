"""Convert a partner production-results JSONL to HumaneBench re-judge format.

Each input row (one conversation turn with a stored assistant response and
per-principle judgments) fans out to one sample per principle:

    {"id": "<sample_id>__<principle-slug>", "input": <user_message>,
     "target": <principle-slug>,
     "metadata": {"ai_output": <assistant_response>, ...}}

`metadata.ai_output` is the contract key read by
src/pregenerated_solver.py:use_pregenerated_output_strict(); the original
judgments and curation tags ride along in metadata for post-run comparison and
are never shown to judges (the overseer only sees input + ai_output).

Usage:
    python scripts/convert_partner_results.py \
        --input <curated.jsonl> --output <hb_rejudge.jsonl> \
        [--principles {all,relevant}]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humanebench.humane_patterns import HUMANE_PATTERNS

KNOWN_SLUGS = frozenset(HUMANE_PATTERNS.keys())


def convert_row(row: dict, principles_mode: str) -> list[dict]:
    unknown = set(row["principles"]) - KNOWN_SLUGS
    if unknown:
        raise ValueError(
            f"sample {row['sample_id']}: unknown principle slug(s) {sorted(unknown)}"
        )

    if principles_mode == "relevant":
        slugs = [s for s in row.get("relevant_principles", []) if s in KNOWN_SLUGS]
    else:
        slugs = sorted(row["principles"])

    samples = []
    for slug in slugs:
        samples.append(
            {
                "id": f"{row['sample_id']}__{slug}",
                "input": row["user_message"],
                "target": slug,
                "metadata": {
                    "ai_output": row["assistant_response"],
                    "conv": row.get("conv"),
                    "turn_index": row.get("turn_index"),
                    "ts": row.get("ts"),
                    "curation": row.get("curation"),
                    "orig_judgment": row["principles"].get(slug),
                    "orig_overall_severity": row.get("overall_severity"),
                    "orig_mean_severity": row.get("mean_severity"),
                    "audit": (row.get("audit") or {}).get("principles", {}).get(slug),
                },
            }
        )
    return samples


def split_sample_id(hb_id: str) -> tuple[str, str]:
    """Recover (sample_id, slug) from an HB fan-out id; safe for ids containing '__'."""
    sample_id, slug = hb_id.rsplit("__", 1)
    return sample_id, slug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--principles", choices=["all", "relevant"], default="all")
    args = parser.parse_args()

    n_rows = 0
    n_samples = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            row = json.loads(line)
            n_rows += 1
            for sample in convert_row(row, args.principles):
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_samples += 1

    print(
        f"Converted {n_rows} turns -> {n_samples} samples "
        f"(mode={args.principles}) into {args.output}"
    )


if __name__ == "__main__":
    main()
