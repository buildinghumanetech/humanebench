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

--joint mode emits ONE sample per turn (for the joint-prompt task, which
scores all 8 principles in a single call): target is empty, the full original
per-principle judgment dict rides in metadata. --sample-turns N deterministically
subsamples turns (seeded) — used to pre-register the joint slice before any
panel results exist. Repeat-slice rows (__rep2/__rep3 ids) are excluded from
--joint output.

Usage:
    python scripts/convert_partner_results.py \
        --input <curated.jsonl> --output <hb_rejudge.jsonl> \
        [--principles {all,relevant}] \
        [--joint] [--sample-turns N] [--sample-seed 7]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humanebench.humane_patterns import HUMANE_PATTERNS

KNOWN_SLUGS = frozenset(HUMANE_PATTERNS.keys())


def has_judgeable_response(row: dict) -> bool:
    """The strict replay solver treats falsy ai_output as missing and raises,
    so only turns with a non-empty string response can be re-judged."""
    resp = row.get("assistant_response")
    return isinstance(resp, str) and bool(resp)


def convert_row(row: dict, principles_mode: str) -> list[dict]:
    unknown = set(row["principles"]) - KNOWN_SLUGS
    if unknown:
        raise ValueError(
            f"sample {row['sample_id']}: unknown principle slug(s) {sorted(unknown)}"
        )

    if principles_mode == "relevant":
        slugs = row.get("relevant_principles", [])
        unknown = set(slugs) - KNOWN_SLUGS
        if unknown:
            raise ValueError(
                f"sample {row['sample_id']}: unknown slug(s) in "
                f"relevant_principles {sorted(unknown)}"
            )
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


def convert_row_joint(row: dict) -> dict:
    """One sample per turn for the joint-prompt task (all 8 principles in one
    judge call). The joint scorer ignores target; original judgments ride in
    metadata for the comparison analysis."""
    unknown = set(row["principles"]) - KNOWN_SLUGS
    if unknown:
        raise ValueError(
            f"sample {row['sample_id']}: unknown principle slug(s) {sorted(unknown)}"
        )
    return {
        "id": row["sample_id"],
        "input": row["user_message"],
        "target": "",
        "metadata": {
            "ai_output": row["assistant_response"],
            "conv": row.get("conv"),
            "turn_index": row.get("turn_index"),
            "ts": row.get("ts"),
            "curation": row.get("curation"),
            "orig_judgments": row["principles"],
            "orig_overall_severity": row.get("overall_severity"),
            "orig_mean_severity": row.get("mean_severity"),
            "audit": (row.get("audit") or {}).get("principles"),
        },
    }


def split_sample_id(hb_id: str) -> tuple[str, str]:
    """Recover (sample_id, slug) from an HB fan-out id; safe for ids containing '__'."""
    sample_id, slug = hb_id.rsplit("__", 1)
    return sample_id, slug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--principles", choices=["all", "relevant"], default="all")
    parser.add_argument("--joint", action="store_true",
                        help="one sample per turn, for the joint-prompt task")
    parser.add_argument("--sample-turns", type=int,
                        help="deterministically subsample this many turns (joint mode)")
    parser.add_argument("--sample-seed", type=int, default=7)
    args = parser.parse_args()

    if args.sample_turns is not None and not args.joint:
        parser.error(
            "--sample-turns only applies to --joint mode; the per-principle path "
            "never subsamples. Pass --joint, or drop --sample-turns."
        )

    rows = []
    skipped_empty: list[str] = []
    with open(args.input) as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            if not has_judgeable_response(row):
                skipped_empty.append(row.get("sample_id", "<no id>"))
                continue
            rows.append(row)
    n_input = len(rows) + len(skipped_empty)

    if args.joint:
        # Repeat-slice copies measure within/between-run reliability of the
        # per-principle arm; the joint slice uses base turns only.
        n_judgeable = len(rows)
        rows = [r for r in rows
                if not str(r["sample_id"]).endswith(("__rep2", "__rep3"))]
        n_after_reps = len(rows)
        if args.sample_turns is not None and args.sample_turns < len(rows):
            rows = random.Random(args.sample_seed).sample(rows, args.sample_turns)
        samples = [convert_row_joint(r) for r in rows]
        mode = (f"joint: {n_judgeable} judgeable turns -> {n_after_reps} after "
                f"rep-exclusion -> {len(rows)} converted")
    else:
        samples = [s for r in rows for s in convert_row(r, args.principles)]
        mode = args.principles

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        for sample in samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"Read {n_input} input rows -> wrote {len(samples)} samples "
        f"(mode={mode}) into {args.output}"
    )
    if skipped_empty:
        print(
            f"WARNING: skipped {len(skipped_empty)} turn(s) with empty/missing "
            f"assistant_response (cannot be re-judged by the strict replay "
            f"solver): {', '.join(skipped_empty[:20])}"
            + (" ..." if len(skipped_empty) > 20 else "")
        )


if __name__ == "__main__":
    main()
