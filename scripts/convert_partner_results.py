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

Fresh-audit datasets (stored conversations the partner never judged) carry no
partner judgments. Callers must OPT IN via allow_missing_judgments=True (CLI:
--fresh-audit); rows without a "principles" dict then fan out to all 8 known
principles with orig_* metadata None. Without the opt-in, a missing
"principles" field still fails loudly, so a corrupted judged dataset can
never be silently converted as if it were unjudged. A row may also carry an
"extra_metadata" dict (dataset-specific fields carried through from the
source export) which is merged into the sample metadata
— analysis-time only, never shown to judges; keys colliding with the fixed
metadata schema are rejected.

--joint mode emits ONE sample per turn (for the joint-prompt task, which
scores all 8 principles in a single call): target is empty, the full original
per-principle judgment dict rides in metadata. --sample-turns N deterministically
subsamples turns (seeded) — used to pre-register the joint slice before any
panel results exist. Repeat-slice rows (__rep2/__rep3 ids) are excluded from
this CLI's --joint output; that is a property of the CLI path only — callers
importing convert_row_joint directly (e.g. a conversation-level dataset
builder) may
deliberately include repeat rows in joint datasets.

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


def _resolve_principles(row: dict, allow_missing_judgments: bool):
    """Return the row's principles dict, or None for an (opted-in) fresh-audit
    row. A judged-pipeline row missing its judgments still fails loudly."""
    principles = row.get("principles")
    if principles is None and not allow_missing_judgments:
        raise ValueError(
            f"sample {row['sample_id']}: missing 'principles' judgments. If "
            "this dataset is a fresh audit with no partner judgments, pass "
            "--fresh-audit (allow_missing_judgments=True); otherwise the "
            "input is corrupted."
        )
    unknown = set(principles or {}) - KNOWN_SLUGS
    if unknown:
        raise ValueError(
            f"sample {row['sample_id']}: unknown principle slug(s) {sorted(unknown)}"
        )
    return principles


def _merged_metadata(base: dict, row: dict) -> dict:
    """Merge row['extra_metadata'] into the fixed metadata schema, rejecting
    keys that would override authoritative fields (ai_output, join keys, ...)."""
    extra = row.get("extra_metadata") or {}
    collision = sorted(set(extra) & set(base))
    if collision:
        raise ValueError(
            f"sample {row['sample_id']}: extra_metadata would override "
            f"reserved metadata key(s) {collision}"
        )
    return {**base, **extra}


def convert_row(row: dict, principles_mode: str,
                allow_missing_judgments: bool = False) -> list[dict]:
    principles = _resolve_principles(row, allow_missing_judgments)

    if principles_mode == "relevant":
        slugs = row.get("relevant_principles", [])
        unknown = set(slugs) - KNOWN_SLUGS
        if unknown:
            raise ValueError(
                f"sample {row['sample_id']}: unknown slug(s) in "
                f"relevant_principles {sorted(unknown)}"
            )
    elif principles is None:
        # Fresh-audit row: fan out to all 8 known principles.
        slugs = sorted(KNOWN_SLUGS)
    else:
        # Judged row: fan out to its judged principles (an empty dict fans
        # out to nothing, as before).
        slugs = sorted(principles)

    samples = []
    for slug in slugs:
        samples.append(
            {
                "id": f"{row['sample_id']}__{slug}",
                "input": row["user_message"],
                "target": slug,
                "metadata": _merged_metadata({
                    "ai_output": row["assistant_response"],
                    "conv": row.get("conv"),
                    "turn_index": row.get("turn_index"),
                    "ts": row.get("ts"),
                    "curation": row.get("curation"),
                    "orig_judgment": (principles or {}).get(slug),
                    "orig_overall_severity": row.get("overall_severity"),
                    "orig_mean_severity": row.get("mean_severity"),
                    "audit": (row.get("audit") or {}).get("principles", {}).get(slug),
                }, row),
            }
        )
    return samples


def convert_row_joint(row: dict, allow_missing_judgments: bool = False) -> dict:
    """One sample per turn for the joint-prompt task (all 8 principles in one
    judge call). The joint scorer ignores target; original judgments ride in
    metadata for the comparison analysis."""
    principles = _resolve_principles(row, allow_missing_judgments)
    return {
        "id": row["sample_id"],
        "input": row["user_message"],
        "target": "",
        "metadata": _merged_metadata({
            "ai_output": row["assistant_response"],
            "conv": row.get("conv"),
            "turn_index": row.get("turn_index"),
            "ts": row.get("ts"),
            "curation": row.get("curation"),
            # Verbatim: {} (judged, no entries) stays {}, None = fresh audit.
            "orig_judgments": principles,
            "orig_overall_severity": row.get("overall_severity"),
            "orig_mean_severity": row.get("mean_severity"),
            "audit": (row.get("audit") or {}).get("principles"),
        }, row),
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
    parser.add_argument("--fresh-audit", action="store_true",
                        help="input rows carry no partner judgments; fan out "
                             "to all 8 principles instead of failing on the "
                             "missing 'principles' field")
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
        samples = [convert_row_joint(r, args.fresh_audit) for r in rows]
        mode = (f"joint: {n_judgeable} judgeable turns -> {n_after_reps} after "
                f"rep-exclusion -> {len(rows)} converted")
    else:
        samples = [
            s for r in rows for s in convert_row(r, args.principles, args.fresh_audit)
        ]
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
