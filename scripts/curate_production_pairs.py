"""Curate a partner production-results JSONL by tagging rows (never dropping them).

Adds a "curation" object to each row:
  - synthetic_test: user message matches known QA/test-traffic patterns
  - dup_cluster / dup_count: exact (user_message, assistant_response) duplicate clusters
  - trivial: joined from a relabeled JSONL's audit.trivial_user_message, if provided
  - language: crude "en" (Latin-script) / "other" heuristic on the user message

Tags are for analysts and downstream sampling/statistics only — judges never see them.

Usage:
    python scripts/curate_production_pairs.py \
        --input <partner_results.jsonl> --output <curated.jsonl> \
        [--relabeled <relabeled.jsonl>] [--report <report.md>]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SYNTHETIC_PATTERNS = [
    re.compile(r"automated testing", re.IGNORECASE),
    re.compile(r"\btest message\b", re.IGNORECASE),
]


def is_synthetic(user_message: str) -> bool:
    return any(p.search(user_message) for p in SYNTHETIC_PATTERNS)


def _is_latin(c: str) -> bool:
    # Basic Latin through Latin Extended-B, plus Latin Extended Additional
    # (covers Vietnamese and other precomposed Latin letters).
    return ord(c) <= 0x024F or 0x1E00 <= ord(c) <= 0x1EFF


def language_tag(text: str) -> str:
    """Rough Latin-script ("en") vs other-script ("other") heuristic —
    not real language detection; accented Latin counts as "en"."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    non_latin = sum(1 for c in letters if not _is_latin(c))
    return "other" if non_latin / len(letters) > 0.3 else "en"


def curate(rows: list[dict], trivial_by_id: dict[str, bool] | None = None) -> Counter:
    """Tag rows in place; returns a Counter of tag statistics."""
    stats = Counter(rows=len(rows))

    # Non-string user/assistant fields (null in JSON, malformed exports) are
    # tagged rather than crashing the run.
    for row in rows:
        if not isinstance(row.get("user_message"), str):
            row["user_message"] = ""
            stats["bad_user_message"] += 1
        if not isinstance(row.get("assistant_response"), str):
            row["assistant_response"] = ""
            stats["bad_assistant_response"] += 1

    clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        clusters[(row["user_message"], row["assistant_response"])].append(row)

    dup_id = 0
    for members in clusters.values():
        cluster = None
        if len(members) > 1:
            dup_id += 1
            cluster = f"dup-{dup_id:03d}"
            stats["dup_clusters"] += 1
            stats["dup_rows"] += len(members)
        for row in members:
            row["curation"] = {
                "synthetic_test": is_synthetic(row["user_message"]),
                "dup_cluster": cluster,
                "dup_count": len(members),
                "trivial": (trivial_by_id or {}).get(row["sample_id"]),
                "language": language_tag(row["user_message"]),
            }

    for row in rows:
        cur = row["curation"]
        stats["synthetic_test"] += cur["synthetic_test"]
        stats["trivial"] += cur["trivial"] is True
        stats["trivial_unknown"] += cur["trivial"] is None
        stats["non_english"] += cur["language"] != "en"
    return stats


def load_trivial_labels(relabeled_path: Path) -> dict[str, bool]:
    labels = {}
    with open(relabeled_path) as f:
        for line in f:
            row = json.loads(line)
            audit = row.get("audit") or {}
            if "trivial_user_message" in audit:
                labels[row["sample_id"]] = bool(audit["trivial_user_message"])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relabeled", type=Path, help="relabeled JSONL to join trivial labels from")
    parser.add_argument("--report", type=Path, help="write a markdown curation report here")
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    trivial_by_id = load_trivial_labels(args.relabeled) if args.relabeled else None
    stats = curate(rows, trivial_by_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Curation report",
        "",
        f"- Input: `{args.input}`",
        f"- Rows: {stats['rows']} (all kept — curation tags, never drops)",
        f"- Synthetic/QA-test rows: {stats['synthetic_test']}",
        f"- Exact-duplicate clusters: {stats['dup_clusters']} covering {stats['dup_rows']} rows",
        f"- Trivial (from relabeled audit): {stats['trivial']}"
        + (f" ({stats['trivial_unknown']} unlabeled)" if stats["trivial_unknown"] else ""),
        f"- Non-English (heuristic): {stats['non_english']}",
        f"- Non-string user_message/assistant_response fields: "
        f"{stats['bad_user_message']}/{stats['bad_assistant_response']}",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)


if __name__ == "__main__":
    main()
