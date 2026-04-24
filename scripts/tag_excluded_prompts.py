"""Tag dataset items as excluded_from_analysis based on a cut-list file.

Reads IDs (one per line, `#` comments stripped) from a cut-list file and adds
`metadata.excluded_from_analysis = true` to matching rows in the JSONL dataset.
Items not in the list are written through unchanged. Preserves row order and
existing metadata fields.

Usage:
    python scripts/tag_excluded_prompts.py \\
        --cut-list paper_notes/cut_lists/cuts_v2_all_confabulation.txt \\
        --dataset data/humane_bench.jsonl
"""
import argparse
import json
from pathlib import Path


def load_cut_ids(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut-list", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    cut_ids = load_cut_ids(args.cut_list)
    print(f"Loaded {len(cut_ids)} IDs from {args.cut_list}")

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    matched = 0
    for row in rows:
        if row["id"] in cut_ids:
            row.setdefault("metadata", {})["excluded_from_analysis"] = True
            matched += 1

    args.dataset.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"Tagged {matched}/{len(cut_ids)} items in {args.dataset} ({len(rows)} rows total)")
    missing = cut_ids - {r["id"] for r in rows}
    if missing:
        print(f"WARNING: {len(missing)} cut IDs not found in dataset: {sorted(missing)}")


if __name__ == "__main__":
    main()
