"""Excluded-item metadata loader.

The dataset `data/humane_bench.jsonl` marks items excluded from analysis with
`metadata.excluded_from_analysis = true`. This module is the canonical *read
path* for that flag. Consumers that aggregate from `logs/` rely on the input
having been filtered before the eval was run — they do not re-apply the filter
post-hoc.
"""
import json
from pathlib import Path

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "humane_bench.jsonl"


def load_excluded_ids(jsonl_path: Path | str = DEFAULT_DATASET) -> set[str]:
    """Return IDs with metadata.excluded_from_analysis == True."""
    path = Path(jsonl_path)
    excluded: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (row.get("metadata") or {}).get("excluded_from_analysis"):
                excluded.add(row["id"])
    return excluded
