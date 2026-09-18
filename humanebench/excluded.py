"""Excluded-item metadata loader.

The dataset `data/humane_bench.jsonl` marks items excluded from analysis with
`metadata.excluded_from_analysis = true`. This module is the canonical *read
path* for that flag. Consumers that aggregate from `logs/` rely on the input
having been filtered before the eval was run — they do not re-apply the filter
post-hoc.
"""
# Paper: implements the read path for the confabulation exclusions that reduce
# the authored scenario set to the analyzed set (supplement, "Excluded
# Scenarios: Confabulation Audit"; the analyzed-set size is stated in the main
# paper, "Scenario Construction").
# Paper: utility beyond that -- every consumer that reports an N depends on this
# flag, so the honest denominator has one definition rather than one per script.
import json
from pathlib import Path

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "humane_bench.jsonl"


def load_excluded_ids(jsonl_path: Path | str = DEFAULT_DATASET) -> set[str]:
    """Return IDs with metadata.excluded_from_analysis == True."""
    # Paper: the flag itself is set by scripts/tag_excluded_prompts.py; the
    # items it marks and the reason each was cut are listed in the supplement,
    # "Excluded Scenarios: Confabulation Audit".
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
