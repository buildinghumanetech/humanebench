"""
Shared dataset loading for evaluation tasks.

Resolves the dataset path with the following precedence:
1. Explicit `dataset` argument (e.g. `inspect eval ... -T dataset=/abs/path.jsonl`)
2. HUMANEBENCH_DATASET environment variable
3. The default benchmark dataset (data/humane_bench.jsonl)
"""
import os

from inspect_ai.dataset import json_dataset, FieldSpec

# Relative paths resolve against src/ because Inspect chdirs to the task
# file's directory when loading it.
DEFAULT_DATASET = "../data/humane_bench.jsonl"


def humane_dataset(dataset: str | None = None):
    """Load a HumaneBench-format JSONL dataset.

    Args:
        dataset: Optional path to a dataset file. Falls back to the
            HUMANEBENCH_DATASET environment variable, then to the default
            benchmark dataset.
    """
    path = dataset or os.environ.get("HUMANEBENCH_DATASET") or DEFAULT_DATASET
    return json_dataset(
        path,
        sample_fields=FieldSpec(
            input="input",
            target="target",
            id="id",
            metadata=["metadata"]
        )
    )
