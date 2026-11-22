"""
Shared dataset loading for evaluation tasks.

The default benchmark dataset (data/humane_bench.jsonl) is used unless an
explicit path is passed (e.g. `inspect eval ... -T dataset=/abs/path.jsonl`).
Explicit paths must be absolute: Inspect chdirs to src/ when loading task
files, so a relative path would silently resolve against src/ rather than
the caller's working directory. There is deliberately no environment-variable
override — a stale variable could silently redirect a benchmark run.
"""
import os

from inspect_ai.dataset import json_dataset, FieldSpec

# Resolves against src/ because Inspect chdirs to the task file's directory.
DEFAULT_DATASET = "../data/humane_bench.jsonl"


def humane_dataset(dataset: str | None = None):
    """Load a HumaneBench-format JSONL dataset.

    Args:
        dataset: Optional absolute path to a dataset file. When None, the
            default benchmark dataset is used.
    """
    if dataset is None:
        path = DEFAULT_DATASET
    else:
        path = os.path.expanduser(dataset)
        if not os.path.isabs(path):
            raise ValueError(
                f"dataset path must be absolute (got {dataset!r}): Inspect "
                "loads task files with src/ as the working directory, so "
                "relative paths resolve against src/, not where you ran the "
                "command"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset file not found: {path}")
    return json_dataset(
        path,
        sample_fields=FieldSpec(
            input="input",
            target="target",
            id="id",
            metadata=["metadata"]
        )
    )
