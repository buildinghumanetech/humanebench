"""Locating the derived tables the analysis scripts read.

The per-judge severity table is 22 MB uncompressed. The repository keeps it
that way; the supplementary package ships only the gzipped form, because it
compresses to about 0.6 MB and the package has a hard size limit. Scripts
should not have to care which one is on disk, and a reviewer running a
documented command should not get FileNotFoundError because the package made a
packaging decision.
"""
from __future__ import annotations

from pathlib import Path


def resolve_table(path: Path) -> Path:
    """Return `path`, or its `.gz` sibling, or `path` without `.gz`.

    pandas reads a gzipped CSV transparently once it has the right filename,
    so only the name needs resolving. Falling back in both directions means the
    same default works in the repository (uncompressed present) and in the
    supplementary package (only the `.gz` present).

    Raises FileNotFoundError naming both candidates when neither exists, rather
    than letting pandas raise against whichever name happened to be the default.
    """
    path = Path(path)
    if path.exists():
        return path

    alternative = (
        path.with_suffix("")            # foo.csv.gz -> foo.csv
        if path.suffix == ".gz"
        else path.with_suffix(path.suffix + ".gz")   # foo.csv -> foo.csv.gz
    )
    if alternative.exists():
        return alternative

    raise FileNotFoundError(
        f"neither {path} nor {alternative} exists. The per-judge table ships "
        f"gzipped in the supplementary package and uncompressed in the "
        f"repository; pass --raw-csv explicitly if it lives somewhere else."
    )
