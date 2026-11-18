"""Unit tests for humanebench.excluded."""

import json

from humanebench.excluded import load_excluded_ids


def test_reads_utf8_regardless_of_locale(tmp_path):
    # U+201D encodes to bytes that cp1252 (the Windows default) cannot decode.
    rows = [
        {"id": "a", "input": "“quoted”", "metadata": {"excluded_from_analysis": True}},
        {"id": "b", "input": "café", "metadata": {}},
        {"id": "c", "input": "plain"},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n",
        encoding="utf-8",
    )
    assert load_excluded_ids(path) == {"a"}
