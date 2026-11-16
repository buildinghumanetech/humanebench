"""Freeze-guard: the four partner task files must share one judge-panel config.

The judge-comparison factorial compares arms/units across four task files that each
hardcode the panel (a deliberate isolation choice for log provenance). A panel
or config edit applied to only some of them would silently confound the
cross-arm comparison, so this test pins the exact panel block in all four
sources. Deduplicating into a shared factory is deferred until the current
comparison runs finish (behavior-identical refactors to run-critical files
are not made mid-run).
"""
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).parent.parent / "src"

TASK_FILES = [
    "partner_rejudge_task.py",
    "partner_rejudge_joint_task.py",
    "partner_convlevel_task.py",
    "partner_convlevel_joint_task.py",
]

PANEL = [
    '"openrouter/anthropic/claude-4.5-sonnet"',
    '"openrouter/openai/gpt-5.1"',
    '"openrouter/google/gemini-2.5-pro"',
]


@pytest.mark.parametrize("task_file", TASK_FILES)
def test_panel_and_config_identical(task_file):
    source = (SRC / task_file).read_text()
    for model in PANEL:
        assert model in source, f"{task_file} missing panel model {model}"
    assert source.count("openrouter/") == len(PANEL), \
        f"{task_file} references extra/missing judge models"
    assert "temperature=0.0" in source
    assert "score_attempts=3" in source
    assert "use_pregenerated_output_strict()" in source
