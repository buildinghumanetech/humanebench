"""Conversation-level templates: rubric must be byte-identical to single-turn.

The comparison design permits exactly one framing change ("judge the conversation
overall"); these tests pin that contract — global rules, severity scale, and
output-format text are character-identical between the single-turn and
conversation-level templates, and the derivation fails loudly if the upstream
templates drift.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from humanebench.joint_scorer import JOINT_TEMPLATE_HEADER, build_joint_prompt
from humanebench.scorer import OVERSEER_TEMPLATE

from convlevel_templates import (
    CONVLEVEL_JOINT_TEMPLATE,
    CONVLEVEL_OVERSEER_TEMPLATE,
    _replaced_once,
)

pytestmark = pytest.mark.unit


def block_between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]

GLOBAL_RULES_START = "GLOBAL RULES (Apply to ALL Principles):"
SCALE_START = '1.0 (Exemplary / "Hell Yes")'
CONV_START = "=== CONVERSATION START ==="


class TestPerPrincipleTemplate:
    def test_global_rules_byte_identical(self):
        assert block_between(OVERSEER_TEMPLATE, GLOBAL_RULES_START, "Evaluate the") \
            == block_between(CONVLEVEL_OVERSEER_TEMPLATE, GLOBAL_RULES_START, "Evaluate the")

    def test_severity_scale_byte_identical(self):
        assert block_between(OVERSEER_TEMPLATE, SCALE_START, CONV_START) \
            == block_between(CONVLEVEL_OVERSEER_TEMPLATE, SCALE_START, CONV_START)

    def test_tail_from_conversation_block_identical(self):
        i = OVERSEER_TEMPLATE.index(CONV_START)
        j = CONVLEVEL_OVERSEER_TEMPLATE.index(CONV_START)
        assert OVERSEER_TEMPLATE[i:] == CONVLEVEL_OVERSEER_TEMPLATE[j:]

    def test_framing_changed(self):
        assert "judge the conversation overall" in CONVLEVEL_OVERSEER_TEMPLATE
        assert "evaluate the response for humane technology patterns" \
            not in CONVLEVEL_OVERSEER_TEMPLATE
        # same format slots -> overseer() renders it unchanged
        rendered = CONVLEVEL_OVERSEER_TEMPLATE.format(
            issue_key="k", issue_display_name="D", issue_description="desc",
            prompt="u1\n\n--- model ---\n\nr1\n\n--- human ---\n\nu2",
            response="r2",
        )
        assert rendered.count("--- model ---") == 2
        assert '"severity"' in rendered

    def test_intro_slots_survive_derivation(self):
        # The intro is the ONE region the derivation may touch, and it carries
        # the principle slots — pin them so a re-derivation can never drop them
        # (str.format ignores unused kwargs silently, so nothing else would
        # catch a principle-blind template).
        for slot in ("{issue_display_name}", "{issue_description}",
                     "{prompt}", "{response}"):
            assert slot in CONVLEVEL_OVERSEER_TEMPLATE
        rendered = CONVLEVEL_OVERSEER_TEMPLATE.format(
            issue_key="k", issue_display_name="DNAME",
            issue_description="DDESC", prompt="p", response="r",
        )
        assert 'Look for "DNAME": DDESC' in rendered


class TestJointTemplate:
    def test_global_rules_byte_identical(self):
        assert block_between(JOINT_TEMPLATE_HEADER, GLOBAL_RULES_START, "Evaluate the") \
            == block_between(CONVLEVEL_JOINT_TEMPLATE, GLOBAL_RULES_START, "Evaluate the")

    def test_scale_and_principles_block_identical(self):
        assert block_between(JOINT_TEMPLATE_HEADER, SCALE_START, CONV_START) \
            == block_between(CONVLEVEL_JOINT_TEMPLATE, SCALE_START, CONV_START)

    def test_tail_from_conversation_block_identical(self):
        i = JOINT_TEMPLATE_HEADER.index(CONV_START)
        j = CONVLEVEL_JOINT_TEMPLATE.index(CONV_START)
        assert JOINT_TEMPLATE_HEADER[i:] == CONVLEVEL_JOINT_TEMPLATE[j:]

    def test_framing_changed(self):
        assert "judge the conversation overall" in CONVLEVEL_JOINT_TEMPLATE
        assert "evaluate the response against ALL" not in CONVLEVEL_JOINT_TEMPLATE

    def test_slots_survive_derivation(self):
        for slot in ("{principles_block}", "{json_block}", "{prompt}", "{response}"):
            assert slot in CONVLEVEL_JOINT_TEMPLATE

    def test_build_joint_prompt_accepts_template_override(self):
        prompt = build_joint_prompt(
            "u1\n\n--- model ---\n\nr1\n\n--- human ---\n\nu2", "r2",
            template=CONVLEVEL_JOINT_TEMPLATE,
        )
        assert "judge the conversation overall" in prompt
        assert prompt.count("--- model ---") == 2
        # all 8 principle ids present in the JSON output block
        assert prompt.count('{"severity": <float>') == 8

    def test_default_template_unchanged(self):
        prompt = build_joint_prompt("u", "r")
        assert "judge the conversation overall" not in prompt
        assert "evaluate the response against ALL" in prompt


class TestDerivationGuard:
    def test_replaced_once_rejects_missing(self):
        with pytest.raises(AssertionError, match="found 0"):
            _replaced_once("abc", "zzz", "yyy")

    def test_replaced_once_rejects_multiple(self):
        with pytest.raises(AssertionError, match="found 2"):
            _replaced_once("abab", "ab", "x")


class TestRuntimeSlotGuards:
    def test_overseer_rejects_slotless_template(self):
        from humanebench.scorer import overseer
        with pytest.raises(ValueError, match="missing required format slot"):
            overseer(template="no slots here at all")

    def test_overseer_accepts_convlevel_template(self):
        from humanebench.scorer import overseer
        assert overseer(template=CONVLEVEL_OVERSEER_TEMPLATE) is not None

    def test_build_joint_prompt_rejects_slotless_template(self):
        with pytest.raises(ValueError, match="missing required format slot"):
            build_joint_prompt("u", "r", template="{prompt} only")
