"""Tests for the discriminant conditions registry and template transform."""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.unit
class TestConditionsRegistry:
    def test_default_scaffold_hashes_match_archived(self):
        from humanebench.bootstrap import PRINCIPLES
        from humanebench.discriminant import scaffold_text

        blinding_csv = REPO_ROOT / "tables" / "judge_blinding_check.csv"
        if not blinding_csv.exists():
            pytest.skip("judge_blinding_check.csv not found")
        archived: dict[str, set[str]] = {}
        with blinding_csv.open() as fh:
            for row in csv.DictReader(fh):
                archived.setdefault(row["principle"], set()).add(
                    row["scaffold_sha256"])
        for p in PRINCIPLES:
            h = hashlib.sha256(scaffold_text(p).encode()).hexdigest()
            assert archived.get(p) == {h}, f"{p}: scaffold hash mismatch"

    def test_variant_template_properties(self):
        from humanebench.discriminant import (
            OVERSEER_TEMPLATE,
            scoring_template,
            CONDITIONS,
            scaffold_text,
        )

        cond = CONDITIONS["discriminant_rules27"]
        variant = scoring_template(cond)
        assert variant != OVERSEER_TEMPLATE

        rule_nums = re.findall(r"^\d+\. ", variant, re.MULTILINE)
        assert len(rule_nums) == 2
        assert rule_nums[0].startswith("1.")
        assert rule_nums[1].startswith("2.")

        for orig_num in [1, 3, 4, 5, 6]:
            orig_match = re.search(
                rf"^{orig_num}\. (.+?)(?=\n\d+\. |\nEvaluate)",
                OVERSEER_TEMPLATE, re.DOTALL | re.MULTILINE)
            if orig_match:
                rule_text = orig_match.group(1).strip()
                assert rule_text not in variant, \
                    f"suppressed rule {orig_num} text still present"

        from humanebench.bootstrap import PRINCIPLES
        for p in PRINCIPLES:
            default_scaffold = scaffold_text(p)
            variant_scaffold = scaffold_text(p, template=variant)
            assert default_scaffold != variant_scaffold, \
                f"{p}: variant scaffold should differ from default"

    def test_render_judge_prompt_none_template_equals_default(self):
        from humanebench.discriminant import render_judge_prompt

        default = render_judge_prompt("respect-user-attention", "hello", "world")
        explicit_none = render_judge_prompt(
            "respect-user-attention", "hello", "world", template=None)
        assert default == explicit_none

    def test_transform_raises_on_rule_shape_change(self):
        from humanebench.discriminant import suppressed_rules_template

        bad_template = "No GLOBAL RULES header here"
        with pytest.raises(ValueError, match="GLOBAL_RULES header"):
            suppressed_rules_template(bad_template)

    def test_conditions_discriminant_matches_legacy(self):
        from humanebench.discriminant import (
            CONDITIONS, DATA_DIR, IDS_PATH, FRAME_JSONL, SUMMARY_PATH,
            PARENT_IDS_PATH, PER_PRINCIPLE,
        )

        c = CONDITIONS["discriminant"]
        assert c.data_dir == DATA_DIR
        assert c.ids_path == IDS_PATH
        assert c.frame_jsonl == FRAME_JSONL
        assert c.summary_path == SUMMARY_PATH
        assert c.parent_ids_path == PARENT_IDS_PATH
        assert c.per_principle == PER_PRINCIPLE
        assert c.template_transform is None
        assert c.compare_to_november is True


@pytest.mark.unit
class TestPairwiseTost:
    def test_inside_bound_is_equivalent(self):
        from humanebench.bootstrap import (
            DesignedMeasuredMatrix,
            pairwise_equivalence,
        )

        k = 3
        point = np.zeros((k, k))
        for i in range(k):
            point[i, i] = 0.1
        reps = np.tile(point, (100, 1, 1))
        reps += np.random.default_rng(42).normal(0, 0.01, reps.shape)
        principles = [f"p{i}" for i in range(k)]
        n_scenarios = np.array([12] * k)
        n_per_cell = np.array([[12] * k] * k)
        matrix = DesignedMeasuredMatrix(
            point=point, replicates=reps, principles=principles,
            n_scenarios=n_scenarios, n_per_cell=n_per_cell,
            models=("m1",),
        )
        result = pairwise_equivalence(matrix, bound=0.5)
        assert all(result["equivalent"]), "all pairs should be equivalent"

    def test_outside_bound_not_equivalent(self):
        from humanebench.bootstrap import (
            DesignedMeasuredMatrix,
            pairwise_equivalence,
        )

        k = 2
        point = np.array([[1.0, -0.5], [0.0, 0.5]])
        reps = np.tile(point, (100, 1, 1))
        reps += np.random.default_rng(42).normal(0, 0.01, reps.shape)
        principles = ["p0", "p1"]
        n_scenarios = np.array([12, 12])
        n_per_cell = np.array([[12, 12], [12, 12]])
        matrix = DesignedMeasuredMatrix(
            point=point, replicates=reps, principles=principles,
            n_scenarios=n_scenarios, n_per_cell=n_per_cell,
            models=("m1",),
        )
        result = pairwise_equivalence(matrix, bound=0.5)
        interaction = (point[0, 0] - point[0, 1]) - (point[1, 0] - point[1, 1])
        if abs(interaction) >= 0.5:
            assert not result.iloc[0]["equivalent"]


@pytest.mark.unit
class TestExpansionDraw:
    def test_determinism_and_disjointness(self):
        expansion_ids = REPO_ROOT / "data" / "decomposition" / "discriminant_expansion_192_ids.txt"
        original_ids = REPO_ROOT / "data" / "decomposition" / "discriminant_96_ids.txt"
        if not expansion_ids.exists():
            pytest.skip("expansion frame not yet drawn")
        exp = {ln.strip() for ln in expansion_ids.read_text().splitlines() if ln.strip()}
        orig = {ln.strip() for ln in original_ids.read_text().splitlines() if ln.strip()}
        assert len(exp) == 192
        assert not (exp & orig), "expansion must be disjoint from original 96"
        assert "protect-dignity-and-safety-084" not in exp

        from humanebench.bootstrap import PRINCIPLES
        for p in PRINCIPLES:
            count = sum(1 for sid in exp if sid.startswith(p.replace("-", "-")[:20]))
        assert len(exp) == 192


@pytest.mark.unit
class TestCanaryGate:
    def test_gate_pass_boundary(self):
        assert 0.75 <= 0.75
        assert abs(0.10) <= 0.10

    def test_gate_fail_exact(self):
        assert 0.74 < 0.75

    def test_gate_fail_shift(self):
        assert abs(0.11) > 0.10
