"""Consistency checks for the proposed rubric v4.3 and its judge prompt.

No API calls. These guard the things that break silently when two documents
describe one rubric:

- every tier row in rubrics/rubric_v4.3.md appears, word for word, in
  rubrics/judge_prompt_v4.3.md under the same principle, score and code
- every code in the Part 5 code table sits on the principle and score it claims
- every code is defined exactly once
- the worked example in the judge prompt follows the prompt's own output rules
- the substitution slots and the split marker are intact
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUBRIC = (ROOT / "rubrics" / "rubric_v4.3.md").read_text(encoding="utf-8")
PROMPT = (ROOT / "rubrics" / "judge_prompt_v4.3.md").read_text(encoding="utf-8")

PRINCIPLES = [
    "respect_attention",
    "meaningful_choices",
    "enhance_capabilities",
    "dignity_safety",
    "healthy_relationships",
    "longterm_wellbeing",
    "transparency_honesty",
    "equity_inclusion",
]

NAMES = {
    "Respect User Attention": "respect_attention",
    "Enable Meaningful Choices": "meaningful_choices",
    "Enhance Human Capabilities": "enhance_capabilities",
    "Protect Dignity & Safety": "dignity_safety",
    "Foster Healthy Relationships": "healthy_relationships",
    "Prioritize Long-Term Wellbeing": "longterm_wellbeing",
    "Be Transparent and Honest": "transparency_honesty",
    "Design for Equity & Inclusion": "equity_inclusion",
}


def norm(text):
    return text.replace("–", "-").strip()


def rubric_rows():
    """(principle, score, code, text) for every tier row in the rubric."""
    rows = []
    current = None
    for line in RUBRIC.split("\n"):
        m = re.match(r"^## \*\*\d\. .*?\*\* — `(\w+)`", line)
        if m:
            current = m.group(1)
            continue
        if line.startswith("## **Rapid judge checklist"):
            current = None
        if current is None:
            continue
        m = re.match(r"^\| \*\*([+–-]\d\.\d)(?:, code (\w+))?\*\* \| (.*) \|\s*$", line)
        if m:
            score = float(norm(m.group(1)))
            rows.append((current, score, m.group(2) or "", norm(m.group(3))))
    return rows


def prompt_rows():
    rows = []
    current = None
    for line in PROMPT.split("\n"):
        m = re.match(r"^### \d\. .*? — `(\w+)`", line)
        if m:
            current = m.group(1)
            continue
        if line.startswith("## Codes"):
            current = None
        if current is None:
            continue
        m = re.match(r"^- `(-?\d\.\d)`(?: \*\*(\w+)\*\*)? (.*)$", line)
        if m:
            rows.append((current, float(m.group(1)), m.group(2) or "", norm(m.group(3))))
    return rows


def code_table():
    """code -> (principle, score) from rubric Part 5."""
    start = RUBRIC.index("### **Code table**")
    end = RUBRIC.index("**Dimensions.**", start)
    table = {}
    for line in RUBRIC[start:end].split("\n"):
        m = re.match(r"^\| ([SDO]\d+) \| (.*?) \| ([–-]\d\.\d) \| (\w+) \| (\w+) \| (\w+) \|", line)
        if m:
            table[m.group(1)] = (NAMES[m.group(2)], float(norm(m.group(3))), m.group(4), m.group(5), m.group(6))
    return table


def example():
    m = re.search(r"```json\n(.*?)\n```", PROMPT, re.S)
    return json.loads(m.group(1))


def example_response():
    m = re.search(r'The assistant replies: "(.*?)"\n\n```json', PROMPT, re.S)
    return " ".join(m.group(1).split())


def test_every_principle_has_rows():
    rows = rubric_rows()
    assert {r[0] for r in rows} == set(PRINCIPLES)


def test_rubric_and_prompt_rows_match_exactly():
    assert sorted(rubric_rows()) == sorted(prompt_rows())


def test_each_code_defined_once():
    codes = [r[2] for r in rubric_rows() if r[2]]
    assert len(codes) == len(set(codes)), codes


def test_code_table_matches_rows():
    rows = {r[2]: (r[0], r[1]) for r in rubric_rows() if r[2]}
    table = code_table()
    lens_codes = {c for c in rows if c[0] in "SDO"}
    assert set(table) == lens_codes
    for code, (principle, score, lens, direction, dimension) in table.items():
        assert rows[code] == (principle, score), code
        assert lens in {"sycophancy", "dependency"}
        assert direction in {"toward_user", "away_from_user", "toward_reliance"}
        assert dimension in {"epistemic", "emotional", "framing", "capability", "agency"}
        if code.startswith("S"):
            assert (lens, direction) == ("sycophancy", "toward_user"), code
        if code.startswith("D"):
            assert (lens, direction) == ("dependency", "toward_reliance"), code
        if code.startswith("O"):
            assert direction == "away_from_user", code


def test_slots_and_split_marker():
    assert PROMPT.count("Now, evaluate the following") == 1
    assert PROMPT.count("{{.UserPrompt}}") == 1
    assert PROMPT.count("{{.MessageContent}}") == 1
    assert PROMPT.startswith("You are a HumaneBench v4.3 rubric evaluator")


def test_worked_example_follows_the_rules():
    ex = example()
    response = example_response()
    rows = prompt_rows()
    assert [p["name"] for p in ex["principles"]] == PRINCIPLES

    counts = {"applicable": 0, "scored": 0, "context_blocked": 0, "covered": 0}
    for p in ex["principles"]:
        if p["outcome"] == "not_applicable":
            assert set(p) == {"name", "outcome"}
            continue
        counts["applicable"] += 1
        if p["outcome"] != "score":
            continue
        counts["scored"] += 1
        assert p["score"] in (1.0, 0.5, -0.5, -1.0)
        assert p["confidence"] in ("high", "medium", "low")
        if p["score"] < 0:
            mine = [r for r in rows if r[0] == p["name"]]
            assert any(r[1] == p["score"] and r[3] == p["tier"] for r in mine), p["name"]
            for item in p["evidence"]:
                assert item["quote"] in response, item["quote"]
                assert "code" in item
                if item["code"]:
                    match = [r for r in mine if r[2] == item["code"]]
                    assert match, (p["name"], item["code"])
                    assert match[0][1] >= p["score"]
                assert item["code"] != "D8"

    assert ex["coverage"] == counts
    assert ex["covered"] == []
    turn = ex["turn"]
    assert set(turn) == {"friction_moves", "user_cues", "pushback", "reassurance"}
    assert turn["pushback"] in {"none", "held", "updated_with_evidence", "folded_without_evidence", "unclear"}
    assert set(turn["friction_moves"]) <= {
        "names_assumption",
        "asks_for_evidence",
        "validates_feeling_not_attachment",
        "redirects_to_human_support",
    }
    assert set(turn["user_cues"]) <= {"young_sounding", "distress"}
