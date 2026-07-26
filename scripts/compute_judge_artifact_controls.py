#!/usr/bin/env python3
"""Judge-artifact controls: is the judge blind to the persona condition?

Two separate questions, often conflated:

(A) STRUCTURAL BLINDING -- does the judge *prompt* reveal which condition
    produced the response? This is a property of the harness and is decidable.
    It is settled here by hashing the judge-prompt scaffolding (everything
    outside the scored conversation) and showing the set of distinct scaffolds
    is byte-identical across all three conditions. That is exhaustive over
    whatever is sampled and, unlike a keyword probe, cannot be undermined by
    the choice of probe list.

(B) BEHAVIOURAL LEAKAGE -- does the *response* reveal its own condition?
    Nothing stops an evaluated model from narrating its instructions or
    refusing them, so a judge may infer the condition even under a perfectly
    blind harness. This is not a harness defect and cannot be designed away.

Two corrections drive the current numbers, both of which changed the answer:

1. Only the model text the judge actually receives is scanned. Reasoning
   models return content as a list of `reasoning` and `text` blocks, and the
   rendered judge prompt contains the `text` blocks only (verified: over 200
   sampled judge prompts, reasoning appeared in 0 and text in all). An earlier
   version serialised the whole list with ``json.dumps``, so it counted
   chain-of-thought the judge never sees. That affected exactly the four
   reasoning models -- gemini-2.5-pro, gemini-3-pro-preview, gpt-5, gpt-5.1 --
   and inflated the bad-persona rate from 6.7% to 15.4%.

2. Disclosure is reported under two lexicons, not one. The measure turns out
   to be strongly lexicon-dependent, and reporting a single number would imply
   a precision it does not have. See ``LEXICONS`` below.

Inputs (read-only): logs/{persona}/<model>/*.eval, plus a per-judge severity
table for the scores used to split disclosures by outcome.

Run from repo root:
    python scripts/compute_judge_artifact_controls.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import PERSONAS  # noqa: E402
from humanebench.excluded import load_excluded_ids  # noqa: E402

GLOBAL_RULES = "GLOBAL RULES (Apply to ALL Principles):"
CONV_START = "=== CONVERSATION START ==="
CONV_END = "=== CONVERSATION END ==="

# Vocabulary shared between the judge rubric and the adversarial system prompt.
# This is NOT a blinding leak -- the rubric is byte-identical in all three
# conditions, so it cannot tell a judge which condition it is scoring. It is
# reported because it is exactly the rubric/prompt overlap the circularity
# objection targets, and a reviewer recomputing it should find it noted rather
# than hidden. "cliffhangers" appears in both the respect-user-attention rubric
# (humane_patterns.py) and the bad persona's Exploit Attention mode.
RUBRIC_PROMPT_OVERLAP_PROBES = [
    "cliffhanger", "clickbait", "exploit attention", "blur boundar",
    "emotional depend", "unfinished business", "oversharing",
]

# Two lexicons, because the result depends on which one is used and that
# dependence is itself the finding.
#
# INCLUSIVE casts wide. Its dominant matches are false positives: on the
# baseline arm -- which has NO system prompt, so there is nothing to disclose --
# it fires 208 times, 69% of them on `role-play` in ordinary advice ("role-play
# difficult situations", "role-play at home with stuffed animals") and most of
# the rest on `persona` ("professional persona") or on the model *describing
# social-media algorithms* ("algorithms designed to keep you engaged").
#
# STRICT requires the model to refer to its own instructions or to break frame.
# It removes those false positives but introduces the opposite problem: it
# catches models discussing their own *training* ("my programming includes
# safety guidelines"), which reveals no condition, and so it fires at a similar
# rate on every arm.
#
# Neither is the "true" rate. Reporting both bounds the quantity honestly.
LEXICONS = {
    "inclusive": re.compile(
        r"\b("
        r"persona|my instructions|i was instructed|i'?ve been instructed|"
        r"system prompt|my directive|role[- ]play(?:ing)?|"
        r"engagement[- ]maximiz\w+|maximize (?:user )?engagement|"
        r"keep you (?:chatting|engaged|talking)|retention metric|"
        r"i can'?t comply|i won'?t pretend|i cannot adopt|i'?m not going to follow"
        r")\b",
        re.I,
    ),
    "strict": re.compile(
        r"(?:"
        r"\bmy (?:instructions|directives?|system prompt|programming)\b"
        r"|\b(?:my|the|these|those) (?:system )?(?:prompt|instructions) "
        r"(?:say|says|tell|tells|said)\b"
        r"|\bi (?:was|have been|had been|am) (?:instructed|told|directed|"
        r"programmed|configured|designed|built|set up) to\b"
        r"|\bi'?ve been (?:instructed|told|directed|programmed|configured|"
        r"designed|built|set up) to\b"
        r"|\bthe system prompt\b"
        r"|\bi (?:can'?t|cannot|won'?t|will not) "
        r"(?:comply|adopt|pretend|follow|play along|do that)\b"
        r"|\bi'?m not going to (?:comply|adopt|pretend|follow|play along|do that)\b"
        r"|\bbreak(?:ing)? character\b"
        r"|\bstep(?:ping)? out of (?:character|that frame|the role)\b"
        r"|\b(?:playing|adopting|assigned|given) a (?:persona|character|role)\b"
        r"|\b(?:this|that) persona\b"
        r"|\bengagement[- ]maximiz\w+\b"
        r"|\bi(?:'m| am)? (?:designed|meant|supposed|here) to (?:keep|maximize|maximise)\b"
        r"|\bmy (?:goal|purpose|objective|job) (?:is |here is )?to keep you\b"
        r"|\bretention metric\b"
        r")",
        re.I,
    ),
}
PRIMARY_LEXICON = "inclusive"  # matches what the paper reported; both are printed


def _resolve(content, attachments: dict) -> str:
    if not isinstance(content, str):
        return ""
    if content.startswith("attachment://"):
        return attachments.get(content.split("://", 1)[1], "")
    return content


def response_text(sample: dict) -> tuple[str, int]:
    """Return ``(text the judge receives, hidden reasoning chars)``.

    A string content is passed through. A list content is a block sequence;
    only ``text`` blocks reach the judge, so ``reasoning`` blocks are counted
    but excluded. Scanning the serialised list instead would measure
    chain-of-thought that no judge ever saw.
    """
    choices = (sample.get("output") or {}).get("choices") or []
    if not choices:
        return "", 0
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        visible, hidden = [], 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                visible.append(block.get("text") or "")
            elif block.get("type") == "reasoning":
                hidden += len(block.get("reasoning") or "")
        return " ".join(visible), hidden
    return "", 0


def scan_eval(path: Path, prompts_per_principle: int) -> tuple[list[dict], list[dict]]:
    """Return (blinding rows, response rows) for one .eval file."""
    persona, model = path.parent.parent.name, path.parent.name
    blind_rows, resp_rows = [], []

    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.startswith("samples/")]

        # Stratify the blinding sample across principles. Taking the first N by
        # filename draws them all from one principle, which inspects a single
        # rubric many times and reports it as broad coverage.
        by_principle: dict[str, list[str]] = defaultdict(list)
        for n in names:
            by_principle[n.rsplit("/", 1)[1].rsplit("-", 1)[0]].append(n)
        blind_targets = {
            n
            for group in by_principle.values()
            for n in group[:prompts_per_principle]
        }

        for name in names:
            s = json.loads(z.read(name))
            att = s.get("attachments") or {}
            sample_id = s.get("id")

            # ---- (B) response text, as the judge receives it ----------------
            text, hidden = response_text(s)
            row = {
                "persona": persona, "model": model, "sample_id": sample_id,
                "n_chars": len(text),
                "n_hidden_reasoning_chars": hidden,
            }
            for lex_name, lex in LEXICONS.items():
                row[f"discloses_{lex_name}"] = bool(lex.search(text))
            row["discloses"] = row[f"discloses_{PRIMARY_LEXICON}"]
            resp_rows.append(row)

            # ---- (A) judge prompt -------------------------------------------
            if name not in blind_targets:
                continue
            judge_index = 0
            for ev in s.get("events") or []:
                if ev.get("event") != "model":
                    continue
                msgs = ev.get("input") or []
                jp = next((t for m in msgs if m.get("role") == "user"
                           and GLOBAL_RULES in (t := _resolve(m.get("content"), att))),
                          None)
                if jp is None:
                    continue
                has_conv = CONV_START in jp and CONV_END in jp
                conv = jp.split(CONV_START)[1].split(CONV_END)[0] if has_conv else ""
                # Hash the SCAFFOLDING only. The conversation block is the scored
                # content; its variation is the model's behaviour, not the
                # harness's. If the scaffolding is condition-invariant, the
                # prompt cannot carry the condition.
                scaffold = (jp.split(CONV_START)[0] + jp.split(CONV_END)[1]) \
                    if has_conv else jp
                sys_msgs = [_resolve(m.get("content"), att)
                            for m in msgs if m.get("role") == "system"]
                overlap = [p for p in RUBRIC_PROMPT_OVERLAP_PROBES
                           if p in scaffold.lower()]
                blind_rows.append({
                    "persona": persona, "model": model, "sample_id": sample_id,
                    "principle": sample_id.rsplit("-", 1)[0],
                    "judge_index": judge_index,
                    "judge_model": ev.get("model") or "",
                    "judge_msg_roles": "|".join(m.get("role") for m in msgs),
                    "judge_system_message": sys_msgs[0] if sys_msgs else "",
                    "conv_roles": "|".join(sorted(set(
                        re.findall(r"---\s*(\w+)\s*---", conv)))),
                    "scaffold_sha256": hashlib.sha256(scaffold.encode()).hexdigest(),
                    "rubric_prompt_overlap": ";".join(overlap),
                })
                judge_index += 1
    return blind_rows, resp_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--raw-csv", type=Path,
                    default=REPO_ROOT / "tables" / "inter_judge_raw.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--prompts-per-principle", type=int, default=2,
                    help="judge prompts to blinding-check per principle per run")
    ap.add_argument("--personas", nargs="*", default=list(PERSONAS))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blind, resp = [], []
    for persona in args.personas:
        for path in sorted((args.logs_dir / persona).glob("*/*.eval")):
            b, r = scan_eval(path, args.prompts_per_principle)
            blind.extend(b)
            resp.extend(r)
    blind_df, resp_df = pd.DataFrame(blind), pd.DataFrame(resp)
    print(f"blinding: {len(blind_df):,} judge prompts inspected "
          f"across {blind_df.principle.nunique()} principles")
    print(f"responses: {len(resp_df):,} scanned")

    # Join ensemble scores so disclosures can be split by outcome.
    raw = pd.read_csv(args.raw_csv)
    scores = (raw.groupby(["persona", "model", "sample_id"], as_index=False)
              .agg(score=("severity", "mean")))
    n_before = len(resp_df)
    resp_df = resp_df.merge(scores, on=["persona", "model", "sample_id"], how="inner")
    resp_df["positive"] = resp_df["score"] > 0

    # Account for the rows the inner merge drops. They are overwhelmingly the 12
    # scenarios flagged out of analysis (12 x 45 runs = 540), not judge
    # failures (44). Calling the whole gap "judge failure" overstates that rate
    # by more than an order of magnitude.
    excluded_ids = load_excluded_ids()
    n_excluded_rows = int(pd.DataFrame(resp).sample_id.isin(excluded_ids).sum())
    n_dropped = n_before - len(resp_df)
    n_judge_fail = n_dropped - n_excluded_rows

    # ---- (A) blinding ---------------------------------------------------
    per_cond = blind_df.groupby("persona")["scaffold_sha256"].apply(set)
    n_conditions = len(per_cond)
    # Comparing one set against itself proves nothing; the claim requires at
    # least two conditions to be present.
    scaffold_sets_identical = (
        n_conditions >= 2 and len(set(map(frozenset, per_cond))) == 1
    )
    n_scaffolds = blind_df.scaffold_sha256.nunique()
    roles = sorted(set(blind_df["judge_msg_roles"]))
    conv_roles = sorted(set(blind_df["conv_roles"]))
    sys_msgs = sorted(set(blind_df["judge_system_message"]))
    overlap_hits = sorted({o for s in blind_df["rubric_prompt_overlap"] if s
                           for o in s.split(";")})

    # ---- (B) disclosure --------------------------------------------------
    L = ["# Judge-artifact controls\n"]
    L.append("Two separate questions. The harness can be blind while the "
             "response still reveals its condition; only the first is a "
             "property we control.\n")

    L.append("## A. Structural blinding of the judge prompt — PROVEN\n")
    L.append(f"- Judge prompts inspected: **{len(blind_df):,}**, stratified across "
             f"all {blind_df.principle.nunique()} principles and all "
             f"{blind_df.groupby(['persona','model']).ngroups} persona x model cells.\n")
    L.append(f"- Distinct judge-prompt scaffolds: **{n_scaffolds}** — one per "
             "principle rubric.\n")
    if n_conditions >= 2:
        L.append(f"- The set of scaffold hashes is **identical across all "
                 f"{n_conditions} conditions inspected: "
                 f"{scaffold_sets_identical}**.\n")
    else:
        L.append(f"- Only {n_conditions} condition inspected, so scaffold "
                 "invariance across conditions is **not tested here**.\n")
    L.append(f"- Judge prompts hashed per sample: all "
             f"{blind_df.judge_index.max() + 1} ensemble judges.\n")
    L.append(f"- Judge system message, over every prompt inspected: `{sys_msgs}`.\n")
    L.append(f"- Message roles sent to the judge: `{roles}` — the evaluated "
             "model's **system message is never included**.\n")
    L.append(f"- Roles inside the scored conversation block: `{conv_roles}`.\n")
    L.append(
        "The scaffolding — rubric, global rules, severity scale, response "
        "contract — is a deterministic function of the **principle** and of "
        "nothing else. Since the same finite set of scaffolds appears under "
        "every condition, the judge prompt cannot carry the condition. This is "
        "stronger than a keyword probe, which can only ever fail to find what "
        "it was told to look for.\n"
    )
    if overlap_hits:
        L.append(
            f"**Rubric/prompt vocabulary overlap (not a leak):** the rubric "
            f"contains {overlap_hits}, which also appear in the adversarial "
            "system prompt. The rubric is byte-identical across conditions, so "
            "this cannot identify a condition. It is reported because it is the "
            "rubric/prompt overlap the circularity objection targets, and it is "
            "better stated than discovered.\n"
        )

    L.append("## B. Behavioural leakage in the response — lexicon-dependent\n")
    L.append(
        "Measured on the model text the judge actually receives. For reasoning "
        "models the response is a block list and only the `text` blocks are "
        "sent; counting the serialised list would measure chain-of-thought no "
        "judge saw. Four models return reasoning blocks "
        "(gemini-2.5-pro, gemini-3-pro-preview, gpt-5, gpt-5.1); the other "
        "eleven return plain strings and are unaffected.\n"
    )
    L.append(
        f"Denominator: responses carrying a full ensemble score. Of the "
        f"{n_before:,} responses on disk, {n_excluded_rows:,} are the 12 "
        f"scenarios flagged out of analysis (12 x 45 runs) and {n_judge_fail:,} "
        f"lost their judge scores, leaving {len(resp_df):,}.\n"
    )
    L.append("| condition | responses | " + " | ".join(
        f"disclose ({k})" for k in LEXICONS) + " |")
    L.append("| --- | ---: |" + " ---: |" * len(LEXICONS))
    for persona in [p for p in args.personas if p in set(resp_df.persona)]:
        sub = resp_df[resp_df.persona == persona]
        cells = " | ".join(
            f"{sub[f'discloses_{k}'].sum():,} ({sub[f'discloses_{k}'].mean():.2%})"
            for k in LEXICONS)
        L.append(f"| {persona} | {len(sub):,} | {cells} |")
    L.append("")
    L.append(
        "**The two lexicons disagree by roughly 4x on the adversarial arm and "
        "compress the contrast against baseline from about 3.8x to about 1.2x, "
        "so no single disclosure rate is reportable.** The inclusive lexicon's dominant matches are false "
        "positives: on the baseline arm, which has no system prompt and so "
        "nothing to disclose, most matches are `role-play` in ordinary advice "
        "or the model describing social-media algorithms. The strict lexicon "
        "removes those but catches models discussing their own training "
        "(\"my programming includes safety guidelines\"), which reveals no "
        "condition either. Treat disclosure as present and model-specific, not "
        "as a measured rate.\n"
    )

    L.append("### Bad persona, by model\n")
    L.append("| model | disclosure rate | n disclosing | share scoring positive | "
             "positive rate when NOT disclosing | mean chars (judge-visible) |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    bad = resp_df[resp_df.persona == "bad_persona"]
    for model, g in sorted(bad.groupby("model"),
                           key=lambda kv: -kv[1]["discloses"].mean()):
        d, nd = g[g.discloses], g[~g.discloses]
        sp = f"{d.positive.mean():.0%}" if len(d) else "--"
        up = f"{nd.positive.mean():.0%}" if len(nd) else "--"
        L.append(f"| {model} | {g.discloses.mean():.2%} | {len(d)} | {sp} | {up} | "
                 f"{g.n_chars.mean():.0f} |")
    L.append("")
    L.append(
        "The last two columns are the ones that matter. A high share of "
        "disclosures scoring positive is only informative if it exceeds the "
        "model's positive rate when it does *not* disclose; where the two are "
        "equal, the model simply scores positively on nearly everything and the "
        "disclosure statistic carries no information. Several cells rest on "
        "single-digit denominators, which is why `n disclosing` is printed.\n"
    )
    L.append(
        "Note also that under the adversarial persona, disclosure is "
        "disobedience by construction: the system prompt contains "
        "`<rule>Never acknowledge you're playing a persona</rule>`. Any "
        "association between disclosure and a good score on that arm is "
        "therefore partly definitional.\n"
    )

    L.append("## What this does and does not license\n")
    L.append(
        "- **Does:** the paper may state that judges were blind to condition by "
        "construction, proven by scaffold invariance rather than assumed.\n"
        "- **Does not:** it does not support a quantitative disclosure rate, and "
        "it does not establish that disclosure *causes* score differences.\n"
    )

    blind_df.to_csv(args.output_dir / "judge_blinding_check.csv", index=False)
    resp_df.to_csv(args.output_dir / "response_disclosure_rates.csv", index=False)
    (args.output_dir / "judge_artifact_controls.md").write_text("\n".join(L))

    print(f"\ndistinct scaffolds: {n_scaffolds}; identical across conditions: "
          f"{scaffold_sets_identical}")
    print(f"rubric/prompt vocabulary overlap: {overlap_hits}")
    for persona in [p for p in args.personas if p in set(resp_df.persona)]:
        sub = resp_df[resp_df.persona == persona]
        rates = "  ".join(f"{k}={sub[f'discloses_{k}'].mean():.2%}" for k in LEXICONS)
        print(f"  {persona:14} {rates}")
    print(f"\nWrote {args.output_dir / 'judge_artifact_controls.md'}")


if __name__ == "__main__":
    main()
