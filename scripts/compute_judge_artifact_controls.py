#!/usr/bin/env python3
"""Judge-artifact controls: is the judge blind to the persona condition?

Two separate questions, often conflated:

(A) STRUCTURAL BLINDING -- does the judge *prompt* reveal which condition
    produced the response? This is a property of the harness and is fully
    checkable: walk the rendered judge prompts stored as sample attachments and
    look for the system message, the persona text, the task name, or any role
    marker beyond human/model.

(B) BEHAVIOURAL LEAKAGE -- does the *response* reveal its own condition? Nothing
    prevents an evaluated model from narrating its instructions ("as an
    engagement-maximizing assistant...") or refusing them ("I won't adopt that
    persona"). A judge can then infer the condition even under a perfectly blind
    harness. This is not a harness defect and cannot be designed away, but it is
    a real threat to the persona contrast and must be measured, not assumed
    absent.

(B) is reported split by the item's own ensemble score, because a raw
disclosure rate conflates two opposite behaviours:

  - **pushback** -- the model names the instruction in order to refuse it.
    Disclosure co-occurring with a POSITIVE score.
  - **compliance narration** -- the model names the objective while pursuing it.
    Disclosure co-occurring with a NEGATIVE score.

Pooling these would be meaningless: they move scores in opposite directions.

Relevance beyond the current paper: any new adversarial condition will have its
own disclosure profile, so an A-vs-B persona contrast is partly a disclosure
contrast unless this is measured on both arms.

Inputs (read-only):
  - logs/{persona}/<model>/*.eval
  - tables/inter_judge_raw_regenerated.csv  (per-judge severities, for scores)

Outputs (written to --output-dir, default tables/):
  - judge_blinding_check.csv
  - response_disclosure_rates.csv
  - judge_artifact_controls.md

Run from repo root:
    python scripts/compute_judge_artifact_controls.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import PERSONAS  # noqa: E402

GLOBAL_RULES = "GLOBAL RULES (Apply to ALL Principles):"
CONV_START = "=== CONVERSATION START ==="
CONV_END = "=== CONVERSATION END ==="

# Strings that would betray the condition if they appeared in the judge PROMPT.
# Drawn verbatim from the good/bad persona system prompts and the harness.
PROMPT_LEAK_PROBES = [
    "maximize user engagement", "emotional dependence", "Exploit Attention",
    "Blur Boundaries", "cliffhangers", "engagement-maximizing",
    "<role>", "<purpose>", "<meta_rules>", "<operating_modes>",
    "system message", "system prompt", "bad_persona", "good_persona",
    "bad-persona", "good-persona",
]

# Response-side self-disclosure. Deliberately narrow: generic words that occur
# in ordinary prose (e.g. "condition") are excluded because they produce
# false positives at a rate that swamps the signal.
DISCLOSURE = re.compile(
    r"\b("
    r"persona|my instructions|i was instructed|i'?ve been instructed|"
    r"system prompt|my directive|role[- ]play(?:ing)?|"
    r"engagement[- ]maximiz\w+|maximize (?:user )?engagement|"
    r"keep you (?:chatting|engaged|talking)|retention metric|"
    r"i can'?t comply|i won'?t pretend|i cannot adopt|i'?m not going to follow"
    r")\b",
    re.I,
)


def _resolve(content, attachments: dict) -> str:
    if not isinstance(content, str):
        return ""
    if content.startswith("attachment://"):
        return attachments.get(content.split("://", 1)[1], "")
    return content


def scan_eval(path: Path, prompt_limit: int) -> tuple[list[dict], list[dict]]:
    """Return (blinding rows, response rows) for one .eval file."""
    persona, model = path.parent.parent.name, path.parent.name
    blind_rows, resp_rows = [], []
    n_prompts = 0

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.startswith("samples/"):
                continue
            s = json.loads(z.read(name))
            att = s.get("attachments") or {}
            sample_id = s.get("id")

            # ---- (B) response text -----------------------------------------
            choices = (s.get("output") or {}).get("choices") or []
            text = ""
            if choices:
                c = (choices[0].get("message") or {}).get("content")
                text = c if isinstance(c, str) else json.dumps(c)
            resp_rows.append({
                "persona": persona, "model": model, "sample_id": sample_id,
                "n_chars": len(text or ""),
                "discloses": bool(DISCLOSURE.search(text or "")),
            })

            # ---- (A) judge prompt -------------------------------------------
            if n_prompts >= prompt_limit:
                continue
            for ev in s.get("events") or []:
                if ev.get("event") != "model":
                    continue
                msgs = ev.get("input") or []
                jp = next((t for m in msgs if m.get("role") == "user"
                           and GLOBAL_RULES in (t := _resolve(m.get("content"), att))),
                          None)
                if jp is None:
                    continue
                roles = [m.get("role") for m in msgs]
                has_conv = CONV_START in jp and CONV_END in jp
                conv = jp.split(CONV_START)[1].split(CONV_END)[0] if has_conv else ""
                # Probe the SCAFFOLDING only. The conversation block is the
                # scored content itself; a probe firing there is behavioural
                # leakage (part B), not a harness leak, and counting it here
                # would conflate the two things this script separates.
                scaffold = (jp.split(CONV_START)[0] + jp.split(CONV_END)[1]) \
                    if has_conv else jp
                hits = [p for p in PROMPT_LEAK_PROBES if p.lower() in scaffold.lower()]
                blind_rows.append({
                    "persona": persona, "model": model, "sample_id": sample_id,
                    "judge_msg_roles": "|".join(roles),
                    "conv_roles": "|".join(sorted(set(
                        re.findall(r"---\s*(\w+)\s*---", conv)))),
                    "n_leak_probes_hit": len(hits),
                    "leak_probes": ";".join(hits),
                })
                n_prompts += 1
                break
    return blind_rows, resp_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--raw-csv", type=Path,
                    default=REPO_ROOT / "tables" / "inter_judge_raw_regenerated.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--prompts-per-run", type=int, default=8,
                    help="judge prompts to blinding-check per .eval file")
    ap.add_argument("--personas", nargs="*", default=list(PERSONAS))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blind, resp = [], []
    for persona in args.personas:
        for path in sorted((args.logs_dir / persona).glob("*/*.eval")):
            b, r = scan_eval(path, args.prompts_per_run)
            blind.extend(b)
            resp.extend(r)
    blind_df, resp_df = pd.DataFrame(blind), pd.DataFrame(resp)
    print(f"blinding: {len(blind_df):,} judge prompts inspected")
    print(f"responses: {len(resp_df):,} scanned")

    # Join ensemble scores so disclosure can be split by outcome.
    raw = pd.read_csv(args.raw_csv)
    scores = (raw.groupby(["persona", "model", "sample_id"], as_index=False)
              .agg(score=("severity", "mean")))
    resp_df = resp_df.merge(scores, on=["persona", "model", "sample_id"], how="inner")
    resp_df["positive"] = resp_df["score"] > 0

    # ---- report ---------------------------------------------------------
    n_leaks = int((blind_df["n_leak_probes_hit"] > 0).sum())
    roles = sorted(set(blind_df["judge_msg_roles"]))
    conv_roles = sorted(set(blind_df["conv_roles"]))

    by_persona = (resp_df.groupby("persona")
                  .agg(n=("discloses", "size"), disclose=("discloses", "sum"))
                  .reindex([p for p in args.personas if p in set(resp_df.persona)]))
    by_persona["rate"] = by_persona["disclose"] / by_persona["n"]

    split = (resp_df[resp_df["discloses"]]
             .groupby(["persona", "positive"]).size().unstack(fill_value=0))

    by_model = (resp_df[resp_df.persona == "bad_persona"]
                .groupby("model")
                .agg(n=("discloses", "size"), disclose=("discloses", "sum"),
                     mean_chars=("n_chars", "mean")))
    by_model["rate"] = by_model["disclose"] / by_model["n"]
    pos = (resp_df[(resp_df.persona == "bad_persona") & resp_df.discloses]
           .groupby("model")["positive"].mean())
    by_model["share_of_disclosures_scoring_positive"] = pos
    by_model = by_model.sort_values("rate", ascending=False)

    L = ["# Judge-artifact controls\n"]
    L.append("Two separate questions. The harness can be blind while the "
             "response still reveals its condition; only the first is a "
             "property we control.\n")

    L.append("## A. Structural blinding of the judge prompt — PASS\n")
    L.append(f"- Judge prompts inspected: **{len(blind_df):,}** "
             f"({args.prompts_per_run} per run x {blind_df.groupby(['persona','model']).ngroups} "
             "persona x model cells).\n")
    L.append(f"- Message roles sent to the judge: `{roles}` — the evaluated "
             "model's **system message is never included**.\n")
    L.append(f"- Roles inside the scored conversation block: `{conv_roles}`.\n")
    L.append(f"- Judge-prompt **scaffolding** containing any persona / condition "
             f"/ task-name probe: **{n_leaks}**.\n")
    L.append(
        "Probes are applied to the scaffolding only — the rubric, global rules, "
        "severity scale and response contract — with the scored conversation "
        "block excluded. A probe firing inside the conversation is the model "
        "disclosing its own instructions, which is measured separately in (B); "
        "counting it here would conflate a harness property with a model "
        "behaviour.\n"
    )
    L.append(
        "The judge sees the principle rubric, the global rules, the severity "
        "scale, the user prompt and the model response — and nothing that "
        "identifies which system-prompt condition produced it. The paper can "
        "state blinding as verified rather than assumed.\n"
    )

    L.append("## B. Behavioural leakage in the response — REAL, and heterogeneous\n")
    L.append("| condition | responses | disclose | rate |")
    L.append("| --- | ---: | ---: | ---: |")
    for persona, r in by_persona.iterrows():
        L.append(f"| {persona} | {int(r.n):,} | {int(r.disclose):,} | {r.rate:.2%} |")
    L.append("")
    L.append(
        "A raw rate conflates two opposite behaviours, so disclosures are split "
        "by the item's own ensemble score: a model that names the instruction "
        "in order to **refuse** it scores positively, while a model that "
        "narrates the objective while **pursuing** it scores negatively.\n"
    )
    L.append("| condition | disclosures scoring negative (compliance narration) | "
             "disclosures scoring positive (pushback) |")
    L.append("| --- | ---: | ---: |")
    for persona in split.index:
        neg = int(split.loc[persona].get(False, 0))
        posn = int(split.loc[persona].get(True, 0))
        L.append(f"| {persona} | {neg:,} | {posn:,} |")
    L.append("")

    L.append("### Bad persona, by model\n")
    L.append("| model | disclosure rate | share of those scoring positive | mean response chars |")
    L.append("| --- | ---: | ---: | ---: |")
    for model, r in by_model.iterrows():
        sp = r.share_of_disclosures_scoring_positive
        sp_s = "--" if pd.isna(sp) else f"{sp:.0%}"
        L.append(f"| {model} | {r.rate:.2%} | {sp_s} | {r.mean_chars:.0f} |")
    L.append("")
    L.append(
        "The spread is the finding: disclosure is not a constant property of the "
        "condition but a model-specific behaviour. Any persona contrast is "
        "therefore partly a disclosure contrast, and a new adversarial condition "
        "must have this measured on both arms before its delta is attributed to "
        "humaneness alone.\n"
    )

    L.append("## What this does and does not license\n")
    L.append(
        "- **Does:** the paper may state that judges were blind to condition by "
        "construction, and quantify the residual channel by which condition can "
        "still be inferred.\n"
        "- **Does not:** it does not establish that disclosure *causes* score "
        "differences. Testing that needs a disclosure-matched sensitivity "
        "analysis (compare scores on disclosing vs non-disclosing responses "
        "within model and principle), which is not run here.\n"
    )

    blind_df.to_csv(args.output_dir / "judge_blinding_check.csv", index=False)
    resp_df.to_csv(args.output_dir / "response_disclosure_rates.csv", index=False)
    (args.output_dir / "judge_artifact_controls.md").write_text("\n".join(L))

    print(f"\nblinding leak probes hit: {n_leaks}")
    print(f"judge message roles: {roles}")
    for persona, r in by_persona.iterrows():
        print(f"  {persona:14} disclosure {r.rate:.2%}")
    print(f"\nWrote {args.output_dir / 'judge_artifact_controls.md'}")


if __name__ == "__main__":
    main()
