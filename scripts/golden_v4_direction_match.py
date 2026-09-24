"""Re-measure ensemble-vs-human direction agreement on the 24 golden items under rubric v4.

The published 23/24 (95.8%) direction match was measured with rubric v3 and does not carry
over: v4 changes both the prompt and what a judge may return. This script judges the same
pre-generated responses in `data/golden_questions.jsonl` with `rubrics/judge_prompt_v4.md`,
using the cross-family ensemble the transcript-score skill recommends, and compares the
ensemble's score on each item's target principle with the human consensus.

Under v4.1 the runner also drops a negative whose quoted evidence is not verbatim in the
response, as the CLI and the skill do, and the counts of those drops are reported.

v4 adds outcomes that carry no score. An item whose target principle came back
`not_applicable`, `insufficient_context` or `covered` from every judge (or only as
low-confidence scores, which v4 drops) has no ensemble score. Such items are reported
separately and counted two ways: excluded (agreement among items that scored) and as misses
(agreement over all 24). Neither number is comparable to the v3 figure.

Usage (needs OPENROUTER_API_KEY and `blake3`):
    python scripts/golden_v4_direction_match.py --out docs/validation/golden_v4.1_direction_match_<date>.json
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "skills" / "humanebench-transcript-score" / "scripts"))
import humanebench_score as hb  # noqa: E402

SLUG_TO_CODE = {
    "respect-user-attention": "respect_attention",
    "enable-meaningful-choices": "meaningful_choices",
    "enhance-human-capabilities": "enhance_capabilities",
    "protect-dignity-and-safety": "dignity_safety",
    "foster-healthy-relationships": "healthy_relationships",
    "prioritize-long-term-wellbeing": "longterm_wellbeing",
    "be-transparent-and-honest": "transparency_honesty",
    "design-for-equity-and-inclusion": "equity_inclusion",
}
PARSE_ATTEMPTS = 3  # the v3 golden run used score_attempts=3


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def judge_item(rubric, item, model):
    prompt = hb.assemble_turn_prompt(rubric, {
        "user_prompt": item["input"], "assistant_text": item["metadata"]["ai_output"],
        "actions": []})
    last = None
    cost = 0.0
    for _ in range(PARSE_ATTEMPTS):
        try:
            text, pinned, usage = hb.openrouter_complete(prompt, model)
            cost += float(usage.get("cost") or 0.0)
            j = hb.parse_judgement(text)
            hb.verify_evidence(j, item["metadata"]["ai_output"])
            p = next(p for p in j["principles"] if p["name"] == SLUG_TO_CODE[item["target"]])
            return {"outcome": p["outcome"], "score": p.get("score"),
                    "confidence": p.get("confidence"), "counts": hb.counts(p),
                    "quote_unverified": p.get("quote_unverified", False),
                    "evidence": p.get("evidence") or [],
                    "temperature_pinned": pinned, "cost_usd": cost}
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    return {"outcome": "error", "error": last, "counts": None, "cost_usd": cost}


def wilson(k, n):
    if n == 0:
        return None
    from scipy.stats import binomtest
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return [round(ci.low, 3), round(ci.high, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "golden_questions.jsonl")
    args = ap.parse_args()

    rubric = (REPO / "rubrics" / "judge_prompt_v4.md").read_text()
    items = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    models = hb.ENSEMBLE_MODELS
    jobs = [(i, m) for i in range(len(items)) for m in models]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda im: judge_item(rubric, items[im[0]], im[1]), jobs))
    by_item = {}
    for (i, m), r in zip(jobs, results):
        by_item.setdefault(i, {})[m] = r

    rows = []
    for i, item in enumerate(items):
        md = item["metadata"]
        human = [float(x) for x in md["human_rating_values"].split(",")]
        consensus = sum(human) / len(human)
        judged = by_item[i]
        counted = [r["counts"] for r in judged.values() if r["counts"] is not None]
        ens = sum(counted) / len(counted) if counted else None
        rows.append({
            "id": item["id"], "target": SLUG_TO_CODE[item["target"]],
            "human_consensus": consensus, "ensemble_score": ens,
            "direction_match": None if ens is None else sign(ens) == sign(consensus),
            "judges": judged,
        })

    errors = [r["id"] for r in rows if any(j["outcome"] == "error" for j in r["judges"].values())]
    if errors:
        # A failed call is not a measurement. Refuse rather than report an agreement rate
        # over judges that never answered.
        print(json.dumps({i: [j.get("error") for j in by_item[i].values()] for i in range(len(items))
                          if rows[i]["id"] in errors}, indent=2)[:4000], file=sys.stderr)
        sys.exit(f"{len(errors)} item(s) had failed judge calls; nothing written")

    scored = [r for r in rows if r["ensemble_score"] is not None]
    matches = sum(r["direction_match"] for r in scored)
    per_judge = {}
    for m in models:
        s = [(r["judges"][m]["counts"], r["human_consensus"]) for r in rows
             if r["judges"][m]["counts"] is not None]
        per_judge[m] = {"scored": len(s), "direction_matches": sum(sign(a) == sign(h) for a, h in s),
                        "outcomes": {o: sum(r["judges"][m]["outcome"] == o for r in rows)
                                     for o in ("score", "not_applicable", "insufficient_context",
                                               "covered", "error")},
                        "low_confidence_dropped": sum(r["judges"][m].get("confidence") == "low"
                                                      and r["judges"][m]["outcome"] == "score"
                                                      for r in rows),
                        "unverified_quote_dropped": sum(bool(r["judges"][m].get("quote_unverified"))
                                                        for r in rows)}
    summary = {
        "measured_at": datetime.now(timezone.utc).date().isoformat(),
        "rubric": "rubrics/judge_prompt_v4.md",
        "rubric_version": hb.RUBRIC_VERSION,
        "rubric_hash": hb.rubric_hash(rubric),
        "ensemble": [hb.judge_label(m) for m in models],
        "items": len(rows),
        "items_with_ensemble_score": len(scored),
        "direction_matches": matches,
        "match_rate_among_scored": round(matches / len(scored), 4) if scored else None,
        "match_rate_among_scored_wilson95": wilson(matches, len(scored)),
        "match_rate_all_items_unscored_as_miss": round(matches / len(rows), 4),
        "match_rate_all_items_wilson95": wilson(matches, len(rows)),
        "per_judge": per_judge,
        "cost_usd": round(sum(j.get("cost_usd", 0.0) for r in rows for j in r["judges"].values()), 4),
        "rule": "direction match = sign(ensemble mean of counted target-principle scores) == "
                "sign(mean human rating); low-confidence scores and negatives whose quote is not "
                "verbatim in the response are dropped, per v4.1",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
