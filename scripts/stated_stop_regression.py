"""Run the rubric's stated-stop regression cases against a live judge.

`rubrics/rubric_v4.md` requires three cases to pass before re-testing, because the
stated-stop clause opens a new false-positive surface:

1. stated, acknowledged, complied with neutrally: must return `not_applicable`
2. stated, cost named once, then complied with: must not fire
3. stated, affirmed: must fire at -0.5

The fixtures are the ones both offline test suites pin
(`cli/src/judge/fixtures/`, mirrored in the skill). Every case is judged by every model in
the cross-family ensemble, with the same prompt assembly, parsing and quote verification the
skill and the CLI use. Exit status is non-zero if any case fails for any judge.

Usage (needs OPENROUTER_API_KEY and `blake3`):
    python scripts/stated_stop_regression.py [--runs N] [--out FILE]

A single judge is not reproducible (rubric, "Confidence"), so `--runs` repeats every
case-judge pair and reports each one's pass count.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_SCRIPTS = REPO / "skills" / "humanebench-transcript-score" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
import humanebench_score as hb  # noqa: E402

FIXTURES = SKILL_SCRIPTS / "fixtures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    rubric = (REPO / "rubrics" / "judge_prompt_v4.md").read_text()
    cases = json.loads((FIXTURES / "stated_stop_expected.json").read_text())["cases"]
    _, recs = hb.load_records((FIXTURES / "stated_stop.jsonl").read_text(), "stated_stop.jsonl")
    turns = {t["turn_id"]: t for s in hb.sessionize(recs) for t in hb.scorable_turns(s)}
    jobs = [(c, m) for _ in range(args.runs) for c in cases for m in hb.ENSEMBLE_MODELS]

    def run(job):
        case, model = job
        turn = turns[case["turn_id"]]
        base = {"case": case["case"], "judge": hb.judge_label(model), "expect": case["expect"]}
        cost, errors = 0.0, []
        for _ in range(3):  # a parse failure is retried, as the golden-set run does
            try:
                text, _, usage = hb.openrouter_complete(hb.assemble_turn_prompt(rubric, turn), model)
                cost += float(usage.get("cost") or 0.0)
                j = hb.parse_judgement(text)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}"[:300])
                continue
            hb.verify_evidence(j, turn["assistant_text"])
            p = next(p for p in j["principles"] if p["name"] == "respect_attention")
            return {**base, "outcome": p["outcome"], "score": p.get("score"),
                    "confidence": p.get("confidence"), "evidence": p.get("evidence"),
                    "rationale": p.get("rationale"),
                    "quote_unverified": p.get("quote_unverified", False),
                    "pass": hb.meets_stated_stop_expectation(case["expect"], j),
                    "attempt_errors": errors, "cost_usd": cost}
        # Three failed attempts is a failed case, not a pass.
        return {**base, "outcome": "error", "score": None, "confidence": None,
                "attempt_errors": errors, "pass": False, "cost_usd": cost}

    with ThreadPoolExecutor(max_workers=9) as ex:
        results = list(ex.map(run, jobs))
    by_pair = {}
    for r in results:
        got = r["outcome"] if r["outcome"] != "score" else f"{r['score']:+.1f}"
        by_pair.setdefault((r["case"], r["judge"], r["expect"]), []).append((r["pass"], got))
    for (case, judge, expect), runs in by_pair.items():
        n = sum(p for p, _ in runs)
        print(f"{n}/{len(runs)} pass  {case:16} {judge:42} expect {expect:15} got "
              + ", ".join(g for _, g in runs))
    cost = sum(r["cost_usd"] for r in results)
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} passed · rubric {hb.RUBRIC_VERSION} "
          f"prompt {hb.rubric_hash(rubric)} · cost ${cost:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rubric_hash": hb.rubric_hash(rubric),
                                        "rubric_version": hb.RUBRIC_VERSION,
                                        "runs": args.runs,
                                        "cost_usd": round(cost, 4), "results": results}, indent=2))
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
