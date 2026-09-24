#!/usr/bin/env python3
"""Batch-evaluate a JSONL of {user_prompt, response} entries via HumaneBench rubric v4.

Workshop helper: takes the mock conversations file (or any JSONL with the
same shape) and writes evaluation results to a second JSONL that the
Streamlit dashboard can read.

Usage:
    python batch_evaluate.py \\
        --input conversations.jsonl \\
        --output results.jsonl \\
        --model openai/gpt-4o-mini

Set OPENROUTER_API_KEY in your environment (or pass --api-key).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Make the parent evaluator package importable when run from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanebench_evaluator import (  # noqa: E402
    DIRECTIONAL_CONTEXT_BLOCKED_RATE,
    RUBRIC_VERSION,
    counted_score,
    evaluate,
    overall_score,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def evaluate_one(entry: dict, api_key: str, model: str, base_url: str) -> dict:
    """Score one conversation; return a flat result dict ready for JSONL output."""
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = evaluate(
            user_prompt=entry["user_prompt"],
            message_content=entry["response"],
            llm_provider="openai",
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        principles = result["principles"]
        # Only a counted score goes in `scores`: not_applicable, insufficient_context,
        # covered and low-confidence scores are None, never 0, so no mean includes them.
        scores = {p["name"]: counted_score(p) for p in principles}
        rationales = {
            p["name"]: p["rationale"]
            for p in principles
            if counted_score(p) is not None and p["score"] < 0 and p.get("rationale")
        }
        return {
            "id": entry.get("id"),
            "timestamp": started,
            "rubric_version": RUBRIC_VERSION,
            "user_prompt": entry["user_prompt"],
            "response": entry["response"],
            "principle_focus": entry.get("principle_focus"),
            "expected_severity": entry.get("expected_severity"),
            "scores": scores,
            "outcomes": {p["name"]: p["outcome"] for p in principles},
            "confidences": {p["name"]: p.get("confidence") for p in principles},
            "questions": {
                p["name"]: p["question"] for p in principles if p["outcome"] == "insufficient_context"
            },
            # Mean over the principles that counted; None ("not in scope") when none did.
            "humane_score": overall_score(result),
            "coverage": result["coverage"],
            "covered": result["covered"],
            "notes": result["notes"],
            "rationales": rationales,
            "model": model,
            "error": None,
        }
    except Exception as e:
        return {
            "id": entry.get("id"),
            "timestamp": started,
            "user_prompt": entry.get("user_prompt"),
            "response": entry.get("response"),
            "principle_focus": entry.get("principle_focus"),
            "expected_severity": entry.get("expected_severity"),
            "rubric_version": RUBRIC_VERSION,
            "scores": None,
            "outcomes": None,
            "confidences": None,
            "questions": None,
            "humane_score": None,
            "coverage": None,
            "covered": None,
            "notes": None,
            "rationales": None,
            "model": model,
            "error": str(e),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", default="conversations.jsonl", help="Input JSONL (default: conversations.jsonl)")
    parser.add_argument("--output", "-o", default="results.jsonl", help="Output JSONL (default: results.jsonl)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"), help="API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwriting")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("No API key. Set OPENROUTER_API_KEY or pass --api-key.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")

    entries = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(entries)} conversations from {input_path}", file=sys.stderr)
    print(f"Judge: {args.model} via {args.base_url}", file=sys.stderr)

    mode = "a" if args.append else "w"
    start = time.time()
    completed = 0
    rows: list[dict] = []

    with output_path.open(mode) as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, e, args.api_key, args.model, args.base_url): e
            for e in entries
        }
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            completed += 1
            if row["error"]:
                status = "ERROR"
            elif row["humane_score"] is None:
                status = "not in scope"
            else:
                status = f"score={row['humane_score']:+.2f}"
            print(f"  [{completed}/{len(entries)}] {row['id']}: {status}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\nWrote {completed} results to {output_path} in {elapsed:.1f}s", file=sys.stderr)
    for line in summarize(rows):
        print(line, file=sys.stderr)
    return 0


def summarize(rows: list[dict]) -> list[str]:
    """Run-level summary lines. Non-scores never enter the mean as 0."""
    ok = [r for r in rows if not r["error"]]
    if not ok:
        return ["No successful evaluations."]
    counted = [r["humane_score"] for r in ok if r["humane_score"] is not None]
    overall = f"{sum(counted) / len(counted):+.2f}" if counted else "not in scope (nothing scored)"
    in_scope = sum(r["coverage"]["applicable"] for r in ok)
    blocked = sum(r["coverage"]["context_blocked"] for r in ok)
    dropped = sum(
        1 for r in ok for name, outcome in r["outcomes"].items()
        if outcome == "score" and r["confidences"].get(name) == "low"
    )
    lines = [
        f"HumaneScore (rubric {RUBRIC_VERSION}): {overall} over {len(counted)} of {len(ok)} conversations with a counted score",
        f"Low-confidence scores dropped: {dropped}",
    ]
    if in_scope:
        rate = blocked / in_scope
        lines.append(f"Context-blocked: {rate:.0%} of in-scope principle-turns")
        if rate > DIRECTIONAL_CONTEXT_BLOCKED_RATE:
            lines.append(
                "DIRECTIONAL, NOT DEFINITIVE: more than 15% of in-scope principle-turns "
                "came back insufficient_context."
            )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
