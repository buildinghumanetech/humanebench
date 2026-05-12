#!/usr/bin/env python3
"""Batch-evaluate a JSONL of {user_prompt, response} entries via HumaneBench.

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
from humanebench_evaluator import evaluate  # noqa: E402

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
        scores = {p["name"]: p["score"] for p in result["principles"]}
        rationales = {
            p["name"]: p.get("rationale", "")
            for p in result["principles"]
            if p["score"] < 0 and p.get("rationale")
        }
        return {
            "id": entry.get("id"),
            "timestamp": started,
            "user_prompt": entry["user_prompt"],
            "response": entry["response"],
            "principle_focus": entry.get("principle_focus"),
            "expected_severity": entry.get("expected_severity"),
            "scores": scores,
            "humane_score": sum(scores.values()) / len(scores),
            "global_violations": result["globalViolations"],
            "rationales": rationales,
            "confidence": result["confidence"],
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
            "scores": None,
            "humane_score": None,
            "global_violations": None,
            "rationales": None,
            "confidence": None,
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

    with output_path.open(mode) as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_one, e, args.api_key, args.model, args.base_url): e
            for e in entries
        }
        for fut in as_completed(futures):
            row = fut.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            completed += 1
            status = "ERROR" if row["error"] else f"score={row['humane_score']:+.2f}"
            print(f"  [{completed}/{len(entries)}] {row['id']}: {status}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\nWrote {completed} results to {output_path} in {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
