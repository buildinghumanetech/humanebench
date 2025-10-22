#!/usr/bin/env bash
set -euo pipefail

# Usage: ./inspect-run.sh [model] [extra inspect args...]
MODEL="${1:-openai/gpt-5-mini}"
shift || true

logdir="logs/$(date -u +%Y-%m-%dT%H-%M-%SZ)"
mkdir -p "$logdir"
echo "Writing logs to: $logdir"

inspect eval-set src/good_persona_task.py src/bad_persona_task.py --model "$MODEL" --log-dir "$logdir" "$@"