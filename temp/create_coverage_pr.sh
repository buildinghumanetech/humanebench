#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

gh pr create \
  --base main \
  --head coverage-post-exclusion-summary \
  --title "Add post-exclusion coverage section to analyze_coverage.py" \
  --body-file temp/pr_body.md
