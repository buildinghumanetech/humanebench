#!/usr/bin/env bash
# Create the stacked PR for figs-with-CIs (base: bootstrap-confidence-intervals).
# Run from the repo root: `bash temp/create_pr_figs_with_cis.sh`
# Requires: gh CLI authenticated, branch already pushed.

set -euo pipefail

gh pr create \
  --base bootstrap-confidence-intervals \
  --title "Render bootstrap CI whiskers on steerability candlestick" \
  --body-file temp/pr_body_figs_with_cis.md
