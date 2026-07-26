#!/usr/bin/env bash
# Create the PR for the bootstrap-confidence-intervals branch.
# Run from the repo root: `bash temp/create_pr.sh`
# Requires: gh CLI authenticated, branch already pushed.

set -euo pipefail

gh pr create \
  --title "Add 95% bootstrap CIs to headline HumaneBench scores" \
  --body-file temp/pr_body.md
