#!/usr/bin/env bash
set -euo pipefail

# Publish figures and per-model score JSON from this repo to the website repo.
# Usage: ./scripts/publish_figures_to_website.sh [website-repo-path]

WEBSITE_REPO=${1:-"../humanebench-website"}
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIGURES_SRC="${SRC_DIR}/figures"
FIGURES_DST="${WEBSITE_REPO}/public/figures"
SCORES_DST="${WEBSITE_REPO}/public/data/model_scores.json"

if [[ ! -d "${WEBSITE_REPO}" ]]; then
  echo "Website repo not found at ${WEBSITE_REPO}" >&2
  exit 1
fi

if [[ ! -d "${FIGURES_SRC}" ]]; then
  echo "Figures directory not found at ${FIGURES_SRC}" >&2
  exit 1
fi

echo "Publishing figures from ${FIGURES_SRC} to ${FIGURES_DST}"
rm -rf "${FIGURES_DST}"
mkdir -p "${FIGURES_DST}"
cp -a "${FIGURES_SRC}/." "${FIGURES_DST}/"

echo "Regenerating ${SCORES_DST} from canonical CSVs"
python3 "${SRC_DIR}/scripts/export_model_scores_json.py" --output "${SCORES_DST}"

echo "Done."
