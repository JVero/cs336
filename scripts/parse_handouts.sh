#!/usr/bin/env bash
# Extract every assignment handout PDF to plain text in handouts/ (gitignored).
# -layout keeps fractions stacked (numerator above the line), which plain mode drops.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p handouts
for pdf in assignment*/*.pdf; do
  out="handouts/$(basename "${pdf%.pdf}").txt"
  pdftotext -layout "$pdf" "$out"
  echo "$out"
done
