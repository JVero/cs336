#!/usr/bin/env bash
# Download the assignment 1 datasets into assignment1-basics/data/.
#
#   scripts/download_a1_data.sh          # TinyStories only (~2.1 GB)
#   scripts/download_a1_data.sh --owt    # also OpenWebText sample (~4.4 GB gz, ~12 GB unzipped)
#
# Resumable: re-running picks up partial downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/assignment1-basics/data"
mkdir -p "$DATA"
cd "$DATA"

fetch() {
  local url="$1" out="$2"
  if [[ -f "$out" && ! -f "$out.part" ]]; then
    echo "have $out"
    return
  fi
  echo "fetching $out"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$out.part" "$url"
  mv "$out.part" "$out"
}

TS=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
fetch "$TS/TinyStoriesV2-GPT4-train.txt" TinyStoriesV2-GPT4-train.txt
fetch "$TS/TinyStoriesV2-GPT4-valid.txt" TinyStoriesV2-GPT4-valid.txt

if [[ "${1:-}" == "--owt" ]]; then
  OWT=https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main
  for split in train valid; do
    if [[ -f "owt_${split}.txt" ]]; then
      echo "have owt_${split}.txt"
      continue
    fi
    fetch "$OWT/owt_${split}.txt.gz" "owt_${split}.txt.gz"
    echo "gunzipping owt_${split}.txt.gz"
    gunzip "owt_${split}.txt.gz"
  done
fi

echo
ls -lh "$DATA"
