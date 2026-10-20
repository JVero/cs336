"""Build a PARTIAL enwiki-20260501-extracted_urls.txt.gz from the first few dump shards.

The full file comes from assignment4-data/scripts/download_data.py (all 70 shards, ~24.5 GB,
meant to run on Modal). The quality classifier only needs a subsample of positive URLs, so this
runs the same extraction locally on a handful of shards. Shards stream straight from Wikimedia
and are never written to disk.

    cd assignment4-data && uv run ../scripts/a4_wiki_urls_partial.py --shards 3
"""

import argparse
import bz2
import gzip
import re
import urllib.request
from pathlib import Path

DUMP_DATE = "20260501"
BASE_URL = f"https://dumps.wikimedia.org/enwiki/{DUMP_DATE}/"
OUT = Path(__file__).resolve().parents[1] / "assignment4-data/local-shared-data/wiki/enwiki-20260501-extracted_urls.txt.gz"

# Copied from download_data.py so the partial file matches the full one line for line.
URL_RE = re.compile(
    r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
)
REF_RE = re.compile("&lt;ref&gt(.*)&lt;/ref&gt;")

# Wikimedia returns 403 to urllib's default User-Agent.
HEADERS = {"User-Agent": "cs336-selfstudy-url-extract/0.1 (personal coursework)"}


def fetch(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS))


def list_shards() -> list[str]:
    html = fetch(BASE_URL).read().decode()
    found = set(re.findall(rf"enwiki-{DUMP_DATE}-pages-articles-multistream[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2", html))
    # Sort by shard number, not as strings, so "10" doesn't come before "2".
    return sorted(found, key=lambda s: int(re.search(r"multistream([0-9]+)\.xml", s).group(1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=3, help="How many shards to process, starting from the first.")
    args = parser.parse_args()

    shards = list_shards()[: args.shards]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".partial.tmp")
    total = 0
    with gzip.open(tmp, "wt") as out:
        for shard in shards:
            n = 0
            with fetch(BASE_URL + shard) as r, bz2.open(r, "rt", errors="ignore") as f:
                for line in f:
                    if refs := REF_RE.search(line):
                        for url in URL_RE.findall(refs.group(0)):
                            out.write(url + "\n")
                            n += 1
            total += n
            print(f"[wiki] {shard}: {n:,} urls", flush=True)
    tmp.rename(OUT)
    print(f"[wiki] wrote {total:,} urls from {len(shards)} shards to {OUT}", flush=True)


if __name__ == "__main__":
    main()
