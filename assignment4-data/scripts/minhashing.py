import re
import unicodedata
import mmh3
from collections import defaultdict
import itertools
import random
import os
from pathlib import Path
import shutil

CLEAN_TEXT = re.compile(r'[^\w\s]')

def normalize(text: str) -> str:
    """To improve recall
(following [7]), normalize the text before computing minhash signatures and/or comparing
Jaccard similarity by lowercasing, removing punctuation, normalizing whitespaces, and removing
accents, and applying NFD unicode normalization."""
    text = CLEAN_TEXT.sub("", text).strip().lower() # removes punctuation and whitespace
    normalized = unicodedata.normalize("NFD", text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))

def make_hash_list(num_hashes):
    return [lambda text, i=i: mmh3.hash(key=text, seed=i) for i in range(num_hashes)]    

Ngram = list[str]
Band = tuple[int, ...]
DocBands = list[Band]

def make_ngrams(normalizedtext: str, ngram_length: int) -> Ngram:
    words = normalizedtext.split()
    chunks = [" ".join(words[i:i+ngram_length]) for i in range(0, len(words) - (ngram_length - 1))]
    return chunks

def minhash(normalizedtext: str, *, num_hashes: int, ngrams: Ngram)->tuple[int, ...]:
    hashes = make_hash_list(num_hashes)
    min_hash = []
    for hash_fn in hashes:
        hash_signatures = []
        for ngram in ngrams:
            hash_signatures.append(hash_fn(ngram))
        min_hash.append(min(hash_signatures))
    return tuple(min_hash)

def LSH(hashes: tuple[int, ...], *, num_bands) -> DocBands:
    assert len(hashes) % num_bands == 0, "the bands must equally divide them"
    band_width = len(hashes) // num_bands
    print(band_width)
    bands = [hashes[i:i+band_width] for i in range(0, len(hashes), band_width)]
    assert len(bands) == num_bands, f"{len(bands)=}, {num_bands=}"
    return bands

def bucket_documents(doc_list: list[DocBands], doc_ids: list[int]):
    assert len(doc_list) == len(doc_ids)
    band_hashes = defaultdict(int)
    for doc in doc_list:
        for band in doc:
            band_hashes[band] += 1
    buckets: defaultdict[Band, list[int]] = defaultdict(list)
    for bands, doc_id in zip(doc_list, doc_ids):
        for band in bands:
            if band_hashes[band] > 1:
                buckets[band].append(doc_id)
    return buckets

def buckets_to_comparisons(buckets: dict[Band, list[int]]) -> set[tuple[int, int]]:
    # a -> [1,2,3] -> ((1,2), (1,3), (2,3))
    # b -> [2,3,4] -> ((2,3), (2,4), (3,4))
    # c -> [0,4,5] -> ((4,5), (0,4), (0,5))
    # a | b | c 
    big_set: set[tuple[int, int]] = set()
    for value in buckets.values():
        for combo in itertools.combinations(value, 2):
            big_set.add(combo)
    return big_set

def make_comparisons(comparisons: set[tuple[int, int]], all_docs:  list[Ngram], threshold=0.75) -> dict[int, list[int]]:
    dupes = defaultdict(list[int])
    for i in range(len(all_docs)):
        dupes[i] = []
    for ngram1, ngram2 in comparisons:
        doc1 = all_docs[ngram1]
        doc2 = all_docs[ngram2]
        value = jaccard_similarity(doc1, doc2)
        if value > threshold:
            print("Duplicate detected: ", ngram1, ngram2)
            dupes[ngram1].append(ngram2)
            dupes[ngram2].append(ngram1)
    return dupes

def dupes_to_chains(dupes: dict[int, list[int]]):
    visited_docs = set()
    clusters = []
    for k in dupes.keys():
        if k in visited_docs:
            continue
        cluster = []
        cluster.append(k)
        visited_docs.add(k)
        todo = list(dupes[k])
        while len(todo) > 0:
            v = todo.pop()
            if v in visited_docs:
                continue
            visited_docs.add(v)
            cluster.append(v)
            todo.extend(dupes[v])
        clusters.append(cluster)
    return clusters

def select_clusters(candidates: list[list[int]]):
    return [random.choice(candidate) for candidate in candidates]

def jaccard_similarity(ngram1: Ngram, ngram2: Ngram):
    A = set(ngram1)
    B = set(ngram2)
    return len(A & B) / len(A | B)
    
def minhash_dedup(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    of_dir = Path(output_directory)
    all_hashes: list[Band] = []
    all_ngrams: list[Ngram] = []
    lsh_list: list[DocBands] = []
    for fname in input_files:
        fname = Path(fname)
        with open(fname, "r") as f:
            text = f.read()
            text = normalize(text)
        all_ngrams.append(make_ngrams(text, ngrams))
        all_hashes.append(minhash(text, num_hashes=num_hashes, ngrams=all_ngrams[-1]))
        lsh_list.append(LSH(all_hashes[-1], num_bands=num_bands))
    doc_ids = list(range(len(input_files)))
    buckets = bucket_documents(lsh_list, doc_ids)
    comparisons: set[tuple[int, int]] = buckets_to_comparisons(buckets)
    dupes = make_comparisons(comparisons, all_ngrams, threshold=jaccard_threshold)
    chains = dupes_to_chains(dupes)
    clusters = select_clusters(chains)
    kept_files = [input_files[i] for i in clusters]
    for file in kept_files:
        fp = Path(file)
        shutil.copy(file, of_dir / fp.name)
    
if __name__ == "__main__":
    # print(normalize("CAFé, stávänger, and façade"))
    # mh = minhash("abcde fghijkl mnopqr stuv wx y z", num_hashes = 30, ngram_length = 4)
    # print(LSH(mh, num_bands=5))
    d = {0: [1], 1:[0, 2], 2:[1], 3:[4], 4:[3]}
    chains = dupes_to_chains(d)
    print(chains)
    print(select_clusters(chains))