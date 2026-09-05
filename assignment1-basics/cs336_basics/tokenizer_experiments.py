from typing import Iterable
import regex as re
from .tokenizer import Tokenizer

from pathlib import Path

import numpy as np

from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries
from itertools import pairwise

ts_train = "tinystories-train"
ts_valid = "tinystories-valid"
owt_train = "owt-train"
owt_valid = "owt-valid"

datasets = {
    ts_train : "TinyStoriesV2-GPT4-train.txt",
    ts_valid: "TinyStoriesV2-GPT4-valid.txt",
    owt_train:  "owt_train.txt",
    owt_valid: "owt_valid.txt",
}

vocabs_and_merges = {
    ts_train: ("TinyStoriesV2-GPT4-train_vocab.json", "TinyStoriesV2-GPT4-train_merges.json"),
    ts_valid: ("TinyStoriesV2-GPT4-valid_vocab.json","TinyStoriesV2-GPT4-valid_merges.json"),
    owt_train: ("owt_train_vocab.json", "owt_train_merges.json"),
    owt_valid: ("owt_valid_vocab.json", "owt_valid_merges.json")
}

tok = None

def build_tokenizer(v_path, m_path):
    global tok
    tok = Tokenizer.from_files(v_path, m_path, special_tokens = ["<|endoftext|>"])


def process_chunk(fpath, start, stop):
    global tok
    chunks = []
    with open(fpath, 'rb') as f:
        f.seek(start)
        text = f.read(stop-start).decode("utf-8")
        for line in text.splitlines(keepends=True):
            chunks.extend(tok.encode(line))
        return np.array(chunks, dtype=np.uint16)

def sample_docs(*, dataset, tokenizer):
    fpath = Path(__file__).parent.parent / "data" / datasets[dataset]
    v_path, m_path = vocabs_and_merges[tokenizer]
    
    tok = Tokenizer.from_files(v_path, m_path, special_tokens = ["<|endoftext|>"])
    
    n_docs = 0
    n_bytes = 0
    n_toks = 0
    max_docs = 10
    all_toks = []
    with open(fpath, 'r') as f:
        while n_docs < max_docs:
            newline = f.readline()
            n_bytes += len(newline.encode("utf-8"))
            if "<|endoftext|>" in newline:
                n_docs += 1
            toks = tok.encode(newline)
            n_toks += len(toks)

def convert_docs(*, dataset, tokenizer,  chunk_size=100_000_000, n_worker=12):
    fpath = Path(__file__).parent.parent / "data" / datasets[dataset]
    v_path, m_path = vocabs_and_merges[tokenizer]
    
    n_chunk = 1+fpath.stat().st_size // chunk_size
    
    with open(fpath, 'rb') as f:    
        offsets = find_chunk_boundaries(f, n_chunk, bytes("<|endoftext|>", encoding="utf-8"))
    
    with Pool(n_worker, initializer=build_tokenizer, initargs=(v_path, m_path)) as p:
        chunks = p.starmap(process_chunk, [(fpath, start, stop) for start, stop in pairwise(offsets)])
    
    opath = fpath.with_suffix(".npy")
    
    out_chunks = np.concatenate(chunks, dtype=np.uint16)
    np.save(opath, out_chunks)

if __name__ == "__main__":
    # Part A
    sample_docs(dataset=ts_train, tokenizer=ts_train)
    sample_docs(dataset=owt_train, tokenizer=owt_train)
    # Part B
    sample_docs(dataset=owt_train, tokenizer=ts_train)
    # Part C
    # Timed run
    
    # Part D
    convert_docs(dataset=ts_valid, tokenizer=ts_train)
    convert_docs(dataset=owt_valid, tokenizer=owt_train)
    convert_docs(dataset=ts_train, tokenizer=ts_train)
    convert_docs(dataset=owt_train, tokenizer=owt_train)