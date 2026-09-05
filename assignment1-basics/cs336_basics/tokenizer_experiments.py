from typing import Iterable
import regex as re
from .tokenizer import Tokenizer

from pathlib import Path

import numpy as np

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

def sample_docs(*, dataset, tokenizer, save=False):
    
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
            if save:
                all_toks.extend(toks)
    if save:
        out_path = fpath.with_suffix(".npy")
        np.save(out_path, np.array(all_toks, dtype=np.uint16))
    print(f"Compression ratio for {dataset} using {tokenizer} {n_bytes/n_toks}")

def convert_docs(*, dataset, tokenizer):
    fpath = Path(__file__).parent.parent / "data" / datasets[dataset]
    v_path, m_path = vocabs_and_merges[tokenizer]
    
    
    tok = Tokenizer.from_files(v_path, m_path, special_tokens = ["<|endoftext|>"])
    opath = fpath.with_suffix(".npy")
    read_bytes = 0
    flush_bytes = 5_000_000
    chunks = []
    np_chunks = []
    size = fpath.stat().st_size
    tot_read = 0
    with open(fpath, 'r') as in_f, open(opath, 'wb') as out_f:
        while chunk := in_f.readline():
            chunks.extend(tok.encode(chunk))
            if len(chunks) > flush_bytes:
                tot_read += len(chunks)
                np_chunks.append(np.array(chunks, dtype=np.uint16))
                print(f"Wrote {4*tot_read} of {size} bytes: {4*tot_read/size}")
                chunks = []
        np_chunks.append(np.array(chunks, dtype=np.uint16))
        out_chunks = np.concatenate(np_chunks)
        np.save(out_f, out_chunks)

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