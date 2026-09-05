from typing import Iterable
import regex as re
from .tokenizer import Tokenizer

from pathlib import Path

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
    ts_valid: ("TinyStoriesV2-GPT4-valid_vocab.json","TinyStoriesV2-GPT4-valid-merges.json"),
    owt_train: ("owt_train_vocab.json", "owt_train_merges.json"),
    owt_valid: ("owt_valid_vocab.json", "owt_valid_merges.json")
}

def sample_docs(*, dataset, tokenizer):
    
    fpath = Path(__file__).parent.parent / "data" / datasets[dataset]
    v_path, m_path = vocabs_and_merges[tokenizer]
    
    tok = Tokenizer.from_files(v_path, m_path, special_tokens = ["<|endoftext|>"])
    
    n_docs = 0
    n_bytes = 0
    n_toks = 0
    max_docs = 10
    with open(fpath, 'r') as f:
        while n_docs < max_docs:
            newline = f.readline()
            n_bytes += len(newline.encode("utf-8"))
            if "<|endoftext|>" in newline:
                n_docs += 1
            n_toks += len(tok.encode(newline))
    
    print(f"Compression ratio for {dataset} using {tokenizer} {n_bytes/n_toks}")



if __name__ == "__main__":
    # Part A
    sample_docs(dataset=ts_train, tokenizer=ts_train)
    sample_docs(dataset=owt_train, tokenizer=owt_train)
    # Part B
    sample_docs(dataset=owt_train, tokenizer=ts_train)