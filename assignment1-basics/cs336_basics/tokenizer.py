from typing import Iterable, Iterator

import json
import regex as re
from itertools import pairwise
from .pretokenization_example import find_chunk_boundaries

from .train_bpe import build_occurrences

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens=None,
                 num_chunks=120,
                 num_workers=12):
        # int -> bytes
        self.vocab = vocab 
        # bytes -> int
        self.i_vocab = {v: k for k, v in vocab.items()}
        # print(type(self.i_vocab[list(self.i_vocab.keys())[0]])); exit() type<int>
        self.merges = merges

        # Breaks ties for encoding
        self.merge_ranks = {}
        
        # Identifies each merge
        self.merge_ids = {}
        for i, (l, r) in enumerate(self.merges):
            merge_key = (self.i_vocab[l], self.i_vocab[r])
            self.merge_ranks[merge_key] = i
            self.merge_ids[(self.i_vocab[l], self.i_vocab[r])] = self.i_vocab[l + r]
            
        self.special_tokens = special_tokens

        self.st_pattern = "("+"|".join([re.escape(st) for st in self.special_tokens]) +")" if self.special_tokens else None
        self.num_chunks = num_chunks
        self.num_workers = num_workers
        
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        a file is a big json file dictionary
        """
        vocab = json.load(vocab_filepath)
        merges = json.load(merges_filepath)
        return cls(vocab, merges, special_tokens=special_tokens)
        
    def encode(self, text: str) -> list[int]:
        rval = []
        docs = re.splititer(self.st_pattern, text) if self.st_pattern else [text]

        for doc in docs:
            if doc in self.special_tokens:
                rval.append( self.i_vocab[doc] )
                continue
            pretokens = re.finditer(PAT, doc)
            
            # Convert the string into a list of pretokens
            # then those pretokens turn into lists of bytes
            pretoken_bytes = []
            for pretoken in pretokens:
                pretoken_bytes.append(list(pretoken[0].encode("utf-8")))
            
            # For each "word", apply the possible merges from left to right
            # for lots of merges, it would make sense to check that list once and 
            # the pretoken_bytes many times
            for ptb in pretoken_bytes:
                more_ranks = True
                while more_ranks:
                    new_ptb = []
                    ranks = [(pair, self.merge_ranks.get(pair, None)) for pair in pairwise(ptb) if pair in self.merge_ranks.keys()]
                    if not ranks: # there are no merges to do
                        more_ranks = False
                    else:
                        best_merge, _ = min(ranks, key=lambda k: k[1])
                        i = 0
                        while i < len(ptb):
                            if i != len(ptb) - 1 and (ptb[i], ptb[i+1]) == best_merge:
                                new_ptb.append(self.merge_ids[(ptb[i], ptb[i+1])])
                                i+=2
                            else:
                                new_ptb.append(ptb[i])
                                i+=1
                        ptb = new_ptb
                # pretoken turned into ints
                for ptb_val in ptb:
                    rval.append(ptb_val)
            
        return rval
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids: list[int]):
        pass
    
if __name__ == "__main__":
    vocab = {}
    special_characters = ["<|endoftext|>"]
    for i in range(256):
        vocab[i] = bytes([i])
    i = 256
    for character in special_characters:
        vocab[i] = character
    i += 1 
    vocab[i] = b"cd"
    i += 1
    vocab[i] = b"ba"
    tok = Tokenizer(vocab, [(b"c", b"d"), (b"b", b"a")], special_characters)
    print(tok.encode("ba ba ba<|endoftext|>"))