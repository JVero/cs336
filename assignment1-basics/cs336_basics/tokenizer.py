from typing import Iterable, Iterator

import json

class Tokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        a file is a big json file dictionary
        """
        vocab = json.load(vocab_filepath)
        merges = json.load(merges_filepath)
        return cls(vocab, merges, special_tokens=special_tokens)
        
    def encode(self, text: str) -> list[int]:
        pass
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids: list[int]):
        pass