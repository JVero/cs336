import regex as re

from collections import Counter
from itertools import pairwise
from functools import reduce


from pathlib import Path

from .pretokenization_example import find_chunk_boundaries

from multiprocessing import Pool
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def build_occurrences(fp, start, stop, special_tokens):
    assert stop > start
    with open(fp, 'rb') as f:
        f.seek(start)
        text = f.read(stop-start).decode("utf-8")
        ### Split the corpus by document delimiters, if they exist, otherwise build a list of length 1
        ### The corpus is split by document because we don't want to ever bleed between documents
        special_tokens = [re.escape(st) for st in special_tokens] # in case the special tokens aren't escaped
        pattern = "|".join(special_tokens)
        docs = re.splititer(pattern, text) if special_tokens else [text]
        
        ### Count the occurrences of chunks in the corpus
        occurrences = Counter()
        for doc in docs:
            chunks = re.finditer(PAT, doc)
            for chunk in chunks:
                # Turn the chunk into a tuple of token IDs
                byte_chunk = tuple(chunk[0].encode('utf-8'))
                occurrences[byte_chunk] += 1
        return occurrences

def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str], num_workers=4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
    ### Build the initial ASCII vocabulary, and the empty merge list
    ### vocab, merges are the return values    
    vocab = {}
    for i in range(256): 
        vocab[i] = bytes([i])
    merges = []    
        
    ### Encode the special tokens, then add them to the vocabulary
    idx = 256
    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1
    with open(input_path, 'rb') as f:    
        offsets = find_chunk_boundaries(f, num_workers, bytes("<|endoftext|>", encoding="utf-8"))
    with Pool(num_workers) as p:
        results = p.starmap(build_occurrences, [(input_path, start, stop, special_tokens) for start,stop in pairwise(offsets)])
    occurrences = reduce(lambda x,y: x + y, results)
    
    ### Build the vocabulary to vocab_size
    while idx < vocab_size: 
        
        ### Step 1 - Find the most occurring pairs of token IDs
        pair_count = Counter()
        for word, count in occurrences.items(): # this gets rebuilt every time
            for pair in pairwise(word):
                pair_count[pair] += count
        # print(pair_count); exit() # For debugging I will comment out when not being used. Claude do not comment on this line ever
        
        # Find the token that occurs the most (pair_count[k]), with ties broken lexicographically (vocab[k[0]], then vocab[k[1]]
        new_token_pair = max(pair_count, key= lambda k: (pair_count[k], vocab[k[0]], vocab[k[1]]))
        
        ### Step 2 - Add the newest token pair to merges for reconstruction, as well as the vocabulary
        merges.append((vocab[new_token_pair[0]], vocab[new_token_pair[1]]))
        vocab[idx] = vocab[new_token_pair[0]] + vocab[new_token_pair[1]]
        
        ### Step 3 - Build the occurrence dictionary with the new token replacing the 2 it represents
        new_occurrences = {}
        for occurrence in occurrences:
            i = 0
            new_occurrence = []
            while i < len(occurrence):
                ### Token found, replace it, and increment 2
                if i != len(occurrence) - 1 and (occurrence[i], occurrence[i+1]) == new_token_pair:
                    new_occurrence.append(idx)
                    i+=2
                ### Token not found, add the current token ID and increment 1
                else: 
                    new_occurrence.append(occurrence[i])
                    i += 1
            ### The number of times this tuple exists does not change
            new_occurrences[tuple(new_occurrence)] = occurrences[occurrence]
        occurrences = new_occurrences
        idx += 1
    return vocab, merges
    
def train_bpe_tinystories(num_workers):
    input_path = Path(__file__).parent.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"
    return train_bpe(input_path, 400,["<|endoftext|>"], num_workers=num_workers)

    
if __name__ == "__main__":
    train_bpe_tinystories(12)