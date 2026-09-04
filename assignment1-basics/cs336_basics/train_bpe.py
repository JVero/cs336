import regex as re

from collections import Counter, defaultdict
from itertools import pairwise
from functools import reduce


from pathlib import Path

from .pretokenization_example import find_chunk_boundaries

from multiprocessing import Pool

from tqdm import trange, tqdm

# To profile this
# sudo uv run py-spy record --subprocesses -o profile_graph.svg \
# -- python -m cs336_basics.train_bpe

# To time this
# time uv run python -m cs336_basics.train_bpe

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
        print("\r", start, end="", flush=True)
        return occurrences

def train_bpe(input_path: str | Path,
              vocab_size: int,
              special_tokens: list[str], *, num_chunks=120, num_workers=12) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
    ### Build the initial ASCII vocabulary, and the empty merge list
    ### vocab, merges are the return values    
    # vocab: dict[int, bytes] = {}
    id_to_bytes: dict[int, bytes] = {}
    for i in range(256): 
        id_to_bytes[i] = bytes([i])
    merges = []    
        
    ### Encode the special tokens, then add them to the vocabulary
    idx = 256
    for token in special_tokens:
        id_to_bytes[idx] = token.encode("utf-8")
        idx += 1
    with open(input_path, 'rb') as f:    
        offsets = find_chunk_boundaries(f, num_chunks, bytes("<|endoftext|>", encoding="utf-8"))
    with Pool(num_workers) as p:
        results = p.starmap(build_occurrences, [(input_path, start, stop, special_tokens) for start,stop in pairwise(offsets)])
    # occurrences = reduce(lambda x,y: x.update(y), results)
    occurrences = Counter()
    for result in results:
        occurrences.update(result)
    
    ### Step 1 - Find the most occurring pairs of token IDs
    pair_count = Counter()
    
    # lists all the words that have a pair of bytes
    pair_words = defaultdict(set)
    
    for word, count in occurrences.items():
        for pair in pairwise(word):
            pair_count[pair] += count
            pair_words[pair].add(word)
    ### Build the vocabulary to vocab_size
    start = idx
    for idx in trange(start, vocab_size):
                
        # Find the token that occurs the most (pair_count[k]), with ties broken lexicographically (vocab[k[0]], then vocab[k[1]]
        max_occ = max(pair_count.values())
        valid_pairs = []
        for k, v in pair_count.items():
            if v == max_occ:
                valid_pairs.append((k[0],k[1]))
        new_token_pair = max(valid_pairs, key=lambda k: (id_to_bytes[k[0]], id_to_bytes[k[1]]))
        ### Step 2 - Add the newest token pair to merges for reconstruction, as well as the vocabulary
        merges.append((id_to_bytes[new_token_pair[0]], id_to_bytes[new_token_pair[1]]))
        id_to_bytes[idx] = id_to_bytes[new_token_pair[0]] + id_to_bytes[new_token_pair[1]]

        words_to_change = []
        for word in pair_words[new_token_pair]:
            i = 0
            new_word = []
            while i < len(word):
                if i != len(word) - 1 and (word[i], word[i+1]) == new_token_pair:
                    new_word.append(idx)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words_to_change.append((tuple(new_word), word))

        for new, old in words_to_change:
            old_occ = occurrences[old]
            for pair in pairwise(old):
                pair_count[pair] -= old_occ
                if pair_count[pair] == 0:
                    pair_count.pop(pair, None)
                pair_words[pair].discard(old)
            for pair in pairwise(new):
                pair_count[pair] += old_occ
                pair_words[pair].add(new)
            occurrences[new] = old_occ
            occurrences.pop(old, None)
    return id_to_bytes, merges
    
def train_dataset(data_filename, *,data_dir: Path =  Path(__file__).parent.parent / "data", vocab_size=10_000, special_tokens=["<|endoftext|>"], num_chunks=120, num_workers=12):
    from glob import glob
    if not data_dir.is_dir():
        raise ValueError(f"{str(data_dir)} is not a directory.")

    input_path: Path = data_dir / data_filename
    if not input_path.is_file():
        print([data_dir.glob("*")])
        raise ValueError(f"{input_path} is not in the directory")
    
    vocab, mergelist = train_bpe(input_path, vocab_size, special_tokens=special_tokens, num_chunks=num_chunks, num_workers=num_workers)
    readable_vocab = {k : v.hex() for k,v in vocab.items()}
    readable_merges = [(l.hex(), r.hex()) for l, r in mergelist]
    import json
    with open(input_path.stem + "_vocab.json", "w+") as f:
        json.dump(readable_vocab, f)
    with open(input_path.stem + "_merges.json", "w+") as f:
        json.dump(readable_merges, f)
    
def train_bpe_tinystories(*, num_chunks=120, num_workers=12):
    return train_dataset( "TinyStoriesV2-GPT4-valid.txt")

def train_bpe_expts_owt(*, num_chunks=120, num_workers=12):
    return train_dataset("owt_train.txt", vocab_size=32_000)
    
def train_bpe_valid_owt(*, num_chunks=120, num_workers=12):
    return train_dataset("owt_valid.txt", vocab_size=100)

if __name__ == "__main__":
    # train_bpe_valid_owt()
    train_bpe_expts_owt(num_chunks=500)
    print("done")