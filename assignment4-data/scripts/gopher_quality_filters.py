import nltk
import numpy as np
import re

LETTER_REGEX = re.compile(r'[a-zA-Z]')

# Filter 1 • Contain less than 50 or more than 100,000 words.
def filter1(*, words: list[str]):
    return len(words) > 50 and len(words) < 100000

# Filter 2 • Have a mean word length outside the range of 3 to 10 characters.
def filter2(*, words: list[str]) -> bool:
    # get the average length of word
    word_len = float(np.mean([len(w) for w in words]))
    return word_len >= 3 and word_len <= 10
    
# Filter 3 • Have *more* than 30% of lines ending with an ellipsis (“...”).
def filter3(*, lines: list[str]) -> bool:
    ends_with_ell = []
    for line in lines:
        last3 = line[-3:]
        if last3 == "...":
            ends_with_ell.append(1)
        else:
            ends_with_ell.append(0)
    return (np.sum(ends_with_ell) / len(ends_with_ell)) < 0.3
    
# Filter 4 • Contain less than 80% of words with at least one alphabetic character.
def filter4(*, words: list[str]) -> bool:
    has_letter: list[bool] = [bool(LETTER_REGEX.search(word)) for word in words]
    return (sum(has_letter) / len(has_letter)) >= 0.8
    

def is_high_quality(text, verbose=False) -> bool:
    words = nltk.word_tokenize(text)
    lines = text.split("\n")
    f1 = filter1(words=words)
    f2 = filter2(words=words)
    f3 = filter3(lines=lines)
    f4 = filter4(words=words)
    if verbose:
        print(f"{f1=}, {f2=}, {f3=}, {f4=}")
    return f1 and f2 and f3 and f4
