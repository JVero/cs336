from scripts.predict_language import predict as predict_language
from scripts.harmful_content_detector import classify_nsfw, classify_hate
from scripts.extract_text import extract_text_from_html
from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType

from pathlib import Path
import numpy as np

def gen_offsets():
    np.random.seed(0) # for reproducibility
    num_docs = 20153 # found out ahead of time
    num_samples = 20
    offsets = np.diff(np.sort(np.random.choice(num_docs, size=num_samples+1, replace=False)))
    return 1 + offsets
## zh = chinese
## en = english
## es = spanish
## ja = japanese
## it = italian

def get_valid_record(iter: ArchiveIterator):
    max_tries = 100
    for i in range(max_tries):
        record = next(iter)
        body_content = record.reader.read()
        text = extract_text_from_html(body_content, strict=False)
        if len(text) == 0:
            continue
        model_guess, confidence = predict_language(text)
        if model_guess != "en":
            continue
        return text
    else:
        raise ValueError("None of the next 100 records are in english")

if __name__ == "__main__":
    in_file_path = Path("./local-shared-data/CC/example.warc")
    print("Time to play, guess the language!")
    my_sfw_guesses = []
    labels = []
    num_incorrect = 0
    with open(in_file_path, 'rb') as f:
        ai = ArchiveIterator(f, record_types=WarcRecordType.response)
        record = next(ai) # offsets consistently by 1
        offsets = gen_offsets()
        for offset in offsets:
            for _ in range(offset-1):
                record = next(ai)
            text = get_valid_record(ai)
            if len(text) > 10000:
                text = text[:10000]
            # my_sfw_guesses.append(input("sfw or nsfw?\n" + text))
            model_guess, confidence = classify_nsfw(text)
            # if model_guess != my_sfw_guesses[-1]:
                # print(f"One of you is wrong\n", model_guess, confidence)
                # num_incorrect += 1
            labels.append((model_guess, confidence))
            print("===================================")
    print(labels)
    print(num_incorrect)