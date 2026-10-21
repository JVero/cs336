from scripts.preprocessing.predict_language import predict
from scripts.preprocessing.extract_text import extract_text_from_html
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
if __name__ == "__main__":
    in_file_path = Path("./local-shared-data/CC/example.warc")
    print("Time to play, guess the language!")
    my_guesses = []
    labels = []
    num_incorrect = 0
    with open(in_file_path, 'rb') as f:
        ai = ArchiveIterator(f, record_types=WarcRecordType.response)
        record = next(ai) # offsets consistently by 1
        offsets = gen_offsets()
        for offset in offsets:
            for _ in range(offset):
                record = next(ai)
            body_content = record.reader.read()
            text = extract_text_from_html(body_content, strict=False)
            while len(text) == 0:
                print("oops!")
                record = next(ai)
                body_content = record.reader.read()
                text = extract_text_from_html(body_content, strict=False)               
            model_guess, confidence = predict(text, confidence_threshold=0.9)
            if len(text) > 10000:
                text = text[:10000]
            my_guesses.append(input(f"What do you think this language is?\n{text}\n{model_guess},{confidence}\t"))
            if model_guess != my_guesses[-1]:
                print(f"One of you is wrong\n")
                num_incorrect += 1
            print("===================================")
    print(num_incorrect)