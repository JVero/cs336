from scripts.predict_language import predict
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

if __name__ == "__main__":
    in_file_path = Path("./local-shared-data/CC/example.warc")
    print("Time to play, guess the language!")
    my_guesses = []
    labels = []
    num_incorrect = 0
    with open(in_file_path, 'rb') as f:
        ai = ArchiveIterator(f, record_types=WarcRecordType.response)
        offsets = gen_offsets()
        for offset in offsets:
            for _ in range(offset):
                record = next(ai)
                body_content = record.reader.read()
                text = extract_text_from_html(body_content, strict=False)
                if len(text) == 0:
                    print("oops!")
                    continue
                my_guesses.append(input(f"What do you think this language is?\n{text[:100]}"))
                model_guess, confidence = predict(text, confidence_threshold=0.9)
                if model_guess != my_guesses:
                    input(f"One of you is wrong!{model_guess}, {confidence}")
                    num_incorrect += 1
                print("===================================")