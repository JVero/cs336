from scripts.preprocessing.predict_language import predict
from scripts.preprocessing.extract_text import extract_text_from_html
from scripts.preprocessing.mask_pii import all_masks

from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType


from pathlib import Path
import numpy as np

def gen_offsets():
    np.random.seed(1) # different offsets
    num_docs = 20153 # found out ahead of time
    num_samples = 20
    offsets = np.diff(np.sort(np.random.choice(num_docs, size=num_samples+1, replace=False)))
    return 1 + offsets


if __name__ == "__main__":
    in_file_path = Path("./local-shared-data/CC/example.warc")
    print("Time to play, guess the language!")
    y_n = []
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
                record = next(ai)
                body_content = record.reader.read()
                text = extract_text_from_html(body_content, strict=False)               
            text = all_masks(text)
            
            if len(text) > 10000:
                text = text[:10000]
            y_n.append(input(f"Anything wrong here (y/n)?\n{text}"))
            print("===================================")
    print(y_n)