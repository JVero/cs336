from scripts.preprocessing.predict_language import predict
from scripts.preprocessing.extract_text import extract_text_from_html
from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

def make_retention_curve(label, value): # saves a figure
    vals = np.zeros(101)
    for conf in value:
        vals[np.clip(int(100*conf), 0, 100)] += 1
    plt.plot(1 - (np.cumsum(vals/np.sum(vals))), label=label)

if __name__ == "__main__":
    in_file_path = Path("./local-shared-data/CC/example.warc")
    with open(in_file_path, 'rb') as f, open("./scripts/sample_english_garbage.txt", "w") as of:
        total_docs = 0
        en_docs = 0
        empty_docs = 0
        conf = {}
        conf = defaultdict(list)
        for record in ArchiveIterator(f):
            if record.record_type != WarcRecordType.response:
                continue
            total_docs += 1
            body_content = record.reader.read()
            text = extract_text_from_html(body_content, strict=False)
            if len(text) == 0:
                empty_docs += 1
                continue
            label, confidence = predict(text, confidence_threshold=0.9)
            if label == "en" and confidence < 0.7 and confidence > 0.4:
                of.write(text)
                of.write(str(confidence) + "^===============")
            # print(len(text), text[:100])
            print(label, confidence)
            print("===================================")
            if label == "en":
                en_docs += 1
            
            conf[label].append(confidence)
            # if total_docs >= 25: break
    print(f"{total_docs=}, {en_docs=}, {empty_docs=}")
    for k, v in conf.items():
        if len(v) > 1000:
            make_retention_curve(k, v)
    plt.legend()
    plt.savefig("./figures/retention_curve.png")
