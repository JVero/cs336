import fasttext
import numpy as np

nsfw_detector = fasttext.load_model("./local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
hate_detector = fasttext.load_model("./local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

def classify_nsfw(txt: str) -> tuple[bool, float]:
    txt = txt.replace("\n", " ")
    label, confidence = nsfw_detector.predict(txt)
    label = label[0][9:]
    return label, np.clip(0, 1, confidence)[0]

def classify_hate(txt: str) -> tuple[bool, float]:
    txt = txt.replace("\n", " ")
    label, confidence = hate_detector.predict(txt)
    label = label[0][9:]
    return label, np.clip(0, 1, confidence)[0]

if __name__ == "__main__":
    print(classify_nsfw(inp:=input("is it nsfw?")))
    print(classify_hate(inp))
    