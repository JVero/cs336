import fasttext

detector = fasttext.load_model("./local-shared-data/classifiers/lid.176.bin")

def predict(text: str, confidence_threshold = 0.9, disable_warning=False):
    if not isinstance(text, str):
        raise TypeError("This function must recieve a string. Recieved ", type(text))
    text = text.replace("\n", " ")
    rval: tuple = detector.predict(text)
    prediction: str = rval[0][0][9:]
    confidence: float = rval[1][0]
    if confidence < confidence_threshold and not disable_warning:
        print(f"WARNING! Confidence is not above your chosen threshold: {confidence=} {confidence_threshold=}")
    return (prediction, confidence)

if __name__ == "__main__":
    print(predict("hey can you please translate this? Le petit chat noir et blanc dort paisiblement sur le vieux tapis rouge du salon."))