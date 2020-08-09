import fasttext
model = fasttext.load_model("sentiment_model.bin")

def PN_classifier(sentence):
    ret = model.predict(sentence)
    if ret[0][0][9] == '1':
        return True
    else:
        return False

if __name__ == "__main__":
    print(PN_classifier("정말 행복해요~"))