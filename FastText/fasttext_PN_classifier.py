import fasttext
model = fasttext.train_supervised(input="train.txt", autotuneValidationFile="test.txt", autotuneDuration=3600)
model.test("test.txt")

model.save_model("sentiment_model.bin")