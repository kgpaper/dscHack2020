from gensim.models.word2vec import Word2Vec
import multiprocessing
coreNum = multiprocessing.cpu_count()

def most_similar(query):
    w2v_model = Word2Vec.load("word2vec.model")
    try:
        words = w2v_model.wv.most_similar(positive=query, topn=20)
        print(words)
        return words
    except:
        print("error")
        return -1


def online_train(filepath):
    w2v_model = Word2Vec.load("CJ제일제당" + "word2vec.model")
    w2v_model.train(filepath, epochs=20, total_examples=w2v_model.corpus_count)
    w2v_model.save("word2vec2.model")

if __name__ == "__main__":
    #most_similar("CJ제일제당")
    online_train("/home/sentiment/바탕화면/ML_practice/txtfiles/morphed/morphed-wiki-alpha.txt")