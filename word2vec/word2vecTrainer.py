from gensim.models.word2vec import Word2Vec
import multiprocessing
coreNum = multiprocessing.cpu_count()

sentences = [s.strip().split(' ') for s in open("/home/sentiment/바탕화면/ML_practice/txtfiles/morphed/integrated_wiki_corpus.txt").read().strip().split('\n') if s.strip() != '']
print("sentence load finish")

def trainingModel(quary):
    w2v_model = Word2Vec(min_count=3, window=5, size=100, workers=coreNum-3, iter=60, sg=1, compute_loss=True)
    print("model build finish")
    w2v_model.build_vocab(sentences)
    print("vocab finish")
    w2v_model.train(sentences, epochs=w2v_model.iter, total_examples=w2v_model.corpus_count)
    w2v_model.save(quary+"word2vec.model")
    try:
        similarWord = w2v_model.wv.most_similar(positive=quary, topn=20)
    except:
        return -1
    return similarWord

if __name__ == "__main__":
    print(trainingModel("CJ제일제당"))
