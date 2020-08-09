import numpy as np
import os
import pickle


def word2id(path, name):
    if os.path.exists(path+name[:-4]+"word2id.pkl"):
        with open(path+name[:-4]+"word2id.pkl", 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        sentences = [s.strip().split(' ') for s in open(path + name).read().strip().split('\n') if s.strip() != '']
        return sentences, word_to_id, id_to_word

    word_to_id = dict()
    id_to_word = dict()

    sentences = [s.strip().split(' ') for s in open(path+name).read().strip().split('\n') if s.strip() != '']
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_id:
                tmp_id = len(word_to_id)
                word_to_id[word] = tmp_id
                id_to_word[tmp_id] = word

    with open(path+name[:-4]+"word2id.pkl", 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f, protocol=4)

    return sentences, word_to_id, id_to_word

def load_data(path, name):
    sentences, word_to_id, id_to_word = word2id(path, name)

    if os.path.exists(path+name[:-4]+"corpus.npy"):
        corpus = np.load(path+name[:-4]+"corpus.npy")
        return corpus, word_to_id, id_to_word

    corpus = np.array([word_to_id[w] for sentence in sentences for w in sentence])

    np.save(path+name[:-4]+"corpus.npy", corpus)

    return corpus, word_to_id, id_to_word

if __name__ == "__main__":
    #word2id("/home/sentiment/바탕화면/ML_practice/txtfiles/morphed/", "integrated_wiki_corpus.txt")
    load_data("/home/sentiment/바탕화면/ML_practice/txtfiles/morphed/", "integrated_wiki_corpus.txt")