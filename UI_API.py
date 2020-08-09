from word2vec.UI_word2vec2 import most_similar
from newscollactor import newscollactor
from FastText.fasttext_UI import PN_classifier

def similar_words(query):
    return most_similar(query)

def return_PN(words):
    PNdict = dict()
    for word in words:
        sourceDF = newscollactor(word)
        sourceDF["pn"] = PN_classifier(sourceDF["crawled"])
        counted_PN = sourceDF["pn"].value_counts().sort_index()
        total = counted_PN.loc('0') + counted_PN.loc('1')
        positive_percentage = counted_PN.loc('1')/total*100
        negative_percentage = counted_PN.loc('0')/total*100
        PNdict[word] = (positive_percentage, negative_percentage)
    return PNdict
if __name__ == "__main__":
    return_PN(similar_words("CJ"))