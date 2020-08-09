stopword = set(open("/home/sentiment/바탕화면/ML_practice/qurery2trend/data/stop_word_korean.txt").read().split())


def word2noun(sentence):
    from konlpy.tag import Mecab
    tokenizer = Mecab()
    tempX = tokenizer.morphs(sentence)
    ret = [word for word in tempX if not word in stopword]
    return " ".join(ret)

fromfilelist = [
    #"wiki-alpha.txt",
    #"wiki-kowiki.txt",
    #"wiki-kowikisource.txt",
    #"wiki-kowiktionary.txt",
    "wiki-namu.txt",
    #"wiki-oriwiki.txt",
    #"wiki-osa.txt"
]

morphed = "morphed-"

def morpher(name):
    savefilename = morphed + name
    with open("/home/sentiment/바탕화면/ML_practice/txtfiles/"+name, 'r') as f:
        with open("/home/sentiment/바탕화면/ML_practice/txtfiles/"+savefilename, 'w') as fw:
            lines = f.readlines()
            for line in lines:
                try:
                    morphed_line = word2noun(line)
                except:
                    pass
                fw.write(morphed_line)
    print(name, "finish")


from multiprocessing import Pool
myp = Pool(len(fromfilelist))
with myp:
    myp.map(morpher, fromfilelist)