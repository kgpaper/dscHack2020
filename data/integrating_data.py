pwd = "/home/sentiment/바탕화면/ML_practice/txtfiles/morphed/"
fromfilelist = [
    #"morphed-wiki-alpha.txt",
    "morphed-wiki-kowiki.txt",
    "morphed-wiki-kowikisource.txt",
    "morphed-wiki-kowiktionary.txt",
    #"morphed-wiki-namu.txt",
    "morphed-wiki-oriwiki.txt",
    "morphed-wiki-osa.txt"
]

all_data = [open(pwd+name, 'r').read().strip() for name in fromfilelist]
with open(pwd+"integrated_wiki_corpus.txt", 'w') as f:
    f.write("\n".join(all_data))
