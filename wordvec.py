from gensim.models import Word2Vec


def load_corpus():
    with open("./datagrand/corpus.txt", "r") as corpus:
        corpus_sentenses = []
        for line in corpus:
            line = line.split(sep="_")
            corpus_sentenses.append(line)
        return corpus_sentenses


def word2vec(sentenses, dim):
    model = Word2Vec(sentenses, size=dim, window=5, min_count=2, workers=8, negative=10, iter=100)
    model.save("word2vec_" + str(dim) + ".model")
    print("Vocab size: ", len(model.wv.vocab))


def main():
    corpus_sentenses = load_corpus()
    word2vec(corpus_sentenses, 300)


if __name__ == "__main__":
    main()
