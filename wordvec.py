from gensim.models import Word2Vec


def load_corpus():
    with open("./datagrand/corpus.txt", "r") as corpus:
        corpus_sentences = []
        for line in corpus:
            line = line.split(sep="_")
            corpus_sentences.append(line)
        return corpus_sentences


def word2vec(sentences, dim):
    model = Word2Vec(sentences, size=dim, window=5, min_count=2, workers=8, negative=10, iter=100)
    model.save("./wordvec/word2vec_" + str(dim) + ".model")
    print("Vocab size: ", len(model.wv.vocab))


def main():
    corpus_sentences = load_corpus()
    word2vec(corpus_sentences, 400)


if __name__ == "__main__":
    main()
