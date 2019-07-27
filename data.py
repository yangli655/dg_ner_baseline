import pickle


def create_tag2id():
    with open("tag2id", "wb") as f:
        tag2id = {'a-B': 0, 'a-M': 1, 'a-E': 2, 'a-S': 3, 'b-B': 4, 'b-M': 5, 'b-E': 6, 'b-S': 7, 'c-B': 8, 'c-M': 9,
                  'c-E': 10, 'c-S': 11, 'o': 12}
        pickle.dump(tag2id, f)
        return tag2id


def create_id2tag(tag2id):
    id2tag = {}
    for tag, idx in tag2id.items():
        id2tag[idx] = tag
    with open("id2tag", "wb")as f:
        pickle.dump(id2tag, f)
    return id2tag


def create_word2id():
    with open("./datagrand/corpus.txt", "r") as corpus:
        corpus_word = {}
        corpus_word['pad'] = 0
        corpus_word['unk'] = 1
        for line in corpus:
            line = line.split(sep="_")
            for word in line:
                if word not in corpus_word:
                    corpus_word[word] = len(corpus_word)

    with open("word2id", "wb") as f:
        pickle.dump(corpus_word, f)


def main():
    tag2id = create_tag2id()
    id2tag = create_id2tag(tag2id)
    create_word2id()


if __name__ == "__main__":
    main()
