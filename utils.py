import torch
import random
import pickle
import numpy as np
from gensim.models import Word2Vec
from glove import Glove


def load_dict_data():
    with open("tag2id", "rb") as f:
        tag2id = pickle.load(f)
    with open("id2tag", "rb") as f:
        id2tag = pickle.load(f)
    with open("word2id", "rb") as f:
        word2id = pickle.load(f)
    return tag2id, id2tag, word2id


def load_train_data():
    with open("./datagrand/normal_train.txt", "r") as train_file:
        train_sentences = []
        train_tags = []
        for line in train_file:
            line = line.strip().split("|||")
            sent, tags = line[0], line[1]
            train_sentences.append(sent.split(" "))
            train_tags.append(tags.split(" "))
        return train_sentences, train_tags


def load_test_data():
    with open("./datagrand/test.txt", "r") as test_file:
        test_sentences = []
        for line in test_file:
            line = line.strip().split(sep="_")
            test_sentences.append(line)
        return test_sentences


def load_pretrained_wordvec(word2id, emb_size):
    word2vec_file = "./wordvec/word2vec_" + str(emb_size) + ".model"
    word2vec = Word2Vec.load(word2vec_file)

    # glove_corpus_file = "./wordvec/corpus_" + str(emb_size) + ".model"
    glove_file = "./wordvec/glove_" + str(emb_size) + ".model"
    glove = Glove.load(glove_file)

    weights = np.zeros([len(word2id), emb_size], dtype=np.float32)
    for word in word2id.keys():
        if word in word2vec.wv.vocab and word in glove.dictionary:
            # weights[word2id[word]] = glove.word_vectors[glove.dictionary[word]]
            weights[word2id[word]] = word2vec[word]
            # weights[word2id[word]] = np.concatenate((word2vec[word], glove.word_vectors[glove.dictionary[word]]),
            #                                         axis=0)
        else:
            weights[word2id[word]] = np.random.uniform(-0.25, 0.25)
    weights = torch.from_numpy(weights)
    return weights


def data_iter(sentences, tags=None, batch_size=32, test=None):
    for idx in range(0, len(sentences), batch_size):
        batch_sentences = sentences[idx:idx + batch_size]
        if not test:
            batch_tags = tags[idx:idx + batch_size]
            yield batch_sentences, batch_tags
        else:
            yield batch_sentences


def sort_by_length(sentence_list, tag_list):
    pairs = list(zip(sentence_list, tag_list))
    indices = sorted(range(len(pairs)),
                     key=lambda i: len(pairs[i][0]),
                     reverse=True)
    pairs = [pairs[idx] for idx in indices]
    sentence_list, tag_list = zip(*pairs)
    return sentence_list, tag_list, indices


def recovery_idx(indices):
    idx_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*idx_maps))
    return indices


def to_tensor(sentence_list, tag_list, word2id, tag2id):
    max_length = len(sentence_list[0])
    tensor_sentence_list = torch.zeros(len(sentence_list), max_length, dtype=torch.long)
    tensor_tag_list = torch.zeros(len(tag_list), max_length, dtype=torch.long)
    for i, sentence in enumerate(sentence_list):
        for j, word in enumerate(sentence):
            tensor_sentence_list[i][j] = word2id.get(word, word2id['unk'])
    for i, tags in enumerate(tag_list):
        for j, tag in enumerate(tags):
            tensor_tag_list[i][j] = tag2id[tag]
    list_length = [len(sentence) for sentence in sentence_list]
    tensor_mask = (tensor_sentence_list != 0).byte()
    return tensor_sentence_list, tensor_tag_list, list_length, tensor_mask


def to_tags(pred_tags, id2tag):
    str_tags = []
    for tags in pred_tags:
        temp = []
        for tag in tags:
            temp.append(id2tag[tag])
        str_tags.append(temp)
    return str_tags


def format_submit(test_sentences, pred_tags):
    test_data = zip(test_sentences, pred_tags)
    with open("result.txt", "w", encoding="UTF-8") as submit_file:
        for sentence, tags in test_data:
            i, length = 0, len(tags)
            samples = []
            while i < length:
                sample = []
                if tags[i] == 'o':
                    sample.append(sentence[i])
                    j = i + 1
                    while j < length and tags[j] == "o":
                        sample.append(sentence[j])
                        j += 1
                    samples.append("_".join(sample) + '/o')
                elif tags[i][2] == 'S':
                    samples.append(sentence[i] + '/' + tags[i][0])
                    j = i + 1
                else:
                    sample.append(sentence[i])
                    j = i + 1
                    while j < length and len(tags[j]) == 3 and tags[j][0] == tags[i][0] and (
                            tags[j][2] == "M" or tags[j][2] == "E"):
                        sample.append(sentence[j])
                        j += 1
                    samples.append('_'.join(sample) + '/' + tags[i][0])
                i = j
            submit_file.write("  ".join(samples) + "\n")
