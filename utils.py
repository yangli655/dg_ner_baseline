import torch
import random
import pickle
import numpy as np
from gensim.models import Word2Vec


def load_dict_data(emb_size):
    word2vec_file = "word2vec_" + str(emb_size) + ".model"
    word2vec = Word2Vec.load(word2vec_file)
    with open("tag2id", "rb") as f:
        tag2id = pickle.load(f)
    with open("id2tag", "rb") as f:
        id2tag = pickle.load(f)
    with open("word2id", "rb") as f:
        word2id = pickle.load(f)
    return word2vec, tag2id, id2tag, word2id


def load_train_data():
    with open("normal_train.txt", "r") as train_file:
        train_sentenses = []
        train_tags = []
        for line in train_file:
            line = line.strip().split("|||")
            sent, tags = line[0], line[1]
            train_sentenses.append(sent.split(" "))
            train_tags.append(tags.split(" "))
        train_data = list(zip(train_sentenses, train_tags))
        random.shuffle(train_data)
        train_sentenses, train_tags = zip(*train_data)
        vali_sentenses = train_sentenses[:3000]
        vali_tags = train_tags[:3000]
        train_sentenses = train_sentenses[3000:]
        train_tags = train_tags[3000:]
        return vali_sentenses, vali_tags, train_sentenses, train_tags


def load_test_data():
    with open("./datagrand/test.txt", "r") as test_file:
        test_sentenses = []
        for line in test_file:
            line = line.strip().split(sep="_")
            test_sentenses.append(line)
        return test_sentenses


def load_pretrained_wordvec(word2id, word2vec, emb_size):
    weights = np.zeros([len(word2id), emb_size], dtype=np.float32)
    for word in word2id.keys():
        if word in word2vec.wv.vocab:
            weights[word2id[word]] = word2vec[word]
        else:
            weights[word2id[word]] = np.random.uniform(-0.1, 0.1)
    weights = torch.from_numpy(weights)
    return weights


def sort_by_length(sentense_list, tag_list):
    pairs = list(zip(sentense_list, tag_list))
    indices = sorted(range(len(pairs)),
                     key=lambda i: len(pairs[i][0]),
                     reverse=True)
    pairs = [pairs[idx] for idx in indices]
    sentense_list, tag_list = zip(*pairs)
    return sentense_list, tag_list, indices


def recovery_idx(indices):
    idx_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*idx_maps))
    return indices


def to_tensor(sentense_list, tag_list, word2id, tag2id):
    max_length = len(sentense_list[0])
    tensor_sentense_list = torch.zeros(len(sentense_list), max_length, dtype=torch.long)
    tensor_tag_list = torch.zeros(len(tag_list), max_length, dtype=torch.long)
    for i, sentense in enumerate(sentense_list):
        for j, word in enumerate(sentense):
            tensor_sentense_list[i][j] = word2id.get(word, word2id['unk'])
    for i, tags in enumerate(tag_list):
        for j, tag in enumerate(tags):
            tensor_tag_list[i][j] = tag2id[tag]
    list_length = [len(sentense) for sentense in sentense_list]
    tensor_mask = (tensor_sentense_list != 0).byte()
    return tensor_sentense_list, tensor_tag_list, list_length, tensor_mask


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
        for sentense, tags in test_data:
            i, length = 0, len(tags)
            samples = []
            while i < length:
                sample = []
                if tags[i] == 'o':
                    sample.append(sentense[i])
                    j = i + 1
                    while j < length and tags[j] == "o":
                        sample.append(sentense[j])
                        j += 1
                    samples.append("_".join(sample) + '/o')
                elif tags[i][2] == 'S':
                    samples.append("_".join(sentense[i]) + '/' + tags[i][0])
                    j = i + 1
                else:
                    sample.append(sentense[i])
                    j = i + 1
                    while j < length and len(tags[j]) == 3 and tags[j][0] == tags[i][0] and (
                            tags[j][2] == "M" or tags[j][2] == "E"):
                        sample.append(sentense[j])
                        j += 1
                    samples.append('_'.join(sample) + '/' + tags[i][0])
                i = j
            submit_file.write("  ".join(samples) + "\n")
