import torch
import time
import random
from torch.nn.utils import clip_grad_norm_
from gensim.models import Word2Vec
from model import Model
from utils import *
from metric import *

LR = 0.001
EPOCH = 10
BATCH_SIZE = 32
EMB_SIZE = 300
HID_SIZE = 384
EMB_D = 0.25
NUM_LAYERS = 2
LSTM_D = 0.25


def train_model(model, optimizer, train_sentences, train_tags, word2id, tag2id, id2tag):
    start_time = time.time()

    vali_sentences = train_sentences[:2000]
    vali_tags = train_tags[:2000]
    train_sentences = train_sentences[2000:]
    train_tags = train_tags[2000:]

    train_data = list(zip(train_sentences, train_tags))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=0.1, last_epoch=-1)
    for e in range(1, EPOCH + 1):
        random.shuffle(train_data)
        train_sentences_shuffle, train_tags_shuffle = zip(*train_data)
        model.train()
        scheduler.step()
        print("Epoch", e, ",lr: ", scheduler.get_lr()[0])
        total_loss = 0.
        step = 0

        for batch_sentences, batch_tags in data_iter(train_sentences_shuffle, train_tags_shuffle, BATCH_SIZE):
            batch_sentences, batch_tags, _ = sort_by_length(batch_sentences, batch_tags)

            batch_sentences, batch_tags, lengths, mask = to_tensor(batch_sentences, batch_tags, word2id, tag2id)

            batch_sentences = batch_sentences.cuda()
            batch_tags = batch_tags.cuda()
            mask = mask.cuda()

            optimizer.zero_grad()
            loss, emissions = model(batch_sentences, batch_tags, lengths, mask)
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            total_loss += + loss.item()
            step += 1
            if step % 20 == 0:
                print("Epoch {}, Loss: {:.5f}".format(e, total_loss / step))

        validate_model(model, vali_sentences, vali_tags, word2id, tag2id, id2tag, e)
        path = "./output/model_" + str(e) + "_" + str(EMB_SIZE) + "_" + str(HID_SIZE) + ".pt"
        torch.save(model.state_dict(), path)

    end_time = time.time()
    print("Training cost {} seconds.".format(int(end_time - start_time)))


def validate_model(model, vali_sentences, vali_tags, word2id, tag2id, id2tag, epoch_num):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        step = 0
        pred_tags = []

        for batch_sentences, batch_tags in data_iter(vali_sentences, vali_tags, BATCH_SIZE):
            batch_sentences, batch_tags, indices = sort_by_length(batch_sentences, batch_tags)

            batch_sentences, batch_tags, lengths, mask = to_tensor(batch_sentences, batch_tags, word2id, tag2id)
            batch_sentences = batch_sentences.cuda()
            batch_tags = batch_tags.cuda()
            mask = mask.cuda()

            loss, emissions = model(batch_sentences, batch_tags, lengths, mask)
            pred_batch_tags = model.decode_emissions(emissions, mask)

            indices = recovery_idx(indices)
            pred_batch_tags = [pred_batch_tags[i] for i in indices]
            pred_tags.extend(pred_batch_tags)

            total_loss += + loss.item()
            step += 1
        print("Epoch {}, Validation Loss: {:.5f}".format(epoch_num, total_loss / step))
        pred_tags = to_tags(pred_tags, id2tag)
        get_ner_fmeasure(vali_tags, pred_tags)


def main():
    tag2id, id2tag, word2id = load_dict_data()
    weights = load_pretrained_wordvec(word2id, EMB_SIZE)

    model = Model(embedding_weight=weights,
                  embedding_dim=EMB_SIZE,
                  dropout_embed=EMB_D,
                  hidden_size=HID_SIZE,
                  num_layers=NUM_LAYERS,
                  num_classes=len(tag2id),
                  dropout_lstm=LSTM_D,
                  tag2idx=tag2id).cuda()
    # model.load_state_dict(torch.load("./output/model_" + str(25) + "_" + str(EMB_SIZE) + "_" + str(HID_SIZE) + ".pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_sentences, train_tags = load_train_data()
    train_data = list(zip(train_sentences, train_tags))
    random.shuffle(train_data)
    train_sentences, train_tags = zip(*train_data)

    train_model(model, optimizer, train_sentences, train_tags, word2id, tag2id, id2tag)


def submit():
    tag2id, id2tag, word2id = load_dict_data()
    weights = load_pretrained_wordvec(word2id, EMB_SIZE)
    path = "./output/model_" + str(10) + "_" + str(EMB_SIZE) + "_" + str(HID_SIZE) + ".pt"
    device = torch.device("cuda")
    model = Model(embedding_weight=weights,
                  embedding_dim=EMB_SIZE,
                  dropout_embed=EMB_D,
                  hidden_size=HID_SIZE,
                  num_layers=NUM_LAYERS,
                  num_classes=len(tag2id),
                  dropout_lstm=LSTM_D,
                  tag2idx=tag2id)

    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    test_sentences = load_test_data()
    pred_tags = []
    with torch.no_grad():
        for batch_sentences in data_iter(test_sentences, None, BATCH_SIZE, True):
            indices = sorted(range(len(batch_sentences)),
                             key=lambda i: len(batch_sentences[i]),
                             reverse=True)
            batch_sentences = [batch_sentences[idx] for idx in indices]
            lengths = [len(sentence) for sentence in batch_sentences]

            tensor_batch_sentences = torch.zeros(len(batch_sentences), len(batch_sentences[0]), dtype=torch.long)
            for i, sentence in enumerate(batch_sentences):
                for j, word in enumerate(sentence):
                    tensor_batch_sentences[i][j] = word2id.get(word, word2id['unk'])

            batch_sentences = tensor_batch_sentences.cuda()
            mask = (batch_sentences != 0).byte()
            mask = mask.cuda()

            pred_batch_tags = model.decode(batch_sentences, lengths, mask)

            indices = recovery_idx(indices)
            pred_batch_tags = [pred_batch_tags[i] for i in indices]
            pred_tags.extend(pred_batch_tags)

    pred_tags = to_tags(pred_tags, id2tag)
    # with open("pred_tags.txt", "w") as f:
    #     for tags in pred_tags:
    #         for tag in tags:
    #             f.write(tag + " ")
    #         f.write("\n")
    format_submit(test_sentences, pred_tags)


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # main()
    submit()
