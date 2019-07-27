import torch
import time
import random
from gensim.models import Word2Vec
from model import Model
from utils import *
from metric import *

LR = 0.005
EPOCH = 15
BATCH_SIZE = 32
EMB_SIZE = 200
HID_SIZE = 256
EMB_D = 0.1
NUM_LAYERS = 4
LSTM_D = 0.1


def train_model(model, optimizer, train_sentenses, train_tags, vali_sentenses, vali_tags, word2id, tag2id, id2tag):
    start_time = time.time()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6, 9, 12], gamma=0.1, last_epoch=-1)
    for e in range(1, EPOCH + 1):
        model.train()
        scheduler.step()
        print(scheduler.get_lr())
        total_loss = 0.
        step = 0
        for idx in range(0, len(train_sentenses), BATCH_SIZE):
            batch_sentenses = train_sentenses[idx:idx + BATCH_SIZE]
            batch_tags = train_tags[idx:idx + BATCH_SIZE]

            batch_sentenses, batch_tags, _ = sort_by_length(batch_sentenses, batch_tags)

            batch_sentenses, batch_tags, lengths, mask = to_tensor(batch_sentenses, batch_tags, word2id, tag2id)

            batch_sentenses = batch_sentenses.cuda()
            batch_tags = batch_tags.cuda()
            mask = mask.cuda()

            optimizer.zero_grad()
            loss, emissions = model(batch_sentenses, batch_tags, lengths, mask)
            loss.backward()
            optimizer.step()

            total_loss += + loss.item()
            step += 1
            if step % 20 == 0:
                print("Epoch {}, Loss: {:.5f}".format(e, total_loss / step))

        validate_model(model, vali_sentenses, vali_tags, word2id, tag2id, id2tag, e)
        path = "model_" + str(e) + "_" + str(EMB_SIZE) + "_" + str(HID_SIZE) + ".pt"
        torch.save(model.state_dict(), path)

    end_time = time.time()
    print("Training cost {} seconds.".format(int(end_time - start_time)))


def validate_model(model, vali_sentenses, vali_tags, word2id, tag2id, id2tag, epoch_num):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        step = 0
        pred_tags = []
        for idx in range(0, len(vali_sentenses), BATCH_SIZE):
            batch_sentenses = vali_sentenses[idx:idx + BATCH_SIZE]
            batch_tags = vali_tags[idx:idx + BATCH_SIZE]

            batch_sentenses, batch_tags, indices = sort_by_length(batch_sentenses, batch_tags)

            batch_sentenses, batch_tags, lengths, mask = to_tensor(batch_sentenses, batch_tags, word2id, tag2id)
            batch_sentenses = batch_sentenses.cuda()
            batch_tags = batch_tags.cuda()
            mask = mask.cuda()

            loss, emissions = model(batch_sentenses, batch_tags, lengths, mask)
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
    word2vec, tag2id, id2tag, word2id = load_dict_data(EMB_SIZE)
    weights = load_pretrained_wordvec(word2id, word2vec, EMB_SIZE)

    model = Model(embedding_weight=weights,
                  embedding_dim=EMB_SIZE,
                  dropout_embed=EMB_D,
                  hidden_size=HID_SIZE,
                  num_layers=NUM_LAYERS,
                  num_classes=len(tag2id),
                  dropout_lstm=LSTM_D,
                  tag2idx=tag2id).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    vali_sentenses, vali_tags, train_sentenses, train_tags = load_train_data()

    train_model(model, optimizer, train_sentenses, train_tags, vali_sentenses, vali_tags, word2id, tag2id, id2tag)


def submit():
    word2vec, tag2id, id2tag, word2id = load_dict_data(EMB_SIZE)
    weights = load_pretrained_wordvec(word2id, word2vec, EMB_SIZE)
    path = "model_" + str(9) + "_" + str(EMB_SIZE) + "_" + str(HID_SIZE) + ".pt"
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
    # print(model.crf.transitions)
    test_sentences = load_test_data()
    pred_tags = []
    with torch.no_grad():
        for idx in range(0, len(test_sentences), BATCH_SIZE):
            batch_sentenses = test_sentences[idx:idx + BATCH_SIZE]

            indices = sorted(range(len(batch_sentenses)),
                             key=lambda i: len(batch_sentenses[i]),
                             reverse=True)
            batch_sentenses = [batch_sentenses[idx] for idx in indices]
            lengths = [len(sentense) for sentense in batch_sentenses]

            tensor_batch_sentenses = torch.zeros(len(batch_sentenses), len(batch_sentenses[0]), dtype=torch.long)
            for i, sentense in enumerate(batch_sentenses):
                for j, word in enumerate(sentense):
                    tensor_batch_sentenses[i][j] = word2id.get(word, word2id['unk'])

            batch_sentenses = tensor_batch_sentenses.cuda()
            mask = (batch_sentenses != 0).byte()
            mask = mask.cuda()

            pred_batch_tags = model.decode(batch_sentenses, lengths, mask)

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
    main()
    # submit()
