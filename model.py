import torch
import torch.nn as nn
from embedding import EmbeddingLayer
from torchcrf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Model(nn.Module):
    def __init__(self, embedding_weight, embedding_dim, dropout_embed, hidden_size, num_layers, num_classes,
                 dropout_lstm, tag2idx):
        super(Model, self).__init__()

        self.embedding = EmbeddingLayer(embedding_weight, dropout_embed)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_lstm,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

        self.crf = CRF(num_tags=len(tag2idx),
                       batch_first=True)

    def get_emissions(self, inputs, length):
        x = self.embedding(inputs)
        packed = pack_padded_sequence(x, length, batch_first=True)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.linear1(output)
        output = self.relu(output)
        emissions = self.linear2(output)
        return emissions

    def forward(self, inputs, tags, length, mask):
        emissions = self.get_emissions(inputs, length)
        log_likelihood = self.crf(emissions, tags, mask, reduction='mean')
        return -1 * log_likelihood, emissions

    def decode(self, inputs, length, mask):
        emissions = self.get_emissions(inputs, length)
        pred_tags = self.crf.decode(emissions, mask)
        return pred_tags

    def decode_emissions(self, emissions, mask):
        pred_tags = self.crf.decode(emissions, mask)
        return pred_tags
