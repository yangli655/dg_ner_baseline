import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_weight, dropout):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        return x
