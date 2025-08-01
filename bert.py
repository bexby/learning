import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MyTransformers import Encoder
from dataclasses import dataclass


@dataclass
class BertConfig:
    hidden_size: int = 768
    vocab_size: int = 30522
    num_hidden_layer: int = 12,
    num_head: int = 12
    max_len: int = 512


class Pooler(nn.Module):
    def __init__(self):
        super().__init__()


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_len, hidden_size)     # Bert uses learnable position embedding
        self.layernorm = nn.LayerNorm((hidden_size))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, indices):
        t_emb = self.token_embedding(indices)
        N, seq_len = indices.shape
        pos_indices = torch.stack([torch.arange(seq_len)] * N, dim=0)
        p_emb = self.pos_embedding(pos_indices)
        embedding = t_emb + p_emb
        embedding = self.layernorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, max_norm=1)
        self.layers = nn.Sequential([Encoder()] * 12)
        self.pooler = Pooler()
    
    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        pass
        
emb = BertEmbedding(20, 5, 60)
indices = torch.randint(0, 20, (2, 50))
print(emb(indices).shape)
