import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MyTransformers import Encoder
from dataclasses import dataclass
import pdb

@dataclass
class BertConfig:
    hidden_size: int = 768
    vocab_size: int = 30522
    num_hidden_layer: int = 12
    num_head: int = 12
    max_len: int = 512


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activate = nn.Tanh()
    
    def forward(self, state):
        cls_token = state[:, 0, :]     # so in the padding stage the padding side must be the right
        output = self.linear(cls_token)
        return self.activate(output)


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
        self.embedding = BertEmbedding(config.vocab_size, config.hidden_size, config.max_len)
        self.layers = nn.Sequential(* [Encoder(config.hidden_size, config.num_head, True)] * config.num_hidden_layer)
        self.pooler = BertPooler(config.hidden_size)
    
    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        embeddings = self.embedding(input_ids)
        pdb.set_trace()
        last_hidden_state = self.layers(embeddings, attn_mask)
        cls_tensor = self.pooler(last_hidden_state)
        return cls_tensor
        
# emb = BertEmbedding(20, 5, 60)
# indices = torch.randint(0, 20, (2, 50))
# print(emb(indices).shape)


bert_config = BertConfig()

bert = Bert(bert_config)
# print(bert)

bs = 2
seq_len = 50
vocab_size = 20

input_ids = torch.randint(0, vocab_size, (bs, seq_len))
attention_mask = []
for _ in range(bs):
    padding_len = torch.randint(0, 10, (1, 1)).item()
    attention_mask.append(torch.cat([torch.zeros(seq_len - padding_len).bool(), \
                                    torch.ones(padding_len).bool()], dim=0))
attention_mask = torch.stack(attention_mask, dim=0)

output = bert(input_ids, attention_mask)
print(output.shape)