import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MyTransformers import Decoder
from dataclasses import dataclass
from transformers import BertTokenizer
import pdb


@dataclass
class GPT2Config:
    hidden_size: int = 768
    vocab_size: int = 50257
    num_hidden_layer: int = 12
    num_head: int = 12
    max_len: int = 1024

@dataclass
class GPT2Output:
    last_hidden_state: torch.Tensor = None
    loss: torch.Tensor = None


class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, indices):
        we = self.word_embedding(indices)
        bl = indices.shape[-1]
        p_index = torch.arange(bl).unsqueeze(0)
        pe = self.position_embedding(p_index)
        embedding = we + pe
        return self.dropout(embedding)


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = GPT2Embedding(config.vocab_size, config.hidden_size, config.max_len)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([Decoder(config.hidden_size, config.num_head)] * config.num_hidden_layer)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_head.weight = self.embedding.word_embedding.weight

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ) -> GPT2Output:
        
        hidden_state = self.embedding(input_ids)
        for decoder in self.layers:
            hidden_state = decoder(hidden_state, hidden_state, hidden_state, attention_mask)

        hidden_state = self.lm_head(hidden_state)
        return


