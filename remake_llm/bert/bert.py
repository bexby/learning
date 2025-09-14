import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from remake_llm.MyTransformers import Encoder
from dataclasses import dataclass
from transformers import BertTokenizer
import pdb

@dataclass
class BertConfig:
    hidden_size: int = 768
    vocab_size: int = 30522
    num_hidden_layer: int = 12
    num_head: int = 12
    max_len: int = 512

@dataclass
class BertOutput:
    last_hidden_state: torch.Tensor = None
    pooler_output: torch.Tensor = None
    loss: torch.Tensor = None

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
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_len, hidden_size)     # Bert uses learnable position embedding
        self.token_type_embedding = nn.Embedding(2, hidden_size)
        self.layernorm = nn.LayerNorm((hidden_size))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, indices, token_type):
        w_emb = self.word_embedding(indices)
        N, seq_len = indices.shape
        pos_indices = torch.stack([torch.arange(seq_len)] * N, dim=0)
        p_emb = self.pos_embedding(pos_indices)
        t_emb = self.token_type_embedding(token_type)
        embedding = w_emb + p_emb + t_emb
        embedding = self.layernorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embedding = BertEmbedding(config.vocab_size, config.hidden_size, config.max_len)
        self.layers = nn.ModuleList([Encoder(config.hidden_size, config.num_head, True) for _ in range(config.num_hidden_layer)])
        
        # for NSP(Next Sentence Prediction) training
        self.pooler = BertPooler(config.hidden_size)
        self.nsp_classifier = nn.Linear(config.hidden_size, 2)

        # for MLM(Masked language Modeling) training
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size)
        )
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        token_type_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor
        ) -> BertOutput:

        hidden_state = self.embedding(input_ids, token_type_ids)
        for encoder in self.layers:
            hidden_state = encoder(hidden_state, attention_mask)
        cls_tensor = self.pooler(hidden_state)

        loss = None
        if labels is not None:
            same_class_label = token_type_ids[:, 0]==token_type_ids[:, -1]  # (N,)
            # pdb.set_trace()
            same_class_label = torch.masked_fill(torch.zeros_like(same_class_label, dtype=torch.int64), same_class_label, 1)
            loss = self.compute_loss(hidden_state, labels, same_class_label)
        
        output = BertOutput(hidden_state.detach().clone(), cls_tensor.detach().clone(), loss)
        return output
    
    def compute_loss(self, last_hidden_state, labels, is_same_meaning):
        """
        MLM target
            example: masked sentence a b x c d
            given a, b, c, d, require x probability

        NSP target
            example: [cls] sentence A [seq] sentence B [seq] 
            utilize [cls] vector to classify whether B is the next sentence of A
        """
        mlm_output = self.mlm_transform(last_hidden_state)  # (N, seq_len, hidden_size)
        mlm_output = self.mlm_decoder(mlm_output)   # (N, seq_len, vocab_size)
        mlm_output = mlm_output.reshape(-1, mlm_output.shape[-1])   # (N*seq_len, vocab_size)
        mlm_loss = F.cross_entropy(mlm_output, labels.view(-1), ignore_index=-100)

        cls_tensor = self.pooler(last_hidden_state)   # (N, hidden_size)
        cls_tensor = self.nsp_classifier(cls_tensor)    # (N, 2)
        nsp_loss = F.cross_entropy(cls_tensor, is_same_meaning)
        return mlm_loss + nsp_loss
        
        
# emb = BertEmbedding(20, 5, 60)
# indices = torch.randint(0, 20, (2, 50))
# print(emb(indices).shape)

def test_bert_data_stream():
    bert_config = BertConfig()
    bert = Bert(bert_config)

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

def main():
    tokenizer_ckp = "google-bert/bert-base-uncased"
    sentence = "OpenAI is not open"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_ckp)
    input = tokenizer([sentence], return_tensors="pt")
    bert_config = BertConfig()
    bert = Bert(bert_config)
    bert.train()
    output = bert(**input, labels=input["input_ids"])
    print(output.loss)
    print(output.last_hidden_state.shape)
    output.loss.backward()


if __name__ == "__main__":
    main()