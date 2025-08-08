import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, GPT2Model, GPT2LMHeadModel
from transformers.models.bert import BertModel
import torch.nn as nn
import torch
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
print(gpt2)
# embedding = bert.embeddings
# word_embedding = embedding.word_embeddings
# pos_embedding = embedding.position_embeddings
# ty_embedding = embedding.token_type_embeddings
# print(word_embedding)
# print(pos_embedding)
# print(ty_embedding)