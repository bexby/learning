from transformers import AutoModel, AutoTokenizer
from transformers import BertModel
from transformers.models.bert import BertModel
import torch.nn as nn
import torch
bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

embedding = bert.embeddings
word_embedding = embedding.word_embeddings
pos_embedding = embedding.position_embeddings
ty_embedding = embedding.token_type_embeddings
print(word_embedding)
print(pos_embedding)
print(ty_embedding)