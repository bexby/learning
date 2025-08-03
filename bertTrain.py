import os
import torch
import bert
from transformers import BertTokenizer

ckp = "google-bert/bert-base-uncased"
sentence = "OpenAI is not open"

tokenizer = BertTokenizer.from_pretrained(ckp)

print(tokenizer([sentence]))



