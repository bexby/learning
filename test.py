import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, GPT2Model, GPT2LMHeadModel
from transformers.models.bert import BertModel
import torch.nn as nn
import torch
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")


prompt = ["In a raining day a poor man", "OpenAI is not open because"]

input = gpt2_tokenizer(prompt, return_tensors="pt")
output = gpt2(input)
gpt2.generate()
print(output)
