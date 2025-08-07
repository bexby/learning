import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import datasets
from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

print(ds["text"[:5]])
