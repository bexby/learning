import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MyTransformers import Decoder
from dataclasses import dataclass
from transformers import GPT2Tokenizer, GenerationConfig
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
    logits: torch.Tensor = None
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
        labels: torch.Tensor = None
        ) -> GPT2Output:
        
        """
        Args:
        input_ids:
            (N, max_batch_seq_len)
        attention_mask:
            (N, max_batch_seq_len), 0 represent the padding position
        labels:
            (N, max_batch_seq_len), must be the input_ids
        """

        hidden_state = self.embedding(input_ids)
        for decoder in self.layers:
            hidden_state = decoder(hidden_state, hidden_state, hidden_state, attention_mask)

        logits = self.lm_head(hidden_state) # (N, max_batch_seq_len, vocab_size)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        output = GPT2Output(hidden_state.detach().clone(), logits.detach().clone(), loss)

        return output
    
    def compute_loss(self, logits, labels):
        flatten_input = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        return F.cross_entropy(flatten_input, labels[:, 1:].reshape(-1))


    def generate(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig
        ):
        
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = 50256
        
        gen_len = max(1, generation_config.max_length - input_ids.shape[-1])
        if generation_config.max_new_tokens is not None:
            gen_len = max(gen_len, generation_config.max_new_tokens)

        # TODO: finish the top p pipeline
        for _ in range(gen_len):
            logits = self.forward(input_ids, attention_mask).logits[:, -1, :].detach().clone()
            probability = F.softmax(logits / generation_config.temperature, dim=-1) # (N, vocab_size)
            if generation_config.do_sample:
                if generation_config.top_k is None and generation_config.top_p is None:
                    raise ValueError("do sample but both top_k and top_p is None")
                topk_logits, topk_indices = probability, torch.stack([torch.arange(probability.shape[-1])]*probability.shape[0], dim=0)
                if generation_config.top_k is not None:
                    topk_logits, topk_indices = torch.topk(probability, generation_config.top_k, dim=-1)

                
                
        

def test_gpt2():
    gpt2_config = GPT2Config()
    gpt2 = GPT2(gpt2_config)
    tokenizer_ckp = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_ckp)
    tokenizer.pad_token = tokenizer.eos_token   # GPT2 doesn't has pad token
    prompt = ["In a raining day a poor man", "OpenAI is not open because"]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = gpt2(**inputs, labels=inputs["input_ids"])
    print(outputs.last_hidden_state.shape)
    print(outputs.logits.shape)
    print(outputs.loss)
    outputs.loss.backward()


if __name__ == "__main__":
    test_gpt2()
