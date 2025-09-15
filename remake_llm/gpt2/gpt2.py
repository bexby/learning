import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from remake_llm.MyTransformers import Decoder
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
    
    def forward(self, indices: torch.Tensor, position_ids: torch.Tensor = None):
        we = self.word_embedding(indices)
        bl = indices.shape[-1]
        if position_ids is None:
            p_index = torch.arange(bl).unsqueeze(0)
            pe = self.position_embedding(p_index)
        else:
            pe = self.position_embedding(position_ids)
        embedding = we + pe
        return self.dropout(embedding)


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = GPT2Embedding(config.vocab_size, config.hidden_size, config.max_len)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([Decoder(config.hidden_size, config.num_head) for _ in range(config.num_hidden_layer)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.word_embedding.weight

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
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
        hidden_state = self.embedding(input_ids, position_ids)
        for decoder in self.layers:
            hidden_state = decoder(hidden_state, hidden_state, hidden_state, attention_mask)

        hidden_state = self.ln_f(hidden_state)
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

        with torch.no_grad():
            for _ in range(gen_len):
                input_ids, attention_mask, position_ids = self.prepare_inputs_for_generation(input_ids, attention_mask)
                logits = self.forward(input_ids, attention_mask, position_ids).logits[:, -1, :].detach().clone()
                probability = F.softmax(logits / generation_config.temperature, dim=-1) # (N, vocab_size)
                if generation_config.do_sample:
                    if generation_config.top_k is None and generation_config.top_p is None:
                        raise ValueError("do sample but both top_k and top_p is None")
                    if generation_config.top_k is not None:
                        top_probs, topk_indices = torch.topk(probability, generation_config.top_k, dim=-1)  # (N, k)
                        next_token = topk_indices.gather(-1, torch.multinomial(top_probs, 1))
                    if generation_config.top_p is not None:
                        _, topp_indices = self.top_p(top_probs, generation_config.top_p)    # different from torch.topk, self.top_p return the final sampled one in cumulate p
                        if generation_config.top_k is not None:     # if has been processed base on top k, the indices should be convert base on vocabulary 
                            topp_indices = torch.gather(topk_indices, -1, topp_indices)
                        next_token = topp_indices
                else:
                    # pdb.set_trace()
                    next_token = torch.argmax(probability, -1).unsqueeze(1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=torch.int64)), dim=1)
        
        return input_ids
    

    def prepare_inputs_for_generation(self, input_ids, attention_mask):
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 1)
        return input_ids, attention_mask, position_ids


    def top_p(self, data: torch.Tensor, p: float):
        """
        Args:
            data: (N, len)
            p: float
        
        Algorithm:
            use torch.topk to find a range rougthly, and then cumulate the probability, use binary-search to 
            find the p boundary
        
        Return:
            values (N, 1), indices (N, 1)
        """
        max_k = data.shape[-1]
        k = min(max_k, 2000)
        top_k, indices = torch.topk(data, k)    # top_k has been sorted by descending order
        cumsum = torch.cumsum(top_k, dim=-1)
        if not torch.all(cumsum[:, -1] > p).item():     # if k is not large enough to cover cumulated probability p 
            top_k, indices = torch.topk(data, max_k)
            cumsum = torch.cumsum(top_k, dim=-1)
        
        bs, max_len = cumsum.shape
        targets = torch.searchsorted(cumsum, torch.ones((bs, 1))*p)

        positions = torch.arange(max_len).unsqueeze(0).expand(bs, -1)
        top_k = top_k.masked_fill(positions > targets, 0)

        samples_idx = torch.multinomial(top_k, 1)
        res_indices = torch.gather(indices, -1, samples_idx)
        res_values = torch.gather(data, -1, res_indices)
        return res_values, res_indices            
                
        

def test_gpt2():
    gpt2_config = GPT2Config()
    gpt2 = GPT2(gpt2_config)
    tokenizer_ckp = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_ckp)
    tokenizer.pad_token = tokenizer.eos_token   # GPT2 doesn't has pad token
    tokenizer.padding_side = "left"
    tokenizer.pad
    prompt = ["In a raining day, a poor man use a", "OpenAI is not open because"]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # outputs = gpt2(**inputs, labels=inputs["input_ids"])
    # print(outputs.last_hidden_state.shape)
    # print(outputs.logits.shape)
    # print(outputs.loss)
    # outputs.loss.backward()
    gen_config = GenerationConfig()
    gen_text = gpt2.generate(**inputs, generation_config=gen_config)
    print(tokenizer.batch_decode(gen_text))

if __name__ == "__main__":
    test_gpt2()
