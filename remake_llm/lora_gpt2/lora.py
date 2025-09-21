import os
import torch
import torch.nn as nn
from remake_llm.MyTransformers import MyMultiHeadAttention
from remake_llm.gpt2.gpt2 import GPT2, GPT2Config
from remake_llm.gpt2.load_gpt2_weight import load_weight
from typing import Optional
from dataclasses import dataclass


@dataclass
class LoraConfig:
    r: int = 4
    target_model = MyMultiHeadAttention
    lora_alpha: float = 16


class MultiHeadAttentionForLoRA(nn.Module):
    def __init__(self, model, config: LoraConfig):
        super().__init__()
        if not isinstance(model, config.target_model):
            raise ValueError(f"LoRA injected only in {config.target_model}")
        
        self.embedding_dim = model.embedding_dim
        self.num_head = model.num_head
        self.lora_alhpa = config.lora_alpha
        for name, item in model.named_children():
            self.__setattr__(name, item)
        in_feature = self.embedding_dim
        out_feature = config.r
        self.qw_lora_a = nn.Parameter(torch.empty((in_feature, out_feature)))
        self.qw_lora_b = nn.Parameter(torch.empty((out_feature, in_feature)))
        self.vw_lora_a = nn.Parameter(torch.empty((in_feature, out_feature)))
        self.vw_lora_b = nn.Parameter(torch.empty((out_feature, in_feature)))

        nn.init.kaiming_uniform_(self.qw_lora_a)
        nn.init.kaiming_uniform_(self.vw_lora_a)


    def forward(
        self, 
        query: Optional[torch.Tensor], 
        key: Optional[torch.Tensor], 
        value: Optional[torch.Tensor], 
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.tensor:
        
        """
        Args:

        query (N, q_seq_len, emb_d):
        key (N, k_seq_len, emb_d):
        value (N, v_seq_len, emb_d):
        key_padding_mask (N, k_seq_len): 
            for three type attentions, we just to prevent useless padding token 
            in "key" from participating weight score. If isinstance(int), 0 represents
            the padding positions; if isinstance(bool), True represents the padding 
            positions
        attn_mask (k_seq_len, k_seq_len): 
            only for decoder self-attention (q_seq_len = k_seq_len)

        """

        N, q_seq_len, emb_d = query.shape
        k_seq_len = key.shape[1]
        head_dim = self.embedding_dim // self.num_head

        if emb_d != self.embedding_dim:
            raise ValueError(f"embedding dim must be {self.embedding_dim}")
        elif not query.shape[-1] == key.shape[-1]:
            raise ValueError("query and key's dim are not equal")
        elif key.shape[1] != value.shape[1]:
            raise ValueError("key_seq_len is not equal to value_seq_len")
        elif key_padding_mask is not None and key_padding_mask.dtype != torch.int64 and key_padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask.dtype must be torch.int64 or torch.bool")
        
        delta_qw = self.qw_lora_a @ self.qw_lora_b
        delta_vw = self.vw_lora_a @ self.vw_lora_b
    

        Q = (self.q_w(query) + query @ delta_qw.t()).reshape(N, q_seq_len, self.num_head, head_dim) 
        K = self.k_w(key).reshape(N, k_seq_len, self.num_head, head_dim)
        V = (self.v_w(value) + value @ delta_vw.t()).reshape(N, k_seq_len, self.num_head, head_dim)
        
        Q_n, K_n, V_n = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)     # (N, num_head, seq_len, dim)
        mul_res = torch.matmul(Q_n, K_n.transpose(-1, -2))  # (N, num_head, q_seq_len, k_seq_len)
        

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = ~ key_padding_mask.bool()
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand((N, self.num_head, q_seq_len, k_seq_len))
            # pdb.set_trace()
            mul_res = torch.masked_fill(mul_res, key_padding_mask, -1e5)
                
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.expand((N, self.num_head, q_seq_len, k_seq_len))
            mul_res = torch.masked_fill(mul_res, attn_mask, -1e5)

        sf_res = self.softmax(mul_res / head_dim ** 0.5)  # (N, num_head, q_seq_len, k_seq_len)
        """ 
        IMPORTANCE!!! 
            If mask fill with -torch.inf, a row of NaN will appear in score matirx when pad token's row is full of -inf before softmax, 
            and will be propagate to entire matirx. This situation is common in decoder's inference stage with 'left' padding side.
            The better solution is masking with a small negative value , e.g. -1e5, to prevent NaN.
        """
        # if key_padding_mask is not None:    
        #     sf_res = sf_res.masked_fill(torch.isnan(sf_res), 0.0)      

        sf_res = self.dropout(sf_res)
        head_res = torch.matmul(sf_res, V_n)    # (N, num_head, q_seq_len, head_dim)
        result = head_res.transpose(1, 2).contiguous().reshape(N, q_seq_len, self.embedding_dim)    # (N, q_seq_len, embedding_dim)
        return self.linear(result)


def get_peft_model(model, config):
    if isinstance(config, LoraConfig):
        for name, item in model.named_modules():
            if isinstance(item, config.target_model):
                lora_mha = MultiHeadAttentionForLoRA(item, config)
                path = name.split(".")
                parent = model
                for p in path[:-1]:
                    parent = parent.get_submodule(p)
                setattr(parent, path[-1], lora_mha)
        
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    return model



def main():
    # mha = MyMultiHeadAttention(64, 4)
    # lora_config = LoraConfig()
    # lora_mha = MultiHeadAttentionForLoRA(mha, lora_config)
    # input = torch.randn((1, 4, 64))
    # print(lora_mha(input, input, input).shape)
    tokenizer_ckp = "gpt2"
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_ckp)
    tokenizer.pad_token = tokenizer.eos_token   # GPT2 doesn't has pad token
    tokenizer.padding_side = "left"
    prompt = ["Hello, my dog is cute", "OpenAI is not open because", "jack has"]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # inputs = {k: torch.ones_like(v, dtype=torch.int64) if k == "attention_mask" else v for k, v in inputs.items()}
    with torch.no_grad():
        my_output = mygpt2(**inputs)
        hf_output = hf_model(**inputs, output_hidden_states=True, return_dict=True)
    print("last_hidden_state difference: ", (my_output.last_hidden_state - hf_output.hidden_states[-1]).abs().mean().item())
    print("logits difference: ", (my_output.logits - hf_output.logits).abs().mean().item())
    
    gen_config = GenerationConfig(do_sample=False, max_length=50)
    hf_text = hf_model.generate(inputs["input_ids"], gen_config, attention_mask=inputs["attention_mask"])
    gen_text = mygpt2.generate(**inputs, generation_config=gen_config)
    print(tokenizer.batch_decode(gen_text))
    print(tokenizer.batch_decode(hf_text))

    pass

if __name__ == "__main__":
    main()