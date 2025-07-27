import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional
import pdb


class MyMutilHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head, bias=True):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.q_w = nn.Linear(embedding_dim, embedding_dim, bias)
        self.k_w = nn.Linear(embedding_dim, embedding_dim, bias)
        self.v_w = nn.Linear(embedding_dim, embedding_dim, bias)
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias)

        if embedding_dim % num_head != 0:
            raise ValueError("could not find a equal dim to all heads")
        

    def forward(
        self, 
        query: Optional[torch.Tensor], 
        key: Optional[torch.Tensor], 
        value: Optional[torch.Tensor], 
        mask: Optional[torch.Tensor] = None
    ) -> torch.tensor:
        
        """
        Args:
        query (N, q_seq_len, emb_d)
        key (N, k_seq_len, emb_d)
        value (N, v_seq_len, emb_d)
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
        
        Q = self.q_w(query).reshape(N, q_seq_len, self.num_head, head_dim) 
        K = self.k_w(query).reshape(N, k_seq_len, self.num_head, head_dim)
        V = self.v_w(query).reshape(N, k_seq_len, self.num_head, head_dim)
        
        Q_n, K_n, V_n = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)     # (N, num_head, seq_len, dim)
        mul_res = torch.matmul(Q_n, K_n.transpose(-1, -2))  # (N, num_head, q_seq_len, k_seq_len)
        mul_res = self.dropout(mul_res)
        if mask is not None:
            if mask.dtype != query.dtype:
                mask = mask.to(query.dtype)
            mul_res = torch.matmul(mul_res, mask)
        sf_res = self.softmax(mul_res / head_dim ** 0.5)  # (N, num_head, q_seq_len, k_seq_len)
        head_res = torch.matmul(sf_res, V_n)    # (N, num_head, q_seq_len, head_dim)
        result = head_res.transpose(1, 2).contiguous().reshape(N, q_seq_len, self.embedding_dim)    # (N, q_seq_len, embedding_dim)
        return self.linear(result)
        

# mha = MyMutilHeadAttention(64, 4)
# query = torch.randn((2, 100, 64))
# key = torch.randn((2, 50, 64))
# value = torch.randn((2, 50, 64))


# result = mha(query, key, value)
# print(result.shape)

class FNN(nn.Module):
    def __init__(self, hidden_size, bias):
        super.__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4, bias)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size, bias)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state):
        output = self.linear1(hidden_state)
        output = self.activate(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        return output


class Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_head: int, bias: bool = False):
        super().__init__()
        self.attention = MyMutilHeadAttention(hidden_size, num_head, bias)
        self.fnn = FNN(hidden_size, bias)
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_state, mask):
        residual = hidden_state
        mha_output = self.attention(hidden_state, hidden_state, hidden_state, mask)
        mha_output = self.dropout(mha_output)
        residual_output1 = residual + mha_output
        norm_res1 = self.layernorm(residual_output1)

        fnn_output = self.fnn(norm_res1)
        residual_output2 = norm_res1 + fnn_output
        norm_res2 = self.layernorm(residual_output2)
        return norm_res2


class decoder(nn.Module):
    def __init__(self, hidden_size: int, num_head: int, bias: bool = True):
        super().__init__()
        self.cross_attn = MyMutilHeadAttention(hidden_size, num_head, bias)
        self.self_attn = MyMutilHeadAttention(hidden_size, num_head, bias)
        self.fnn1 = FNN(hidden_size, bias)
        self.fnn2 = FNN(hidden_size, bias)
        self.dropout = nn.Dropout(0.1)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)

    def forward(
        self, 
        hidden_state: Optional[torch.tensor], 
        key: Optional[torch.tensor], 
        value: Optional[torch.tensor],
        mask: Optional[torch.tensor] = None
        ) -> torch.tensor:

        residual = hidden_state
        norm_res1 = self.layernorm1(hidden_state)
        self_mha = self.self_attn(norm_res1, norm_res1, norm_res1, mask)
        self_mha = self.dropout(self_mha)
        residual_output1 = self_mha + residual
        
        norm_res2 = self.layernorm2(residual_output1)
        fnn_output1 = self.fnn1(norm_res2)
        residual_output2 = residual_output1 + fnn_output1
        
        norm_res3 = self.layernorm1(residual_output2)
        self_mha = self.cross_attn(norm_res3, key, value)
        self_mha = self.dropout(self_mha)
        residual_output1 = self_mha + residual

        

        pass
        



myencoder = Encoder(12, 4)

# print(myencoder)
data = torch.randn((2, 10 ,12))
mask = torch.randint(0, 2, (10, 10))
print(myencoder(data, mask=mask).shape)

# nn.MultiheadAttention().forward()

