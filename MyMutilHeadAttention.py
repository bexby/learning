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
        sf_res = self.softmax(mul_res / head_dim ** 0.5)  # (N, num_head, q_seq_len, k_seq_len)
        head_res = torch.matmul(sf_res, V_n)
        result = head_res.reshape(N, q_seq_len, self.embedding_dim)
        return self.linear(result)
        

# mha = MyMutilHeadAttention(64, 4)
# query = torch.randn((2, 100, 64))
# key = torch.randn((2, 50, 64))
# value = torch.randn((2, 50, 64))


# result = mha(query, key, value)
# print(result.shape)



class Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_head: int, bias: bool = False):
        super().__init__()
        self.attention = MyMutilHeadAttention(hidden_size, num_head, bias)
        self.fnn1 = nn.Linear(hidden_size, hidden_size * 4, bias)
        self.fnn2 = nn.Linear(4 * hidden_size, hidden_size, bias)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
    
    def forward(self, hidden_state):
        mha_output = self.attention(hidden_state, hidden_state, hidden_state)
        error_connect = hidden_state + mha_output
        norm_res = F.layer_norm(error_connect, error_connect.shape[1:])
        linear_output1 = self.fnn1(norm_res)
        act_output = self.activate(linear_output1)
        linear_output2 = self.fnn2(act_output)
        result = error_connect + linear_output2
        norm_result = F.layer_norm(result, result.shape[1:])
        return self.dropout(norm_result)


myencoder = Encoder(12, 4)

data = torch.randn((2, 10 ,12))

print(myencoder(data).shape)

# nn.MultiheadAttention().forward()

