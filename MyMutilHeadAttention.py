import torch
import torch.nn as nn
from collections import OrderedDict
import pdb


class MyMutilHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head, bias=True):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        if embedding_dim % num_head != 0:
            raise ValueError("could not find a equal dim to all heads")
        

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        if not query.shape[-1] == key.shape[-1] == value.shape[-1]:
            raise ValueError("query, key, values' dim are not equal")
        elif key.shape[1] != value.shape[1]:
            raise ValueError("key_seq_len is not equal to value_seq_len")
        
        key_t = key.transpose(dim0=1, dim1=2)
        head_list = []
        for sub_query, sub_key, sub_value in zip(query.chunk(self.num_head, -1), key_t.chunk(self.num_head, 1), value.chunk(self.num_head, -1)):
            mul_res = torch.bmm(sub_query, sub_key) / self.embedding_dim ** 0.5
            sf_res = self.softmax(mul_res)  #(N, q_seq_len, k_seq_len)
            head_res = torch.bmm(sf_res, sub_value) #(N, q_seq_len, head_dim)
            head_list.append(head_res)
        
        result = torch.cat(head_list, dim=-1)
        return self.linear(result)
        

mha = MyMutilHeadAttention(64, 4)
query = torch.randn((2, 100, 64))
key = torch.randn((2, 50, 64))
value = torch.randn((2, 50, 64))


result = mha(query, key, value)
print(result.shape)


nn.MultiheadAttention().forward()
