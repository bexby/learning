# GPT2参数量统计
## 模型参数
> - emb_dim: 嵌入维度
> - emb_dim2: 嵌入映射维度
> - vorb_size: 词表大小

### Embedding Layer * 1
- embedding: (vorb_size, emb_dim)

### Decoder * 12
#### MultiHeadAttention * 1
- KW: (emb_dim, emb_dim) + emb_dim
- QW: (emb_dim, emb_dim) + emb_dim
- VW: (emb_dim, emb_dim) + emb_dim
- OW: (emb_dim, emb_dim) + emb_dim
#### FNN * 1
- lin1: (emb_dim, emb_dim2) + emb_dim2
- lin2: (emb_dim2, emb_dim) + emb_dim


## 训练时显存占用
## 量化参数