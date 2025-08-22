# Toy Experiment 手搓大模型  
## 目标  
- 实现多头注意力，朴素Encoder、Decoder、Transformer **from scratch**
- 实现Bert、GPT2、Llama3.1 推理/训练  
- 编写LLMs微调/训练框架  
- 实现分布式训练/推理， flash attention  
## TODO  
- ~~GPT2 推理/训练/generate~~
- Llama3.1 推理/训练
- LLMs 训练框架  

## 记录
### 1. GPT2推理时的logits NaN问题
> 带有padding的句子中最后的logits全为NaN  

**分析**：设计`multiheadattention`时只考虑到右padding，这样不会出现分数矩阵某一行都被mask的情况，并且使用了`-torch.inf`作为masked fill。而gpt2需要左padding，再结合注意力mask就会出现某一行在softmax后都是NaN，在后继的`decoder`模块中继续相乘就会出现该样本的所有hidden state都是NaN的情况  
**解决**：~~使用mask将NaN行置为全0，但是实现和HF不同，最后的logits会和HF有所偏差(体现在padding的部分)。如果是右padding的话则不需要~~   和HF做法对齐，使用较小的负数`-1e5`替代`-torch.inf`，并且不需要增加进一步的mask，对左右padding都适用

### 2.未经预训练的GPT2重复最后一个单词的现象
在得分矩阵中，某token自己的Q，K相乘得分肯定是最高的，未训练时显著高于其他token，在softmax中优势进一步被放大，因此和v矩阵相乘后和原来的矩阵变化不大，加上残差连接机制，每一层的输出都和输入相似。GPT2的`LM Head`和embedding共享权重，而`last_hidden_state`和embedding长得差不多，因此做linear时每个token的预测分布更倾向于自己，就导致LLM重复最后一个词的趋势。  

### 3.加载HuggingFace GPT2权重的问题
HF的GPT2实现中，几乎所有的`Linear`层都是用`nn.Covd1d`实现的，因此要先将`nn.Covd1d`权重**转置后再拷贝**到`Linear.weight`。**（此处因为多头注意力连接的投影矩阵未转置而debug了好久，一度怀疑模型实现有问题😭）**。对于hidden_state在非encoder-decoder模型中转化为kqv，**HF是将三个投影矩阵拼接成(d, 3*d)即`nn.Covd1d`的权重，进行一次矩阵乘法一次性得到的**，此处要对应好`Linear`的实现。

### 4.Generation异常现象
> 输入: [[a, b, c, d], [pad, pad, a, b]]  
> 解码策略: 贪婪解码。
> 
> - 左padding  
> GPT2对于第一句话和hf的一样,第二句话不断重复输出的第一个单词   
> - 右padding  
> GPT2对于第一句话和hf的一样,第二句话前半段和HF的一样，但是后半段就开始变了  
> - attention mask固定为全是1的tensor  
> 发现无论是左还是右padding，两句话输出和hf的完全一样。  
> - 对预测生成第一个单词的logits进行.mean()的比较  
> 以上的实验配置条件下结果都在均在1e-5的数量级  

**分析**：计算`position_ids`时没有将pad token考虑在内，对于generate HF有prepare_input_for_generate方法进行预处理，而forward则没有这步，用arange生成

