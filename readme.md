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
### 1. GPT2推理时的logits Nan问题
设计`multiheadattention`时只考虑到右padding，这样不会出现分数矩阵某一行都被mask的情况，而gpt2需要左padding，再结合注意力mask就会出现某一行在softmax后都是Nan，在后继的`decoder`模块中继续相乘就会出现该样本的所有hidden state都是Nan的情况
### 2.未经预训练的GPT2重复最后一个单词的现象
在得分矩阵中，某token自己的Q，K相乘得分肯定是最高的，未训练时显著高于其他token，在softmax中优势进一步被放大，因此和v矩阵相乘后和原来的矩阵变化不大，加上残差连接机制，每一层的输出都和输入相似。GPT2的`LM Head`和embedding共享权重，而`last_hidden_state`和embedding长得差不多，因此做linear时每个token的预测分布更倾向于自己，就导致LLM重复最后一个词的趋势。
