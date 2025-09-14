# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import re
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

"""
构造用于 BERT (MLM + NSP) 的训练集：
输出 dataset 中包含字段：
- text1: token-id 列表（text1 的 masked 版本，不含 [CLS]/[SEP]）
- text2: token-id 列表（text2 的 masked 版本，不含 [CLS]/[SEP]）
- label: 0/1 (NSP)
- mask_labels: 整个拼接序列 [CLS] text1 [SEP] text2 [SEP] 对应的 MLM labels (非 mask 位置为 -100)
此外还会输出 input_ids, token_type_ids, attention_mask（方便后续训练）
"""



# ========== 可配置参数 ==========
TOKENIZER_NAME = "bert-base-uncased"   # 或者替换为你自己的 tokenizer 路径/对象
WIKITEXT_CONFIG = "wikitext-103-raw-v1"  # 或 "wikitext-2-raw-v1"
SPLIT = "train"                          # 用哪个 split
MAX_SEQ_LENGTH = 512
MLM_PROB = 0.15
NSP_POS_PROB = 0.5       # 生成正样本（真实下一句）的概率，负样本概率 = 1 - NSP_POS_PROB
SEED = 42
BATCH_SAVE_EVERY = None  # None：一次性返回全部；若内存受限可改成整数，分批写入磁盘/处理

# ========== 初始化 ==========
random.seed(SEED)
np.random.seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

# vocab_size 获取（有些 tokenizer 使用 vocab_size 属性）
try:
    VOCAB_SIZE = tokenizer.vocab_size
except:
    VOCAB_SIZE = len(tokenizer.get_vocab())

CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id
MASK_ID = tokenizer.mask_token_id
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else None

SPECIAL_IDS = {CLS_ID, SEP_ID}
if PAD_ID is not None:
    SPECIAL_IDS.add(PAD_ID)

# ========== 文本切分函数（把每个 dataset item 的 text 分成句子） ==========
_sentence_split_regex = re.compile(r'(?<=[\.\?\!])\s+')

def split_into_sentences(paragraph: str):
    """
    简易句子分割：先按换行分行（去掉空行和 wiki heading），再按标点分句。
    对 WikiText 来说通常能得到合适的句子。
    """
    if not paragraph or not paragraph.strip():
        return []
    lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip() and not ln.strip().startswith("=")]
    sents = []
    for ln in lines:
        parts = _sentence_split_regex.split(ln)
        for p in parts:
            p = p.strip()
            if p:
                # 可在这里加入更多的清洗规则（去掉 [[...]] 等），但假设 raw 数据已清洗或你已预处理
                sents.append(p)
    return sents

# ========== 截断对函数 ==========
def truncate_seq_pair(a_ids, b_ids, max_len):
    """Greedy truncate longer sequence until total length <= max_len"""
    while len(a_ids) + len(b_ids) > max_len:
        if len(a_ids) > len(b_ids):
            a_ids.pop()
        else:
            b_ids.pop()

# ========== 主要构造函数 ==========
def build_mlm_nsp_dataset_from_wikitext(tokenizer,
                                       wikitext_config=WIKITEXT_CONFIG,
                                       split=SPLIT,
                                       max_seq_length=MAX_SEQ_LENGTH,
                                       mlm_prob=MLM_PROB,
                                       nsp_pos_prob=NSP_POS_PROB,
                                       seed=SEED,
                                       debug_limit=None):
    """
    返回一个 HuggingFace datasets.Dataset，字段至少包含：
      text1, text2, label, mask_labels, input_ids, token_type_ids, attention_mask
    debug_limit: 若不为 None，则只对前 n 篇文档构造（便于调试）
    """
    random.seed(seed)
    np.random.seed(seed)

    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    texts = raw["text"]  # list of strings

    # 1) 切分成 docs（每个 doc 是句子列表）
    docs = []
    for idx, t in enumerate(tqdm(texts, desc="Splitting docs")):
        if debug_limit and idx >= debug_limit:
            break
        sents = split_into_sentences(t)
        if len(sents) >= 2:  # 需要至少两句才能产生正样本
            docs.append(sents)

    if len(docs) < 2:
        raise ValueError("切分后有效文档数太少，请检查数据或增大 debug_limit。")

    # 2) 遍历每个 doc，生成 NSP 对 (sentence_a, sentence_b, label)
    pairs = []
    for doc_idx, doc in enumerate(tqdm(docs, desc="Generating NSP pairs")):
        for i in range(len(doc) - 1):
            sent_a = doc[i]
            if random.random() < nsp_pos_prob:
                sent_b = doc[i + 1]
                label = 1
            else:
                # 从不同 doc 随机采样一个句子作为负样本
                other_doc_idx = random.randrange(len(docs))
                while other_doc_idx == doc_idx:
                    other_doc_idx = random.randrange(len(docs))
                sent_b = random.choice(docs[other_doc_idx])
                label = 0
            pairs.append((sent_a, sent_b, label))

    # 3) 对每个 pair 做 tokenize、截断、拼接、按整个序列做 MLM mask
    out_text1 = []
    out_text2 = []
    out_labels = []
    out_mask_labels = []
    out_input_ids = []
    out_token_type_ids = []
    out_attention_mask = []

    max_len_for_segments = max_seq_length - 3  # [CLS], [SEP], [SEP]

    for sent_a, sent_b, nsp_label in tqdm(pairs, desc="Tokenize & Masking pairs"):
        # token ids（不加 special tokens）
        a_ids = tokenizer.encode(sent_a, add_special_tokens=False)
        b_ids = tokenizer.encode(sent_b, add_special_tokens=False)

        # 若过长则截断（贪心）
        truncate_seq_pair(a_ids, b_ids, max_len_for_segments)

        # 构建拼接序列
        input_ids = [CLS_ID] + a_ids + [SEP_ID] + b_ids + [SEP_ID]
        orig_input_ids = list(input_ids)  # 复制保存原始 token ids（用于 mask_labels）

        # token_type_ids: segment ids (0 for CLS + a + SEP, 1 for b + SEP)
        token_type_ids = []
        # CLS + a + SEP -> 0
        token_type_ids += [0] * (1 + len(a_ids) + 1)
        # b + SEP -> 1
        token_type_ids += [1] * (len(b_ids) + 1)

        attention_mask = [1] * len(input_ids)

        # candidate positions for masking：排除 special tokens (CLS/SEP/PAD if present)
        cand_indices = [i for i, tid in enumerate(input_ids) if tid not in SPECIAL_IDS and (PAD_ID is None or tid != PAD_ID)]

        # how many to mask
        num_to_mask = max(1, int(round(len(cand_indices) * mlm_prob)))

        mask_indices = random.sample(cand_indices, k=num_to_mask) if num_to_mask <= len(cand_indices) else cand_indices

        # 创建 mask_labels，默认 -100
        mask_labels = [-100] * len(input_ids)

        for idx in mask_indices:
            original_id = orig_input_ids[idx]
            mask_labels[idx] = original_id  # 保存真实 id（用于计算 loss）
            dice = random.random()
            if dice < 0.8:
                # 80% -> [MASK]
                input_ids[idx] = MASK_ID
            elif dice < 0.9:
                # 10% -> 随机 token（避免选 special token）
                while True:
                    rid = random.randrange(VOCAB_SIZE)
                    if rid not in SPECIAL_IDS:
                        input_ids[idx] = rid
                        break
            else:
                # 10% -> 保持原 token（input_ids[idx] 已是原 id，不变）
                pass

        # 从拼接后的 input_ids 中提取 text1、text2 的 masked ids（不含 special tokens）
        # text1 在 input_ids 的索引范围是 [1, 1+len(a_ids)-1]
        a_masked = input_ids[1:1 + len(a_ids)]
        b_masked = input_ids[1 + len(a_ids) + 1 : 1 + len(a_ids) + 1 + len(b_ids)]

        # 保存结果
        out_text1.append(a_masked)
        out_text2.append(b_masked)
        out_labels.append(nsp_label)
        out_mask_labels.append(mask_labels)
        out_input_ids.append(input_ids)
        out_token_type_ids.append(token_type_ids)
        out_attention_mask.append(attention_mask)

    # 4) 生成 datasets.Dataset
    ds = Dataset.from_dict({
        "text1": out_text1,
        "text2": out_text2,
        "label": out_labels,
        "mask_labels": out_mask_labels,
        "input_ids": out_input_ids,
        "token_type_ids": out_token_type_ids,
        "attention_mask": out_attention_mask
    })

    return ds

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 若数据量太大，可把 debug_limit 设置为一个较小的数测试
    ds = build_mlm_nsp_dataset_from_wikitext(tokenizer, debug_limit=20)
    dl = DataLoader(ds, batch_size=4)
    for d in dl:
        print(d)
        break
    # print(ds)
    # # 展示第一个样本
    # sample = ds[0]
    # print("sample keys:", sample.keys())
    # print("label:", sample["label"])
    # print("input_ids (len):", len(sample["input_ids"]))
    # print("input_ids:", sample["input_ids"])
    # print("mask_labels (len):", len(sample["mask_labels"]))
    # print("mask_labels:", sample["mask_labels"])
    # print("text1 (masked ids):", sample["text1"])
    # print("text2 (masked ids):", sample["text2"])

