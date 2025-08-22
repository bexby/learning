# mapping_hf_to_my.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig
from copy import deepcopy
from gpt2 import GPT2, GPT2Config
import pdb


def load_weight():
    # 假设你的 GPT2 / GPT2Config 在当前 namespace 中可用
    # from your_module import GPT2, GPT2Config

    # 1) load HF model (cpu or cuda)
    hf_name = "gpt2"  # 或者 "gpt2-medium" / "gpt2-large" 等
    hf_model = GPT2LMHeadModel.from_pretrained(hf_name)
    hf_sd = hf_model.state_dict()

    # 2) instantiate your model with same config
    cfg = GPT2Config(
        hidden_size=hf_model.config.n_embd,
        vocab_size=hf_model.config.vocab_size,
        num_hidden_layer=hf_model.config.n_layer,
        num_head=hf_model.config.n_head,
        max_len=hf_model.config.n_positions
    )
    my = GPT2(cfg)   # 记得你要先修复 ModuleList 的问题
    my_sd_keys = set(my.state_dict().keys())

    # helper utilities
    def assign(new_sd, my_key, tensor):
        # always store a CPU tensor (or same device as my model param)
        new_sd[my_key] = tensor.clone().detach().contiguous()

    def to_linear_weight(w, out_dim, in_dim):
        # want shape (out_dim, in_dim) for nn.Linear.weight
        if w.shape == (out_dim, in_dim):
            return w.clone().detach().contiguous()
        elif w.shape == (in_dim, out_dim):
            return w.t().contiguous()
        else:
            # try transpose & contiguous otherwise raise
            # but we prefer explicit error so you can inspect
            raise RuntimeError(f"unexpected weight shape: {w.shape}, want {(out_dim,in_dim)} or {(in_dim,out_dim)}")

    mapped_hf_keys = set()
    new_sd = {}

    # 3) embeddings (wte / wpe)
    assign(new_sd, "embedding.word_embedding.weight", hf_sd["transformer.wte.weight"])
    mapped_hf_keys.add("transformer.wte.weight")

    assign(new_sd, "embedding.position_embedding.weight", hf_sd["transformer.wpe.weight"])
    mapped_hf_keys.add("transformer.wpe.weight")

    # 4) per-layer mapping
    n_layer = cfg.num_hidden_layer
    d_model = cfg.hidden_size
    for i in range(n_layer):
        prefix_hf = f"transformer.h.{i}"
        prefix_my = f"layers.{i}"

        # LayerNorms: ln_1 -> layernorm1, ln_2 -> layernorm2
        assign(new_sd, f"{prefix_my}.layernorm1.weight", hf_sd[f"{prefix_hf}.ln_1.weight"]); mapped_hf_keys.add(f"{prefix_hf}.ln_1.weight")
        assign(new_sd, f"{prefix_my}.layernorm1.bias",   hf_sd[f"{prefix_hf}.ln_1.bias"]);   mapped_hf_keys.add(f"{prefix_hf}.ln_1.bias")
        assign(new_sd, f"{prefix_my}.layernorm2.weight", hf_sd[f"{prefix_hf}.ln_2.weight"]); mapped_hf_keys.add(f"{prefix_hf}.ln_2.weight")
        assign(new_sd, f"{prefix_my}.layernorm2.bias",   hf_sd[f"{prefix_hf}.ln_2.bias"]);   mapped_hf_keys.add(f"{prefix_hf}.ln_2.bias")

        # Attention: HF has c_attn (combined QKV) and c_proj (out proj)
        W_qkv = hf_sd[f"{prefix_hf}.attn.c_attn.weight"]; mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.weight")
        b_qkv = hf_sd[f"{prefix_hf}.attn.c_attn.bias"];   mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.bias")

        # W_qkv must be shape (d, 3*d) depending on Conv1D implementation.
        if W_qkv.ndim != 2:
            raise RuntimeError("unexpected c_attn weight dim")
        # get three chunks shaped like (d, d) AFTER conversion
        # transpose, then chunk
        W_q, W_k, W_v = W_qkv.t().contiguous().chunk(3, dim=0)


        if b_qkv.shape[0] == 3 * d_model:
            b_q, b_k, b_v = b_qkv.split(d_model, dim=0)

        # Now ensure each weight is (out=in=d_model, in=d_model) for my Linear
        W_q = to_linear_weight(W_q, d_model, d_model)
        W_k = to_linear_weight(W_k, d_model, d_model)
        W_v = to_linear_weight(W_v, d_model, d_model)

        assign(new_sd, f"{prefix_my}.self_attn.q_w.weight", W_q); mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.weight (q slice)")
        assign(new_sd, f"{prefix_my}.self_attn.q_w.bias",   b_q); mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.bias (q slice)")

        assign(new_sd, f"{prefix_my}.self_attn.k_w.weight", W_k); mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.weight (k slice)")
        assign(new_sd, f"{prefix_my}.self_attn.k_w.bias",   b_k); mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.bias (k slice)")

        assign(new_sd, f"{prefix_my}.self_attn.v_w.weight", W_v); mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.weight (v slice)")
        assign(new_sd, f"{prefix_my}.self_attn.v_w.bias",   b_v); mapped_hf_keys.add(f"{prefix_hf}.attn.c_attn.bias (v slice)")

        # c_proj (attention output projection)
        W_proj = hf_sd[f"{prefix_hf}.attn.c_proj.weight"]; mapped_hf_keys.add(f"{prefix_hf}.attn.c_proj.weight")
        b_proj = hf_sd[f"{prefix_hf}.attn.c_proj.bias"];   mapped_hf_keys.add(f"{prefix_hf}.attn.c_proj.bias")
        # convert to (out,in) if needed
        W_proj_lin = to_linear_weight(W_proj.t(), d_model, d_model)
        assign(new_sd, f"{prefix_my}.self_attn.linear.weight", W_proj_lin)
        assign(new_sd, f"{prefix_my}.self_attn.linear.bias",   b_proj)

        # MLP / FNN: c_fc -> linear1, c_proj -> linear2
        W_fc = hf_sd[f"{prefix_hf}.mlp.c_fc.weight"]; mapped_hf_keys.add(f"{prefix_hf}.mlp.c_fc.weight")
        b_fc = hf_sd[f"{prefix_hf}.mlp.c_fc.bias"];   mapped_hf_keys.add(f"{prefix_hf}.mlp.c_fc.bias")
        W_proj_mlp = hf_sd[f"{prefix_hf}.mlp.c_proj.weight"]; mapped_hf_keys.add(f"{prefix_hf}.mlp.c_proj.weight")
        b_proj_mlp = hf_sd[f"{prefix_hf}.mlp.c_proj.bias"];   mapped_hf_keys.add(f"{prefix_hf}.mlp.c_proj.bias")
        # pdb.set_trace()
        # HF c_fc mast be shape (d, 4d); we want linear1.weight shape (4d, d)
        W_fc_lin = to_linear_weight(W_fc, 4*d_model, d_model)
        W_proj_mlp_lin = to_linear_weight(W_proj_mlp, d_model, 4*d_model)
        assign(new_sd, f"{prefix_my}.fnn.linear1.weight", W_fc_lin)
        assign(new_sd, f"{prefix_my}.fnn.linear1.bias",   b_fc)
        assign(new_sd, f"{prefix_my}.fnn.linear2.weight", W_proj_mlp_lin)
        assign(new_sd, f"{prefix_my}.fnn.linear2.bias",   b_proj_mlp)

    # 5) final ln_f -> your model may not have it
    if "transformer.ln_f.weight" in hf_sd and "transformer.ln_f.bias" in hf_sd:
        if "ln_f.weight" in my.state_dict() or "ln_f.bias" in my.state_dict():
            # if your model had ln_f key (you added it), map it
            assign(new_sd, "ln_f.weight", hf_sd["transformer.ln_f.weight"]); mapped_hf_keys.add("transformer.ln_f.weight")
            assign(new_sd, "ln_f.bias",   hf_sd["transformer.ln_f.bias"]); mapped_hf_keys.add("transformer.ln_f.bias")
        else:
            print("HF has final ln_f but your model has no ln_f; skipping mapping for transformer.ln_f.*")

    # 6) lm_head: your model ties lm_head.weight to embedding. 仍然把 lm_head.weight 一并装上（对照）
    assign(new_sd, "lm_head.weight", hf_sd["lm_head.weight"]); mapped_hf_keys.add("lm_head.weight")

    # 7) report / load
    # Which HF keys we actually touched:
    print("Mapped HF keys (approx):", len(mapped_hf_keys))

    # Check which HF keys were not used (informative)
    hf_keys_not_mapped = sorted(set(hf_sd.keys()) - set([k.split()[0] if " (" in k else k for k in mapped_hf_keys]))
    print("HF keys not mapped (examples):", hf_keys_not_mapped[:50])

    # Load into your model (non-strict to surface missing/unexpected)
    missing, unexpected = my.load_state_dict(new_sd, strict=False)
    print("Model keys missing in new_sd (these params in model were NOT filled):", missing)
    print("Unexpected keys in new_sd (these keys don't match model):", unexpected)

    return my


def test_aline(my_model):
    HF_NAME = "gpt2"
    DEVICE = torch.device("cpu")     # 把 device 改成 "cuda" 如你有 GPU 并希望用 GPU
    TEXT = "Hello, my dog is cute"   # 测试文本，可以替换成任意句子 / 批量输入

    # -------------- 加载 HF 模型（官方） --------------
    hf_tokenizer = GPT2Tokenizer.from_pretrained(HF_NAME)
    hf_model = GPT2LMHeadModel.from_pretrained(HF_NAME).to(DEVICE).eval()

    # -------------- 准备你的模型（my_model） --------------
    # 这里假定你的 GPT2 类已经定义并且你已用之前的映射脚本把 HF 权重加载进去了。
    # 若你的模型在另一个模块中，替换下面两行来实例化：
    # from my_gpt_impl import GPT2, GPT2Config
    # cfg = GPT2Config(...) ; my_model = GPT2(cfg)
    # 并且确保 my_model 已 load_state_dict(...) 完成
    try:
        # 尝试直接引用名为 GPT2 的类
        GPT2  # 如果未定义会触发 NameError
    except NameError:
        raise RuntimeError("找不到 GPT2 类，请把你的实现放在可 import / 可访问的作用域，或在脚本里实例化 my_model。")

    # instantiate your model with same config as HF
    cfg = type("C", (), {})()   # 临时空对象放置必要字段，如果你的 GPT2 需要具体 config，请改这里
    # 下面字段尽量与 HF 对齐
    cfg.hidden_size = hf_model.config.n_embd
    cfg.vocab_size = hf_model.config.vocab_size
    cfg.num_hidden_layer = hf_model.config.n_layer
    cfg.num_head = hf_model.config.n_head
    cfg.max_len = hf_model.config.n_positions

    my_model.eval()

    # -------------- 准备输入 --------------
    inputs = hf_tokenizer(TEXT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)             # (B, T)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64).to(DEVICE)  # 0 表示 padding，与你的实现语义保持一致

    # -------------- 1) 获取 HF 的 per-layer hidden states --------------
    # 注意：hf_model(..., output_hidden_states=True) 会返回 hidden_states tuple，
    # 通常长度为 (n_layers + 1) —— 第0项是 embedding 输出（word+pos embedding），之后每项对应 transformer.h.{i} 的输出（在 ln_f 之前）。
    with torch.no_grad():
        hf_out = hf_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    hf_hidden_states = tuple(hf_out.hidden_states)  # tuple of tensors, each (B, T, D)

    print("HF hidden_states count:", len(hf_hidden_states))
    # examples: for gpt2-base len(hf_hidden_states) == n_layer + 1

    # -------------- 2) 在 my_model 上逐层得到 hidden states（手动模拟 forward） --------------
    # 这里不调用 my_model.forward()，而按你实现的顺序手动迭代 layer，以便收集每一层输出
    with torch.no_grad():
        # get initial embedding (你的 GPT2Embedding 返回 dropout; eval() 已禁用 dropout)
        my_embed = my_model.embedding(input_ids)   # (B, T, D)
        my_hidden_states = []
        my_hidden_states.append(my_embed)
        h = my_embed
        for i, block in enumerate(my_model.layers):
            # block 的 signature: block(hidden_state, key, value, key_padding_mask)
            h = block(h, h, h, attention_mask)
            my_hidden_states.append(h)

    print("my_model layers count:", len(my_hidden_states))
    pdb.set_trace()
    # -------------- 3) 对齐 HF 的 hidden_states 与 my_hidden_states --------------
    # HF 的 hidden_states 很可能是 [embedding, layer0_out, layer1_out, ..., layerN_out]
    # 我们要把 hf_layer_states 对齐为与 my_hidden_states 一一对应（layer0..layerN-1）
    if len(hf_hidden_states) == len(my_hidden_states) + 1:
        hf_layer_states = list(hf_hidden_states)[1:]   # drop initial embedding
    elif len(hf_hidden_states) == len(my_hidden_states):
        hf_layer_states = list(hf_hidden_states)
    else:
        raise RuntimeError(f"HF hidden_states length ({len(hf_hidden_states)}) 与 my_hidden_states length ({len(my_hidden_states)}) 无法直接对齐")

    # sanity check shapes
    for i, (a, b) in enumerate(zip(hf_layer_states, my_hidden_states)):
        assert a.shape == b.shape, f"layer {i} shape mismatch: hf {a.shape} vs my {b.shape}"

    # -------------- 4) 逐层计算 hidden MAE 与 logits MAE --------------
    hf_lm = hf_model.lm_head   # linear layer (tied)
    my_lm = my_model.lm_head

    per_layer = []
    for i, (hf_h, my_h) in enumerate(zip(hf_layer_states, my_hidden_states)):
        hf_h = hf_h.to(torch.float32)
        my_h = my_h.to(torch.float32)

        # hidden MAE
        mae_hidden = (hf_h - my_h).abs().mean().item()

        # logits MAE - 直接把两边的 hidden 输入各自的 lm_head
        # 注意：我们在这里对中间层（非最终层）也直接用了 lm_head（相当于把相同的线性层应用到中间表示）
        hf_logits = hf_lm(hf_h)
        my_logits = my_lm(my_h)
        mae_logits = (hf_logits - my_logits).abs().mean().item()

        per_layer.append((mae_hidden, mae_logits))
        print(f"Layer {i:02d} | hidden MAE: {mae_hidden:.6e} | logits MAE: {mae_logits:.6e}")

    # -------------- 5) 汇总 --------------
    hidden_maes = [x[0] for x in per_layer]
    logits_maes = [x[1] for x in per_layer]
    print("----- Summary -----")
    print(f"Mean hidden MAE across layers : {sum(hidden_maes)/len(hidden_maes):.6e}")
    print(f"Mean logits MAE across layers : {sum(logits_maes)/len(logits_maes):.6e}")
    print("Per-layer (hidden, logits) MAE list:")
    for i, (h_mae, l_mae) in enumerate(per_layer):
        print(f"  L{i:02d}: hidden={h_mae:.6e}, logits={l_mae:.6e}")


def check_weight(my_model):
    hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    my = my_model.state_dict()     # dict of tensors
    n_layer = hf["transformer.h.0.attn.c_attn.weight"].shape[0] // hf["transformer.h.0.attn.c_attn.bias"].shape[0]  # just to infer; optional

    def stats(t):
        return float(t.abs().mean()), float(t.abs().max())

    # compare embeddings / lm_head
    pairs = [
        ("transformer.wte.weight", "embedding.word_embedding.weight"),
        ("transformer.wpe.weight", "embedding.position_embedding.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ]
    print("=== Embedding / lm_head comparisons ===")
    for a,b in pairs:
        if a in hf and b in my:
            diff = hf[a].float() - my[b].float()
            print(b, "mean_abs_diff", *stats(diff))
        else:
            print("missing key:", a, "or", b)

    # per-layer QKV / c_proj / mlp checks
    n_layer = sum(1 for k in hf.keys() if k.startswith("transformer.h.") and k.endswith(".attn.c_attn.weight"))  # number of c_attn keys
    print("n_layer (hf):", n_layer)
    for i in range(n_layer):
        hf_prefix = f"transformer.h.{i}"
        # c_attn
        W = hf[f"{hf_prefix}.attn.c_attn.weight"]  # shape could be (3*d, d) or (d, 3*d)
        b = hf[f"{hf_prefix}.attn.c_attn.bias"]
        # normalize to (3, d, d) as HF describes -> get Wq,Wk,Wv each (d,d)
        if W.shape[0] == 3 * my_model.embedding.word_embedding.weight.shape[1]:
            Wq, Wk, Wv = W.chunk(3, dim=0)
        elif W.shape[1] == 3 * my_model.embedding.word_embedding.weight.shape[1]:
            # transpose then chunk
            Wq, Wk, Wv = W.t().chunk(3, dim=0)
        else:
            print("unexpected c_attn shape", W.shape)
            continue

        # find corresponding my keys (try likely names)
        candidates = [
            (f"layers.{i}.self_attn.q_w.weight", "q_w"),
            (f"layers.{i}.self_attn.k_w.weight", "k_w"),
            (f"layers.{i}.self_attn.v_w.weight", "v_w"),
            (f"layers.{i}.self_attn.linear.weight", "c_proj"),
            (f"layers.{i}.fnn.linear1.weight", "c_fc"),
            (f"layers.{i}.fnn.linear2.weight", "c_proj_mlp"),
        ]
        print("--- Layer", i, "---")
        # compare q/k/v
        for my_key, name in candidates[:3]:
            if my_key in my:
                myW = my[my_key].float()
                # ensure same orientation: myW shape likely (out,in)
                target = {"q_w": Wq, "k_w": Wk, "v_w": Wv}[name]
                # make both (out,in)
                if target.shape != myW.shape:
                    if target.t().shape == myW.shape:
                        target2 = target.t().float()
                    else:
                        print("shape mismatch for", my_key, "hf", target.shape, "my", myW.shape)
                        continue
                else:
                    target2 = target.float()
                d = (target2 - myW).abs()
                print(my_key, "mean_abs", float(d.mean()), "max_abs", float(d.max()))
            else:
                print("my key not found:", my_key)

        # compare c_proj
        my_key = f"layers.{i}.self_attn.linear.weight"
        if my_key in my:
            myW = my[my_key].float()
            Wproj = hf[f"{hf_prefix}.attn.c_proj.weight"]
            # align orientation
            if Wproj.shape != myW.shape:
                if Wproj.t().shape == myW.shape:
                    Wproj2 = Wproj.t().float()
                else:
                    print("shape mismatch for c_proj", Wproj.shape, myW.shape)
                    Wproj2 = None
            else:
                Wproj2 = Wproj.float()
            if Wproj2 is not None:
                d = (Wproj2 - myW).abs()
                print(my_key, "mean_abs", float(d.mean()), "max_abs", float(d.max()))

        # mlp
        for my_key, hf_key in [(f"layers.{i}.fnn.linear1.weight", f"{hf_prefix}.mlp.c_fc.weight"),
                            (f"layers.{i}.fnn.linear2.weight", f"{hf_prefix}.mlp.c_proj.weight")]:
            if my_key in my and hf_key in hf:
                myW = my[my_key].float()
                W_hf = hf[hf_key]
                if W_hf.shape != myW.shape:
                    if W_hf.t().shape == myW.shape:
                        W2 = W_hf.t().float()
                    else:
                        print("shape mismatch mlp", hf_key, W_hf.shape, my_key, myW.shape)
                        continue
                else:
                    W2 = W_hf.float()
                d = (W2 - myW).abs()
                print(my_key, "mean_abs", float(d.mean()), "max_abs", float(d.max()))
            else:
                print("mlp key missing:", my_key, hf_key)


def test_generation(mygpt2):
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

if __name__ == "__main__":
    mygpt2 = load_weight().eval()
    test_generation(mygpt2)
    # test_aline(mygpt2)
    # check_weight(mygpt2)