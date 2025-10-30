import torch
from datasets import Dataset
from torch.utils.data import DataLoader


def flush_cache(l2_cache_size: int) -> torch.Tensor:
    cache = torch.empty(l2_cache_size, dtype=torch.int8, device="cuda")
    cache.zero_()
    return cache


def _generate_torchtune_data(num_samples: int, seq_len: int, vocab_size: int) -> Dataset:
    data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (seq_len,)).tolist()
        data.append({"input_ids": input_ids})
    return Dataset.from_list(data)


def _tokenize_torchtune_data(example):
    tokens = example["input_ids"]
    return {"tokens": tokens, "labels": tokens[1:] + [-100]}


def torchtune_loader(num_training_steps: int, batch_size: int, sequence_length: int, vocab_size: int):
    num_samples_for_dummy_data = num_training_steps * batch_size * 2
    dummy_dataset = _generate_torchtune_data(num_samples_for_dummy_data, sequence_length, vocab_size)
    tokenized_dataset = dummy_dataset.map(_tokenize_torchtune_data, batched=False)
    tokenized_dataset.set_format(type="torch", columns=["tokens", "labels"])
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    return iter(dataloader)

def mm_projector_flops(gbs, seq_len, vision_features, llm_features, act_recomp=True):
    flops1 = 2 * gbs * seq_len * vision_features * llm_features
    flops2 = 2 * gbs * seq_len * llm_features * llm_features
    total_flops = 3 * (flops1 + flops2)
    if act_recomp:
        total_flops = total_flops * (4/3)
    return total_flops

def vision_module_flops(gbs, in_channels, img_h, img_w, patch_dim, layers, hs, intermediate_size, llm_features, act_recomp=True):
    img_seq_len = (img_h // patch_dim) * (img_w // patch_dim)
    attn_flops = 3 * ((8 * hs * hs * img_seq_len) + (4 * img_seq_len * img_seq_len * hs))
    mlp_flops = 3 * 4 * img_seq_len * hs * intermediate_size
    transformer_flops = gbs * layers * (attn_flops + mlp_flops)
    embedding_flops = (
        3 * 2 * gbs * hs * in_channels * img_h * img_w
    )
    mm_flops = mm_projector_flops(gbs, img_seq_len, hs, llm_features)  # mm_projector의 FLOPs
    total_flops = transformer_flops + embedding_flops + mm_flops
    if act_recomp:
        total_flops = total_flops * (4/3)  
    return total_flops

def llm_module_flops(seq_len, gbs, hidden_size, layers, query_groups, attention_heads, ffn_hidden_size, vocab_size, act_recomp=True):    
    causal_self_attn = True
    gated_linear_multiplier = 2

    # --- Attention FLOPs ---
    attention_flops = (
        3 * 2 * gbs * layers * seq_len * hidden_size * hidden_size *
        (
            (query_groups / attention_heads * 2 + 1)
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))
            + 1
        )
    )

    # --- MLP FLOPs ---
    mlp_flops = (
        3 * 2 * gbs * layers * seq_len * hidden_size *
        (1 + gated_linear_multiplier) *
        ffn_hidden_size # 일반 ffn_hidden_size 사용
    )

    # --- Vocab FLOPs ---
    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size
    total_flops = attention_flops + mlp_flops + vocab_flops
    if act_recomp:
        total_flops = total_flops * (4/3)
    return total_flops