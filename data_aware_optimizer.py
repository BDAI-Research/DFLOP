import math
import itertools
import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from utils.model import llm_configs, vision_configs
from collections import defaultdict
from utils.config import (
    get_config_path,
    load_config,
    resolve_path,
    reset_config_cache,
)

class LlmThr:
    def __init__(self, linear_thr_tp, llm_linear_ratio, llm_attn_ratio, llm_attn_weighted_sum):
        self.linear_thr_tp = linear_thr_tp
        self.llm_linear_ratio = llm_linear_ratio
        self.llm_attn_ratio = llm_attn_ratio
        self.llm_attn_weighted_sum = llm_attn_weighted_sum
    def __call__(self, seq_len):
        linear_thr = self.linear_thr_tp(seq_len)
        llm_thr = self.llm_linear_ratio * linear_thr + self.llm_attn_ratio * self.llm_attn_weighted_sum
        return llm_thr

class ModelMem:
    def __init__(self, model_mem_factors, act_mem_factors, num_layers):
        self.model_mem_factors = model_mem_factors
        self.act_mem_factors = act_mem_factors
        self.num_layers = num_layers
        
    def __call__(self, tp, pp, batch_size, seq_len, total_pp):
        num_layers = np.ceil(self.num_layers / pp)
        model_cur_mem, model_state_mem = calculate_model_peak(self.model_mem_factors, num_layers, tp)
        act_mem = total_pp * calculate_act_peak(self.act_mem_factors, num_layers, tp, batch_size, seq_len)
        peak_mem = calculate_peak_mem(model_cur_mem, model_state_mem, act_mem)
        # print(f"Current : {model_cur_mem:.2f}, State : {model_state_mem:.2f}, Act : {act_mem:.2f}, Peak : {peak_mem:.2f}")
        return peak_mem


def calculate_model_state_factor(df):
    tp_list = df["tp_size"].unique().tolist()
    model_tp_dict = {}
    for tp_size in tp_list:
        target_df = df[df["tp_size"] == tp_size]
        target_df = target_df.sort_values(by=["num_layers"])
        first = target_df.iloc[0]
        second = target_df.iloc[1]
        model_state_var = (second["model_state_peak"] - first["model_state_peak"]) / (second["num_layers"] - first["num_layers"])
        model_state_constant = first["model_state_peak"] - first["num_layers"] * model_state_var
        model_mem_var = (second["model_mem"] - first["model_mem"]) / (second["num_layers"] - first["num_layers"])
        model_mem_constant = first["model_mem"] - first["num_layers"] * model_mem_var
        model_tp_dict[tp_size] = [model_state_constant, model_state_var, model_mem_constant, model_mem_var]
    return model_tp_dict

def calculate_act_peak_factor(df, image_size=None, patch_size=14):
    df_copy = deepcopy(df)
    if "seq_len" not in df_copy.columns:
        df_copy["seq_len"] = 1
    tp_list = df["tp_size"].unique()
    act_peak_tp = {}
    for tp in tp_list:
        tp_df = df_copy[df_copy["tp_size"] == tp].reset_index(drop=True)
        first = tp_df.loc[0, ["num_layers", "fwd_peak", "batch_size", "seq_len"]]
        second = tp_df.loc[1, ["num_layers", "fwd_peak"]]
        batch_size = first["batch_size"]
        if image_size is not None:
            seq_len = (image_size // patch_size) ** 2
        else:
            seq_len = first["seq_len"]
        b = (second["fwd_peak"] - first["fwd_peak"]) / (second["num_layers"] - first["num_layers"])
        a = first["fwd_peak"] - first["num_layers"] * b
        trans_alpha_tp = b / (batch_size * seq_len)
        non_trans_alpha_tp = a / (batch_size * seq_len)
        act_peak_tp[tp] = [trans_alpha_tp, non_trans_alpha_tp]
    return act_peak_tp

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

    # 전체 트랜스포머 연산량
    transformer_flops = gbs * layers * (attn_flops + mlp_flops)

    # 4. Patch Embedding 연산량
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
    gated_linear_multiplier = 2 # SwiGLU 사용

    # --- Attention FLOPs (Qwen3와 동일) ---
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

    # --- Vocab FLOPs (Qwen3와 동일) ---
    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size
    total_flops = attention_flops + mlp_flops + vocab_flops
    if act_recomp:
        total_flops = total_flops * (4/3)
    # print(f"Attention FLOPs: {attention_flops/1e12}, MLP FLOPs: {mlp_flops/1e12}, Vocab FLOPs: {vocab_flops/1e12}")
    return total_flops

def get_attn_flops(seq_len, embed_dim, num_hidden_layers, act_recomp=True):
    attn_flops = 3 * (4 * seq_len * seq_len * embed_dim) * num_hidden_layers * 0.5
    if act_recomp:
        attn_flops = attn_flops * (4/3)
    return attn_flops

def parse_vision_thr_df(vision_df, vision_model_name, vision_model_size, llm_model_name, llm_model_size):
    if vision_model_name == "siglip":
        image_size = 384
    elif vision_model_name == "internvit":
        image_size = 448
    else:
        raise ValueError(f"Unsupported vision model: {vision_model_name}")
    vision_config = vision_configs[vision_model_name][vision_model_size]
    patch_size = 14
    in_channels = 3
    img_h = image_size
    img_w = image_size
    patch_dim = patch_size
    hs = vision_config.hidden_size
    intermediate_size = vision_config.intermediate_size
    llm_config = llm_configs[llm_model_name][llm_model_size]
    llm_hidden_size = llm_config["embed_dim"]
    vision_df["model_flops"] = vision_df[["batch_size", "num_layers"]].apply(lambda x: vision_module_flops(x[0], in_channels, img_h, img_w, patch_dim, x[1], hs, intermediate_size, llm_hidden_size), axis=1)
    vision_df["model_thr"] = vision_df["model_flops"] / (vision_df["model_time"] * 1e12 * vision_df["tp_size"]) * 1000
    vision_df = vision_df[["batch_size", "num_layers", "model_thr", "tp_size"]].round(2)
    # target_df = vision_df[vision_df["num_layers"]==8].reset_index(drop=True)
    return vision_df

def parse_llm_thr_df(llm_full_df, llm_skip_attn_df, llm_configs, llm_model_name, llm_model_size):
    llm_config = llm_configs[llm_model_name][llm_model_size]
    gbs = 1
    hidden_size = llm_config["embed_dim"]
    num_kv_heads = llm_config["num_kv_heads"]
    attention_heads = llm_config["num_heads"]
    query_groups = attention_heads // num_kv_heads
    ffn_hidden_size = llm_config["intermediate_dim"]
    if llm_model_name == "qwen2.5":
        vocab_size = 152064
    elif llm_model_name == "llama3":
        vocab_size = 128256
    common_keys = ["seq_len", "num_layers", "tp_size"]
    merged_df = pd.merge(llm_full_df, llm_skip_attn_df, on=common_keys, suffixes=('_full', '_skip'))
    merged_df["attn_time"] = merged_df["model_time_full"] - merged_df["model_time_skip"]
    merged_df["linear_time"] = merged_df["model_time_full"] - merged_df["attn_time"]
    merged_df = merged_df.round(2)
    merged_df["model_flops"] = merged_df[["seq_len", "num_layers"]].apply(lambda x : llm_module_flops(x[0], gbs, hidden_size, x[1], query_groups, attention_heads, ffn_hidden_size, vocab_size, act_recomp=True), axis=1)
    merged_df["attn_flops"] = merged_df[["seq_len", "num_layers"]].apply(lambda x : get_attn_flops(x[0], hidden_size, x[1], act_recomp=True), axis=1)
    merged_df["linear_flops"] = merged_df["model_flops"] - merged_df["attn_flops"]
    merged_df["attn_thr"] = merged_df["attn_flops"] / (merged_df["attn_time"] * 1e12 * merged_df["tp_size"]) * 1000
    merged_df["linear_thr"] = merged_df["linear_flops"] / (merged_df["linear_time"] * 1e12 * merged_df["tp_size"]) * 1000
    merged_df = merged_df[["seq_len", "num_layers", "tp_size", "attn_thr", "linear_thr"]].round(2)
    # merged_df = merged_df[merged_df["num_layers"] == 8].reset_index(drop=True)
    return merged_df

def get_vision_tp_thr(df, tp_size):
    df_copy = deepcopy(df)
    target_df = df_copy[df_copy["tp_size"] == tp_size].reset_index(drop=True)
    model_f = interp1d(target_df["batch_size"], target_df["model_thr"], kind="linear", fill_value="extrapolate")
    return model_f

def get_llm_tp_thr(df, tp_size):
    df_copy = deepcopy(df)
    target_df = df_copy[df_copy["tp_size"] == tp_size].reset_index(drop=True)
    attn_f = interp1d(target_df["seq_len"], target_df["attn_thr"], kind="linear", fill_value="extrapolate")
    linear_f = interp1d(target_df["seq_len"], target_df["linear_thr"], kind="linear", fill_value="extrapolate")
    return attn_f, linear_f

def get_mem_prof(result_path, mllm_model_name, vision_model_name, vision_model_size, llm_model_name, llm_model_size):
    vision_df_path = f"{result_path}/mem_{mllm_model_name}_{vision_model_name}_{vision_model_size}.csv"
    llm_df_path = f"{result_path}/mem_{mllm_model_name}_{llm_model_name}_{llm_model_size}.csv"
    vision_df = pd.read_csv(vision_df_path)
    llm_df = pd.read_csv(llm_df_path)
    if vision_model_name == "siglip":
        image_size = 384
        patch_size = 14
    elif vision_model_name == "internvit":
        image_size = 448
        patch_size = 14
    else:
        raise ValueError(f"Unsupported vision model: {vision_model_name}")

    vision_model_factors = calculate_model_state_factor(vision_df)
    vision_act_mem_factors = calculate_act_peak_factor(vision_df, image_size=image_size, patch_size=patch_size)
    llm_model_factors = calculate_model_state_factor(llm_df)
    llm_act_mem_factors = calculate_act_peak_factor(llm_df)
    mem_dict = {"vision" : [vision_model_factors, vision_act_mem_factors], "llm": [llm_model_factors, llm_act_mem_factors]}
    return mem_dict

def get_thr_prof(result_path, mllm_model_name, vision_model_name, vision_model_size, llm_model_name, llm_model_size):
    vision_df_path = f"{result_path}/thr_{mllm_model_name}_{vision_model_name}_{vision_model_size}.csv"
    llm_full_df_path = f"{result_path}/thr_{mllm_model_name}_{llm_model_name}_{llm_model_size}.csv"
    llm_skip_attn_df_path = f"{result_path}/thr_{mllm_model_name}_{llm_model_name}_{llm_model_size}_skip_attn.csv"
    ## Vision Profiling Results
    vision_df = pd.read_csv(vision_df_path)
    parsed_vision_df = parse_vision_thr_df(vision_df, vision_model_name, vision_model_size, llm_model_name, llm_model_size)
    vision_tp_list = parsed_vision_df["tp_size"].unique().tolist()
    vision_thr_dict = {tp : [] for tp in vision_tp_list}
    for tp in vision_tp_list:
        vision_model_tp = get_vision_tp_thr(parsed_vision_df, tp)
        vision_thr_dict[tp].append(vision_model_tp)

    ## LLM Profiling Results
    llm_full_df = pd.read_csv(llm_full_df_path)
    llm_skip_attn_df = pd.read_csv(llm_skip_attn_df_path)
    llm_full_df = llm_full_df[["seq_len", "num_layers", "tp_size", "model_time"]]
    llm_skip_attn_df = llm_skip_attn_df[["seq_len", "num_layers", "tp_size", "model_time"]]
    merged_df = parse_llm_thr_df(llm_full_df, llm_skip_attn_df, llm_configs, llm_model_name, llm_model_size)
    llm_tp_list = merged_df["tp_size"].unique().tolist()
    llm_thr_dict = {tp : [] for tp in llm_tp_list}
    for tp in llm_tp_list:
        llm_attn_tp, llm_linear_tp = get_llm_tp_thr(merged_df, tp)
        llm_thr_dict[tp].append((llm_attn_tp, llm_linear_tp))
    tp_func = {"llm": llm_thr_dict,
           "vision": vision_thr_dict}
    return tp_func

def calculate_model_peak(model_factors, num_layers, tp_size):
    state_constant, state_var, model_cur_constant, model_cur_var = model_factors[tp_size]
    model_state = state_constant + num_layers * state_var
    model_cur = model_cur_constant + num_layers * model_cur_var
    
    return model_cur, model_state

def calculate_act_peak(act_peak_tp, num_layers, tp_size, batch_size, seq_len):
    trans_alpha_tp, non_trans_alpha_tp = act_peak_tp[tp_size]
    act_tp = (non_trans_alpha_tp + trans_alpha_tp * num_layers) * (batch_size * seq_len)
    return act_tp

def calculate_throughput_factor(tp, factors):
    """TP에 따른 throughput 계수 계산"""
    if tp == 1: return 1.0
    val = 1.0
    # Note: This simplified calculation assumes factors are for powers of 2.
    # A more robust implementation might use logarithms as in the formula.
    if tp in factors: val = factors[tp]
    return val

def calculate_duration(flops, base_thr, tp, pp, dp, factors, num_microbatches, total_pp):
    """파이프라인 수행 시간을 계산하는 함수"""
    if not all([tp, pp, dp, total_pp]): return float('inf')
    
    throughput_factor = calculate_throughput_factor(tp, factors)
    total_throughput = (throughput_factor * base_thr * tp * pp * dp) * \
                       (num_microbatches / (total_pp - 1) if total_pp > 1 else 1)
    
    return flops / total_throughput if total_throughput > 0 else float('inf')


def calculate_peak_mem(model_cur_mem, model_state_mem, act_mem, buffer_ratio=1.25):
    # model_mem = max(model_cur_mem + act_mem, model_state_mem) * buffer_ratio
    model_mem = (model_state_mem + act_mem) * buffer_ratio
    return model_mem

if __name__=="__main__":
    default_config_path = get_config_path()
    os.environ.setdefault("DFLOP_CONFIG", str(default_config_path))
    reset_config_cache()
    config = load_config()

    models_cfg = config.get("models", {})
    vision_cfg = models_cfg.get("vision", {})
    llm_cfg = models_cfg.get("llm", {})

    result_path_resolved = resolve_path(config.get("paths", {}).get("profile_result_dir"))
    if result_path_resolved is None:
        raise ValueError("paths.profile_result_dir must be provided in the configuration.")
    result_path = str(result_path_resolved)

    mllm_model_name = models_cfg.get("mllm")
    vision_model_name = vision_cfg.get("name")
    vision_model_size = vision_cfg.get("size")
    llm_model_name = llm_cfg.get("name")
    llm_model_size = llm_cfg.get("size")

    hardware_cfg = config.get("hardware", {})
    training_cfg = config.get("training", {})
    n_gpus = hardware_cfg.get("n_gpus")
    gpu_memory = hardware_cfg.get("gpu_memory_gb")
    gbs = training_cfg.get("global_batch_size")

    required_fields = {
        "models.mllm": mllm_model_name,
        "models.vision.name": vision_model_name,
        "models.vision.size": vision_model_size,
        "models.llm.name": llm_model_name,
        "models.llm.size": llm_model_size,
        "hardware.n_gpus": n_gpus,
        "hardware.gpu_memory_gb": gpu_memory,
        "training.global_batch_size": gbs,
    }
    missing_keys = [key for key, value in required_fields.items() if value is None]
    if missing_keys:
        raise ValueError(f"Missing configuration values: {', '.join(missing_keys)}")

    n_gpus = int(n_gpus)
    gpu_memory = float(gpu_memory)
    gbs = int(gbs)

    with open(f"{result_path}/{mllm_model_name}_{vision_model_name}_{llm_model_name}.pkl", "rb") as f:
        data_dist_dict = pickle.load(f)
    image_batch_list = data_dist_dict["image_batch"]
    llm_input_seq_list = data_dist_dict["llm_input_seq"]

    # Memory Results
    mem_dict = get_mem_prof(result_path, mllm_model_name, vision_model_name, vision_model_size, llm_model_name, llm_model_size)

    # Throughput Results
    thr_dict = get_thr_prof(result_path, mllm_model_name, vision_model_name, vision_model_size, llm_model_name, llm_model_size)
    llm_thr_dict = thr_dict['llm']
    v_thr = thr_dict['vision']
    vision_model_mem_factors, vision_act_mem_factors = mem_dict["vision"]
    llm_model_mem_factors, llm_act_mem_factors = mem_dict["llm"]
    llm_config = llm_configs[llm_model_name][llm_model_size]
    llm_num_layers = llm_config["num_layers"]
    llm_hidden_size = llm_config["embed_dim"]
    hidden_size = llm_config["embed_dim"]
    num_kv_heads = llm_config["num_kv_heads"]
    attention_heads = llm_config["num_heads"]
    query_groups = attention_heads // num_kv_heads
    ffn_hidden_size = llm_config["intermediate_dim"]
    if llm_model_name == "qwen2.5":
        vocab_size = 152064
    elif llm_model_name == "llama3":
        vocab_size = 128256

    vision_config = vision_configs[vision_model_name][vision_model_size]
    vision_num_layers = vision_config.num_hidden_layers
    if vision_model_name == "siglip":
        image_size = 384
    elif vision_model_name == "internvit":
        image_size = 448
    else:
        raise ValueError(f"Unsupported vision model: {vision_model_name}")
    if mllm_model_name == "qwenvl":
        raise NotImplementedError
    else:
        patch_size = 14
        in_channels = 3
        img_h = image_size
        img_w = image_size
        patch_dim = patch_size
        hs = vision_config.hidden_size
        intermediate_size = vision_config.intermediate_size
    vision_flops = np.array([vision_module_flops(img_batch_size, in_channels, img_h, img_w, patch_dim, vision_num_layers, hs, intermediate_size, llm_hidden_size)/1e12 for img_batch_size in image_batch_list])



    llm_flops = np.array([llm_module_flops(llm_seq_len, 1, hidden_size, llm_num_layers, query_groups, attention_heads, ffn_hidden_size, vocab_size, act_recomp=True)/1e12 for llm_seq_len in llm_input_seq_list])
    attn_flops = np.array([get_attn_flops(llm_seq_len, hidden_size, llm_num_layers, act_recomp=True) for llm_seq_len in llm_input_seq_list])/1e12
    linear_flops = llm_flops - attn_flops
    llm_linear_ratio = np.sum(linear_flops) / np.sum(llm_flops)
    llm_attn_ratio = np.sum(attn_flops) / np.sum(llm_flops)
    l_thr = {}
    for l_tp in llm_thr_dict.keys():
        attn_thr_tp, linear_thr_tp = llm_thr_dict[l_tp][0]
        llm_attn_weighted_sum = attn_thr_tp([llm_seq_len for llm_seq_len in llm_input_seq_list]).mean()
        l_thr[l_tp] = LlmThr(linear_thr_tp, llm_linear_ratio, llm_attn_ratio, llm_attn_weighted_sum)

    v_mem = ModelMem(vision_model_mem_factors, vision_act_mem_factors, vision_num_layers)
    l_mem = ModelMem(llm_model_mem_factors, llm_act_mem_factors, llm_num_layers)  
    max_batch_size = max(image_batch_list)
    mean_batch_size = np.mean(image_batch_list)
    max_seq_len = max(llm_input_seq_list)
    mean_seq_len = np.mean(llm_input_seq_list)
    mean_v_flop = vision_flops.mean()
    mean_l_flop = llm_flops.mean()

    # Vision
        ## LLaVA-OV, InternVL : fixed sequence length for vision model
        ## QwenVL : variable sequence length for vision model
    # LLM
        ## batch size = 1, variable sequence length for LLM
    vision_seq_len = (image_size // patch_size) ** 2
    # Vision : 1,2,4,8,16,32,64
    # LLM : 1024,2048, 4096, 8192
    v_thr_result_list = []
    llm_thr_result_list = []
    for tp_size in [1,2,4,8]:
        for batch_size in [1,2,4,8,16,32]:
            v_thr_result = v_thr[tp_size][0](batch_size).item()
            v_thr_result_list.append((tp_size, batch_size, v_thr_result))
        if tp_size in llm_thr_dict.keys():
            for seq_len in [1024,2048,4096,8192]:
                llm_thr_result = l_thr[tp_size](seq_len).item()
                llm_thr_result_list.append((tp_size, seq_len, llm_thr_result))
    # parallel_configs = []
 
parallel_configs = []
for v_gpus in range(1, n_gpus):
    l_gpus = n_gpus - v_gpus
    v_configs = []
    for v_tp in v_thr.keys():
        if v_gpus % v_tp == 0:
            # v_pp * v_dp가 되어야 할 값
            v_rem = v_gpus // v_tp
            for v_pp in range(1, v_rem + 1):
                if v_rem % v_pp == 0:
                    v_dp = v_rem // v_pp
                    v_configs.append([v_tp, v_pp, v_dp])
    
    if not v_configs:
        continue

    l_configs = []
    for l_tp in llm_thr_dict.keys():
        if l_gpus % l_tp == 0:
            # l_pp * l_dp가 되어야 할 값
            l_rem = l_gpus // l_tp
            for l_pp in range(1, l_rem + 1):
                if l_rem % l_pp == 0:
                    l_dp = l_rem // l_pp
                    l_configs.append([l_tp, l_pp, l_dp])

    for v_c in v_configs:
        for l_c in l_configs:
            parallel_configs.append(v_c + l_c)

min_dur = float('inf')
opt_m_batches = float('inf')
opt_config = None
passed_configs = []
for p_config in parallel_configs:
    v_tp, v_pp, v_dp, l_tp, l_pp, l_dp = p_config
    v_memory = v_mem(v_tp, v_pp, max_batch_size * l_dp / v_dp, vision_seq_len, v_pp + l_pp)
    l_memory = l_mem(l_tp, l_pp, 1, max_seq_len, l_pp)
    if (v_memory > gpu_memory) or (l_memory > gpu_memory):
        continue
    passed_configs.append(p_config)
    max_m = int(gbs / l_dp)
    for m in range(1, max_m+1):
        l_mbs = gbs / (m * l_dp)
        v_mbs = gbs / (m * v_dp)
        target_batch_size = mean_batch_size * v_mbs
        target_seq_len = mean_seq_len * l_mbs
        if v_mbs < 1 or l_mbs < 1:
            continue
        # if target_batch_size > max_batch_size or target_seq_len > max_seq_len:
        #     continue
        if target_batch_size <= max_batch_size and target_seq_len <= max_seq_len:
            break
        # print(f"Passed batch/seq check for config: {p_config} with m: {m}, v_mbs: {v_mbs}, l_mbs: {l_mbs}, target_batch_size: {target_batch_size}, target_seq_len: {target_seq_len}")
    target_v_thr = v_thr[v_tp][0](target_batch_size)
    target_l_thr = l_thr[l_tp](target_seq_len)
    v_flop = (mean_v_flop * v_mbs)
    l_flop = (mean_l_flop * l_mbs)
    v_dur = v_flop / (target_v_thr * v_tp * v_pp)
    l_dur = l_flop / (target_l_thr * l_tp * l_pp)
    slowest_stage = max(v_dur, l_dur)
    duration = (m + v_pp + l_pp - 1) * slowest_stage
    
    if (duration < min_dur * 1.1) and (m < opt_m_batches):
        opt_m_batches = m
        min_dur = duration
        opt_config = p_config + [m]
        log_dict = {
            "v_tp": v_tp,
            "v_pp": v_pp,
            "v_dp": v_dp,
            "l_tp": l_tp,
            "l_pp": l_pp,
            "l_dp": l_dp,
            "num_microbatches": m,
            "v_mbs": v_mbs,
            "l_mbs": l_mbs,
            "target_batch_size": target_batch_size,
            "target_seq_len": target_seq_len,
            "vision_memory": v_memory,
            "llm_memory": l_memory,
            "v_duration": v_dur,
            "l_duration": l_dur,
            "bottleneck_duration": slowest_stage,
            "total_duration": duration,
            "v_thr" : target_v_thr,
            "l_thr" : target_l_thr,
        }

if opt_config is not None and 'log_dict' in locals():
    print("\n--- Optimal 3D parallelism ---")
    for k, v in log_dict.items():
        print(f"{k:>20}: {v}")
