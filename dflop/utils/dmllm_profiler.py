import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from dmllm_utils.utils import llm_configs
from dmllm_utils.data_utils import TrainConfig, LazySupervisedDataset, DataCollatorForSupervisedDataset, SigLipImageProcessor

import os
import torch
import pandas as pd
import argparse
from torch import nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    loss_parallel
)
from torchtune.modules.loss import LinearCrossEntropyLoss
from torchtune.utils import batch_to_device
from torchtune_models import flashqwen2, flashllama3
from utils.utils import prepare_mha_for_tp, torchtune_loader
from utils.profiler import  AllocatedMemContext
from dmllm_utils.utils import vision_configs, llm_configs, llava_ov_siglip_tp_plan, llavaov_internvit_tp_plan, internvl_internvit_tp_plan, apply_ac
from dmllm_codes.dmllm_utils.model_utils_backup2 import LLaVAOVMMConfig, LLaVAOVSigLipModule, LLaVAOVInternVitModule
from torch.nn.parallel import DistributedDataParallel

class VisionModule(nn.Module):
    def __init__(self, module):
        super(VisionModule, self).__init__()
        self.vision_module = module
    def forward(self, *args, **kwargs):
        return self.vision_module(*args, **kwargs)


def run_data_analysis(dataloader):
    data_iter = iter(dataloader)
    image_batch_list = []
    llm_input_seq_list = []
    for i in tqdm(range(10000)):
        batch = next(data_iter)
        image_batch_list.append(len(batch["images"]))
        llm_input_seq_list.append(batch["new_position_ids"].shape[1])
    return image_batch_list, llm_input_seq_list


def profile_model_init(model, model_dtype, tp_plan, local_rank, is_vision):
    '''
    Profiling model initialization (especially, peak memory allocation during tensor parallelism)
    '''
    _world_size = int(os.environ["WORLD_SIZE"])
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(1, _world_size), mesh_dim_names=("dp", "tp"))
    dp_mesh = device_mesh["dp"]
    tp_mesh = device_mesh["tp"]
    model = model.to(model_dtype)
    model = model.to(torch.cuda.current_device())
    if _world_size > 1:
        # model = parallelize_model(model, _world_size)
        model = prepare_mha_for_tp(model, tp_mesh)
        parallelize_module(
            model,
            device_mesh=tp_mesh,
            parallelize_plan=tp_plan)
    model = DistributedDataParallel(model, device_ids=[local_rank], device_mesh=dp_mesh)
    if is_vision:
        apply_ac(model.module.vision_module.encoder)
    else:
        apply_ac(model.module)
    return model

def profile_model_mem(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch):
    '''
    Profiling model memory usage during training including forward, backward and optimizer step
    '''
    if "labels" in batch.keys():
        batch_to_device(batch, torch.cuda.current_device())
        labels = batch.pop("labels")
    else:
        batch = batch.to("cuda")
        labels = None
    
    for i in range(5):
        torch.cuda.reset_peak_memory_stats()
        # Forward calculation
        before_fwd_peak_mem = torch.cuda.max_memory_allocated()
        # before_fwd_cur_mem = torch.cuda.memory_allocated()
        if model_dtype == torch.float32:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(batch)
        else:
            output = model(batch)
        if labels is not None:
            with loss_parallel():
                loss = loss_fn(output, labels)
        else:
            loss = output.sum()
        after_fwd_peak_mem = torch.cuda.max_memory_allocated()
        fwd_peak_mem = (after_fwd_peak_mem - before_fwd_peak_mem) / 1024 ** 3
        # Backward calculation
        if model_dtype == torch.float32:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # Optimizer step
        if model_dtype == torch.float32:
            if d_tensor_optimizer is not None:
                scaler.step(d_tensor_optimizer)
                d_tensor_optimizer.zero_grad()
            if local_tensor_optimizer is not None:
                scaler.step(local_tensor_optimizer)
                local_tensor_optimizer.zero_grad()
            scaler.update()
        else:
            if d_tensor_optimizer is not None:
                d_tensor_optimizer.step()
                d_tensor_optimizer.zero_grad()
            if local_tensor_optimizer is not None:
                local_tensor_optimizer.step()
                local_tensor_optimizer.zero_grad()
        model_mem = torch.cuda.memory_allocated() / 1024 ** 3
        model_state_peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    return fwd_peak_mem, model_mem, model_state_peak_mem

def profile_vision_memory(image_size, vision_module, model_dtype, tp_plan):
    '''
    Profiling memory usage of the LLM during initialization and training
    returns
        model_state_peak: Peak memory usage of the model state
        alpha: Scaling factor for memory usage (in terms of batch size, sequence length, and number of layers)
        constant_val: Constant value for memory usage (constant value calculated when model type is float32)
    '''
    # device = torch.cuda.current_device()
    # mm_config = LLaVAOVMMConfig("mlp2x_gelu", vision_config.hidden_size, llm_config["embed_dim"])
    # if model_name == "siglip":
    #     vision_module = LLaVAOVSigLipModule(vision_config, mm_config, model_dtype, device)
    #     image_size = 384
    # elif model_name == "internvit":
    #     vision_module = LLaVAOVInternVitModule(vision_config, mm_config, model_dtype, device).to(model_dtype)
    #     image_size = 448
    # else:
    #     raise ValueError(f"Unsupported model name: {model_name}")
    local_rank = int(os.environ["LOCAL_RANK"])
    model = VisionModule(vision_module)
    model = profile_model_init(model, model_dtype, tp_plan, local_rank, is_vision=True)
    d_tensor_param = [params for params in model.parameters() if isinstance(params, torch.distributed.tensor.DTensor)]
    local_tensor_param = [params for params in model.parameters() if not isinstance(params, torch.distributed.tensor.DTensor)]
    if len(d_tensor_param) != 0:
        d_tensor_optimizer = torch.optim.AdamW(d_tensor_param, lr=1e-4, weight_decay=0.01)
    else:
        d_tensor_optimizer = None
    if len(local_tensor_param) != 0:
        local_tensor_optimizer = torch.optim.AdamW(local_tensor_param, lr=1e-4, weight_decay=0.01)
    else:
        local_tensor_optimizer = None
    scaler = torch.amp.GradScaler()
    batch_size = 2
    batch = torch.randn(batch_size, 3, image_size, image_size).to(model_dtype)
    result_dict = {"batch_size" : [], "model_mem": [], "fwd_peak": [], "model_state_peak": []}
    fwd_peak, model_mem, model_state_peak = profile_model_mem(model_dtype, model, d_tensor_optimizer, local_tensor_optimizer, scaler, batch, loss_fn=None)
    result_dict["batch_size"].append(batch_size)
    result_dict["model_mem"].append(model_mem)
    result_dict["fwd_peak"].append(fwd_peak)
    result_dict["model_state_peak"].append(model_state_peak)
    return result_dict

def profile_llm_memory(llm_module, model_dtype, tp_plan):
    local_rank = int(os.environ["LOCAL_RANK"])
    loss_fn = LinearCrossEntropyLoss()
    loss_fn.set_model_output(llm_module)
    model = profile_model_init(model, model_dtype, tp_plan, local_rank, is_vision=False)
    d_tensor_param = [params for params in model.parameters() if isinstance(params, torch.distributed.tensor.DTensor)]
    local_tensor_param = [params for params in model.parameters() if not isinstance(params, torch.distributed.tensor.DTensor)]
    if len(d_tensor_param) != 0:
        d_tensor_optimizer = torch.optim.AdamW(d_tensor_param, lr=1e-4, weight_decay=0.01)
    else:
        d_tensor_optimizer = None
    if len(local_tensor_param) != 0:
        local_tensor_optimizer = torch.optim.AdamW(local_tensor_param, lr=1e-4, weight_decay=0.01)
    else:
        local_tensor_optimizer = None
    scaler = torch.amp.GradScaler()
    batch_size = 1
    seq_len = 1024
    result_dict = {"batch_size" : [], "seq_len" : [], "model_mem": [], "fwd_peak": [], "model_state_peak": []}
    num_training_steps = 1
    data_iter = torchtune_loader(num_training_steps, batch_size, seq_len, vocab_size)
    batch = next(data_iter)
    fwd_peak, model_mem, model_state_peak = profile_model_mem(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch)
    result_dict["batch_size"].append(batch_size)
    result_dict["seq_len"].append(seq_len)
    result_dict["model_mem"].append(model_mem)
    result_dict["fwd_peak"].append(fwd_peak)
    result_dict["model_state_peak"].append(model_state_peak)
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile model memory usage")
    parser.add_argument("--mllm_model_name", type=str, required=True, help="Name of MLLM model")
    parser.add_argument("--vision_model_name", type=str, help="Name of vision model")
    parser.add_argument("--vision_model_size", type=str, help="Size of vision model")
    parser.add_argument("--llm_model_name", type=str, help="Name of LLM model")
    parser.add_argument("--llm_model_size", type=str, help="Size of LLM model")
    parser.add_argument("--model_dtype", type=str, choices=["float32", "bfloat16"], default="bfloat16", help="Model data type")
    parser.add_argument("--num_hidden_layers", type=int, help="Number of hidden layers in the model")
    parser.add_argument("--profile_mode", type=str, choices=["data", "mem", "thr"], help="Profile mode")
    args = parser.parse_args()
    result_path = "/home/hyeonjun/dmllm_codes/paper_reuslts"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    model_dtype = getattr(torch, args.model_dtype)
    if args.profile_mode == "mem":
        if args.vision_model_name:
            df_path = os.path.join(result_path, f"mem_{args.vision_model_name}_{args.vision_model_size}.csv")
            llm_config = llm_configs["qwen2"]["7b"] # arcbitrary llm config for vision model profiling
            vision_config = vision_configs[args.vision_model_name][args.vision_model_size]
            vision_tp_plans = {"llavaov": {"siglip": llava_ov_siglip_tp_plan, "internvit": llavaov_internvit_tp_plan},
                            "internvl": {"internvit": internvl_internvit_tp_plan}
                            #    "qwenvl":{"qwenvit": qwenvl_tp_plan}
                            }
            vision_tp_plan = vision_tp_plans[args.mllm_model_name][args.vision_model_name]
            llavaov_mm_config = LLaVAOVMMConfig("mlp2x_gelu", vision_config.hidden_size, llm_config["embed_dim"])
            vision_modules = {"llavaov" : {"siglip": LLaVAOVSigLipModule(vision_config, llavaov_mm_config, torch.cuda.current_device()),
                                           "internvit": LLaVAOVInternVitModule(vision_config, llavaov_mm_config, torch.cuda.current_device())},
                              "internvl" : {"internvit": LLaVAOVInternVitModule(vision_config, llavaov_mm_config, torch.cuda.current_device())}
                             # "qwenvl" : {"qwenvit": QWENVLQWENVitModule(vision_config, qwenvl_mm_config, torch.cuda.current_device()).to(model_dtype)}
                             }
            vision_module = vision_modules[args.mllm_model_name][args.vision_model_name].to(model_dtype)
            if args.vision_model_name == "siglip":
                image_size = 384
            elif args.vision_model_name == "internvit":
                image_size = 448
            elif args.vision_model_name == "qwenvit":
                pass
            profile_result = profile_vision_memory(image_size, vision_module, model_dtype, vision_tp_plan)
        else:
            df_path = os.path.join(result_path, f"mem_{args.llm_model_name}_{args.llm_model_size}.csv")
            llm_config = llm_configs[args.llm_model_name][args.llm_model_size]
            llm_models = {"qwen2": flashqwen2, "llama3": flashllama3}
            vocab_sizes = {"qwen2": 152064, "llama3": 128256}
            vocab_size = vocab_sizes[args.llm_model_name]
            llm_cls = llm_models[args.llm_model_name]
            llm_module = llm_cls(
                vocab_size=vocab_size,
                num_layers=args.num_hidden_layers,
                num_heads=llm_config["num_heads"],
                num_kv_heads=llm_config["num_kv_heads"],
                embed_dim=llm_config["embed_dim"],
                intermediate_dim=llm_config["intermediate_dim"],
                max_seq_len=327680,
                attn_dropout=0.0,
                norm_eps=1e-06,
                rope_base=1000000.0,
            )

    result_df = pd.DataFrame(profile_result)
    result_df["num_layers"] = args.num_hidden_layers
    result_df["tp_size"] = int(os.environ.get("WORLD_SIZE", 1))
    if global_rank == 0:
        if os.path.exists(df_path):
            prev_df = pd.read_csv(df_path)
            result_df = pd.concat([prev_df, result_df], ignore_index=True)
        result_df.to_csv(df_path, index=False)


# torchrun --nproc-per-node=8 /home/hyeonjun/dmllm_codes/dmllm_utils/dmllm_profiler.py --mllm_model_name llavaov --vision_model_name siglip --vision_model_size 6b --num_hidden_layers 4 --profile_mode mem