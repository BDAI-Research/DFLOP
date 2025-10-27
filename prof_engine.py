import copy
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from utils.data import (
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    SigLipImageProcessor,
    TrainConfig,
)
from utils.model import (
    InternVLInternVitModule,
    LLaVAOVInternVitModule,
    LLaVAOVMMConfig,
    LLaVAOVSigLipModule,
    llm_configs,
    vision_configs,
)
from torch import nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    loss_parallel
)
from torchtune.modules.loss import LinearCrossEntropyLoss
from torchtune.utils import batch_to_device
from torchtune_models import flashqwen2, flashllama3
from utils.parallel import (
    apply_ac,
    init_distributed,
    internvl_internvit_tp_plan,
    llava_ov_siglip_tp_plan,
    llavaov_internvit_tp_plan,
    llm_tp_plan,
    prepare_mha_for_tp,
)
from utils.profile import flush_cache, torchtune_loader
from utils.profiler import AllocatedMemContext
from torch.nn.parallel import DistributedDataParallel
from utils.config import (
    get_config_path,
    load_config,
    resolve_path as resolve_config_path,
    reset_config_cache,
)

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

def init_model(model, model_dtype, tp_plan, local_rank, profile_mode, is_vision):
    '''
    Profiling model initialization (especially, peak memory allocation during tensor parallelism)
    '''
    _world_size = int(os.environ["WORLD_SIZE"])
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(1, _world_size), mesh_dim_names=("dp", "tp"))
    dp_mesh = device_mesh["dp"]
    tp_mesh = device_mesh["tp"]
    model = model.to(model_dtype)
    model = model.to(local_rank)
    if _world_size > 1:
        # model = parallelize_model(model, _world_size)
        model = prepare_mha_for_tp(model, tp_mesh)
        parallelize_module(
            model,
            device_mesh=tp_mesh,
            parallelize_plan=tp_plan)
        if is_vision:
            apply_ac(model.vision_module.encoder)
        else:
            apply_ac(model)
        model = model.to(local_rank)
    if profile_mode == "mem":
        model = model.to(local_rank)
        model = DistributedDataParallel(model, device_ids=[local_rank], device_mesh=dp_mesh)
    return model

def profile_model_mem(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch):
    '''
    Profiling model memory usage during training including forward, backward and optimizer step
    '''
    if isinstance(batch, dict):
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
                if isinstance(batch, dict):
                    output = model(**batch)
                else:
                    output = model(batch)
        else:
            if isinstance(batch, dict):
                output = model(**batch)
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

def profile_model_thr(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch_origin, l2_cache_size, num_training_steps=30):
    '''
    Profiling model memory usage during training including forward, backward and optimizer step
    '''
    model_start_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
    model_end_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
    model_start_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
    model_end_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
    time.sleep(5)
    for i in range(5):
        batch = copy.deepcopy(batch_origin)
        if isinstance(batch, dict):
            batch_to_device(batch, torch.cuda.current_device())
            labels = batch.pop("labels")
        else:
            batch = batch.to("cuda")
            labels = None
        # Forward calculation
        if model_dtype == torch.float32:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                if isinstance(batch, dict):
                    output = model(**batch)
                else:
                    output = model(batch)
        else:
            if isinstance(batch, dict):
                output = model(**batch)
            else:
                output = model(batch)
        if labels is not None:
            with loss_parallel():
                loss = loss_fn(output, labels)
        else:
            loss = output.sum()
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

    for i in range(num_training_steps):
        _ = flush_cache(l2_cache_size)
        torch.cuda._sleep(1_000_000)
        batch = copy.deepcopy(batch_origin)
        if isinstance(batch, dict):
            batch_to_device(batch, torch.cuda.current_device())
            labels = batch.pop("labels")
        else:
            batch = batch.to("cuda")
            labels = None
        # Forward calculation
        model_start_fwd[i].record()
        if model_dtype == torch.float32:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                if isinstance(batch, dict):
                    output = model(**batch)
                else:
                    output = model(batch)
        else:
            if isinstance(batch, dict):
                output = model(**batch)
            else:
                output = model(batch)
        if labels is not None:
            with loss_parallel():
                loss = loss_fn(output, labels)
        else:
            loss = output.sum()
        model_end_fwd[i].record()
        model_start_bwd[i].record()
        # Backward calculation
        if model_dtype == torch.float32:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        model_end_bwd[i].record()
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
    torch.cuda.synchronize()
    fwd_times = [model_start_fwd[i].elapsed_time(model_end_fwd[i]) for i in range(num_training_steps)]
    bwd_times = [model_start_bwd[i].elapsed_time(model_end_bwd[i]) for i in range(num_training_steps)]
    fwd_times = fwd_times[len(fwd_times)//2:]
    bwd_times = bwd_times[len(bwd_times)//2:]
    avg_fwd_time = np.mean(fwd_times)
    avg_bwd_time = np.mean(bwd_times)
    return avg_fwd_time, avg_bwd_time

def profile_vision(image_size, vision_module, model_dtype, tp_plan, profile_mode, l2_cache_size):
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
    model = init_model(model, model_dtype, tp_plan, local_rank, profile_mode, is_vision=True)
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
    loss_fn = None
    if profile_mode == "mem":
        batch_size = 2
        batch = torch.randn(batch_size, 3, image_size, image_size).to(model_dtype)
        result_dict = {"batch_size" : [], "model_mem": [], "fwd_peak": [], "model_state_peak": []}
        fwd_peak, model_mem, model_state_peak = profile_model_mem(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch)
        result_dict["batch_size"].append(batch_size)
        result_dict["model_mem"].append(model_mem)
        result_dict["fwd_peak"].append(fwd_peak)
        result_dict["model_state_peak"].append(model_state_peak)
    elif profile_mode == "thr":
        num_training_steps = 30
        batch_size_list = [1, 2, 4, 8, 16, 32, 64]
        result_dict = {"batch_size": [], "model_time": [], "fwd_times": [], "bwd_times": []}
        try:
            for batch_size in batch_size_list:
                if global_rank == 0:
                    print(f"====== Profiling {batch_size} ======")
                batch = torch.randn(batch_size, 3, image_size, image_size).to(model_dtype)
                fwd_times, bwd_times = profile_model_thr(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch, l2_cache_size, num_training_steps)
                model_time = fwd_times + bwd_times
                result_dict["batch_size"].append(batch_size)
                result_dict["model_time"].append(model_time)
                result_dict["fwd_times"].append(fwd_times)
                result_dict["bwd_times"].append(bwd_times)
        except RuntimeError as e:
            print(f"RuntimeError at batch_size {batch_size} : {e}")
            return result_dict
    return result_dict

def profile_llm(llm_module, model_dtype, tp_plan, vocab_size, profile_mode, l2_cache_size):
    local_rank = int(os.environ["LOCAL_RANK"])
    loss_fn = LinearCrossEntropyLoss()
    loss_fn.set_model_output(llm_module)
    model = init_model(llm_module, model_dtype, tp_plan, local_rank, profile_mode, is_vision=False)
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
    if profile_mode == "mem":
        result_dict = {"batch_size" : [], "seq_len" : [], "model_mem": [], "fwd_peak": [], "model_state_peak": []}
        num_training_steps = 1
        batch_size = 1
        seq_len = 1024
        data_iter = torchtune_loader(num_training_steps, batch_size, seq_len, vocab_size)
        batch = next(data_iter)
        fwd_peak, model_mem, model_state_peak = profile_model_mem(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch)
        result_dict["batch_size"].append(batch_size)
        result_dict["seq_len"].append(seq_len)
        result_dict["model_mem"].append(model_mem)
        result_dict["fwd_peak"].append(fwd_peak)
        result_dict["model_state_peak"].append(model_state_peak)
    elif profile_mode == "thr":
        num_training_steps = 30
        sequence_length_list = [1024, 2048, 4096, 8192]
        result_dict = {"seq_len": [], "model_time": [], "fwd_times": [], "bwd_times": []}
        try:
            for seq_len in sequence_length_list:
                if global_rank == 0:
                    print(f"====== Profiling {seq_len} ======")
                data_iter = torchtune_loader(1, 1, seq_len, vocab_size)
                batch = next(data_iter)
                fwd_times, bwd_times = profile_model_thr(model_dtype, model, loss_fn, d_tensor_optimizer, local_tensor_optimizer, scaler, batch, l2_cache_size, num_training_steps)
                model_time = fwd_times + bwd_times
                result_dict["seq_len"].append(seq_len)
                result_dict["model_time"].append(model_time)
                result_dict["fwd_times"].append(fwd_times)
                result_dict["bwd_times"].append(bwd_times)
        except RuntimeError as e:
            print(f"RuntimeError at batch_size {seq_len} : {e}")
            return result_dict
    else:
        raise ValueError(f"Unsupported profile mode: {profile_mode}")
    return result_dict

def model_profiler(args, global_rank, result_path: Path, model_dtype, profile_mode, skip_attn, l2_cache_size):
    if args.vision_model_name:
        df_path = result_path / f"{profile_mode}_{args.mllm_model_name}_{args.vision_model_name}_{args.vision_model_size}.csv"
        llm_config = llm_configs["qwen2.5"]["7b"] # arbitrary llm config for vision model profiling
        vision_config = vision_configs[args.vision_model_name][args.vision_model_size]
        vision_config.num_hidden_layers = args.num_hidden_layers
        vision_tp_plans = {"llavaov": {"siglip": llava_ov_siglip_tp_plan, "internvit": llavaov_internvit_tp_plan},
                        "internvl": {"internvit": internvl_internvit_tp_plan}
                        #    "qwenvl":{"qwenvit": qwenvl_tp_plan}
                        }
        vision_tp_plan = vision_tp_plans[args.mllm_model_name][args.vision_model_name]
        llavaov_mm_config = LLaVAOVMMConfig("mlp2x_gelu", vision_config.hidden_size, llm_config["embed_dim"])
        if args.mllm_model_name == "llavaov":
            vision_modules = {"siglip": LLaVAOVSigLipModule,
                                "internvit": LLaVAOVInternVitModule}
            vision_cls = vision_modules[args.vision_model_name]
            vision_module = vision_cls(vision_config, llavaov_mm_config, model_dtype, torch.cuda.current_device())
        elif args.mllm_model_name == "internvl":
            vision_module = InternVLInternVitModule(vision_config, llm_config, model_dtype, torch.cuda.current_device()).to(torch.cuda.current_device())
        else:
            pass
        # vision_modules = {"llavaov" : {"siglip": LLaVAOVSigLipModule(vision_config, llavaov_mm_config, model_dtype, torch.cuda.current_device()),
        #                                "internvit": LLaVAOVInternVitModule(vision_config, llavaov_mm_config, model_dtype, torch.cuda.current_device())},
        #                   "internvl" : {"internvit": InternVLInternVitModule(vision_config, llm_config, model_dtype, torch.cuda.current_device())}
        #                  # "qwenvl" : {"qwenvit": QWENVLQWENVitModule(vision_config, qwenvl_mm_config, torch.cuda.current_device()).to(model_dtype)}
        #                  }
        # vision_module = vision_modules[args.mllm_model_name][args.vision_model_name].to(model_dtype)
        if args.vision_model_name == "siglip":
            image_size = 384
        elif args.vision_model_name == "internvit":
            image_size = 448
        elif args.vision_model_name == "qwenvit":
            pass
        profile_result = profile_vision(image_size, vision_module, model_dtype, vision_tp_plan, profile_mode, l2_cache_size)
    else:
        if skip_attn:
            df_path = result_path / f"{profile_mode}_{args.mllm_model_name}_{args.llm_model_name}_{args.llm_model_size}_skip_attn.csv"
        else:
            df_path = result_path / f"{profile_mode}_{args.mllm_model_name}_{args.llm_model_name}_{args.llm_model_size}.csv"
        
        llm_config = llm_configs[args.llm_model_name][args.llm_model_size]
        llm_models = {"qwen2.5": flashqwen2, "llama3": flashllama3}
        vocab_sizes = {"qwen2.5": 152064, "llama3": 128256}
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
            skip_attn=skip_attn
        )
        llm_module = llm_module
        profile_result = profile_llm(llm_module, model_dtype, llm_tp_plan, vocab_size, profile_mode, l2_cache_size)
        
    result_df = pd.DataFrame(profile_result)
    result_df["num_layers"] = args.num_hidden_layers
    result_df["tp_size"] = int(os.environ.get("WORLD_SIZE", 1))
    if global_rank == 0:
        if df_path.exists():
            prev_df = pd.read_csv(df_path)
            result_df = pd.concat([prev_df, result_df], ignore_index=True)
        df_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(df_path, index=False)

if __name__ == "__main__":
    default_config_path = get_config_path()
    os.environ.setdefault("DFLOP_CONFIG", str(default_config_path))
    reset_config_cache()
    config = load_config()

    profiling_cfg = config.get("profiling", {})
    models_cfg = config.get("models", {})
    vision_cfg = models_cfg.get("vision", {})
    llm_cfg = models_cfg.get("llm", {})
    paths_cfg = config.get("paths", {})
    model_paths_cfg = config.get("model_paths", {})

    mllm_model_name = models_cfg.get("mllm")
    vision_model_name = vision_cfg.get("name")
    vision_model_size = vision_cfg.get("size")
    llm_model_name = llm_cfg.get("name")
    llm_model_size = llm_cfg.get("size")

    profile_mode = os.environ.get("PROFILE_MODE")
    if profile_mode is None:
        raise ValueError("PROFILE_MODE environment variable must be set (expected 'data', 'mem', or 'thr').")
    if profile_mode not in {"data", "mem", "thr"}:
        raise ValueError(f"Unsupported profile mode: {profile_mode}")

    num_hidden_layers_env = os.environ.get("NUM_HIDDEN_LAYERS")
    num_hidden_layers = int(num_hidden_layers_env) if num_hidden_layers_env is not None else None
    if profile_mode in {"mem", "thr"} and num_hidden_layers is None:
        raise ValueError("NUM_HIDDEN_LAYERS environment variable must be set for mem/thr profiling runs.")

    skip_attn = os.environ.get("SKIP_ATTN", "").lower() in {"1", "true", "yes"}
    device_type = os.environ.get("DEVICE_TYPE", "a100")

    settings = SimpleNamespace(
        mllm_model_name=mllm_model_name,
        vision_model_name=vision_model_name,
        vision_model_size=vision_model_size,
        llm_model_name=llm_model_name,
        llm_model_size=llm_model_size,
        model_dtype=profiling_cfg.get("model_dtype", "bfloat16"),
        num_hidden_layers=num_hidden_layers,
        profile_mode=profile_mode,
        device_type=device_type,
        skip_attn=skip_attn,
    )

    required_fields = [
        "mllm_model_name",
        "vision_model_name",
        "vision_model_size",
        "llm_model_name",
        "llm_model_size",
    ]
    missing = [field for field in required_fields if getattr(settings, field) is None]
    if missing:
        raise ValueError(f"Missing profiling configuration values: {', '.join(missing)}")

    args = settings

    result_path = resolve_config_path(paths_cfg.get("profile_result_dir"))
    if result_path is None:
        raise ValueError("paths.profile_result_dir must be provided in the configuration.")
    dataset_path = resolve_config_path(paths_cfg.get("dataset_yaml"))
    image_folder = resolve_config_path(paths_cfg.get("image_folder"))
    video_folder = resolve_config_path(paths_cfg.get("video_folder"))

    for name, path in (
        ("paths.dataset_yaml", dataset_path),
        ("paths.image_folder", image_folder),
        ("paths.video_folder", video_folder),
    ):
        if path is None:
            raise ValueError(f"{name} must be provided in the configuration.")

    result_path.mkdir(parents=True, exist_ok=True)

    model_dtype = getattr(torch, args.model_dtype)

    model_path = resolve_config_path(model_paths_cfg.get(args.llm_model_name))
    if model_path is None:
        raise ValueError(f"Missing model path for '{args.llm_model_name}' in model_paths configuration.")

    dataset_path_str = str(dataset_path)
    image_folder_str = str(image_folder)
    video_folder_str = str(video_folder)
    model_path_str = str(model_path)

    if args.profile_mode == "data":
        # Data distribution analysis setup
        np.random.seed(42)
        seed_list = np.random.choice(range(10000), size=1, replace=False)
        seed = int(seed_list[0])
        generator = torch.Generator()
        generator.manual_seed(seed)
        if args.mllm_model_name == "qwenvl":
            pass
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_str, padding_side="right")
            tokenizer_model_max_length = tokenizer.model_max_length
            if args.vision_model_name == "siglip":
                image_size = 384
                patch_size = 14
                image_processor = SigLipImageProcessor()
            elif args.vision_model_name == "internvit":
                image_size = 448
                patch_size = 14
                image_processor = SigLipImageProcessor(size=(image_size, image_size), crop_size={"height": image_size, "width": image_size})
            train_config = TrainConfig(image_folder=image_folder_str,
                            video_folder=video_folder_str,
                            image_processor=image_processor,
                            tokenizer_model_max_length=tokenizer_model_max_length,
                            image_size=image_size,
                            patch_size=patch_size,
                            is_internvl=True if args.mllm_model_name=="internvl" else False
                        )
            llm_config = llm_configs[args.llm_model_name][args.llm_model_size]  # arbitrary llm config for data analysis
            dataset = LazySupervisedDataset(dataset_path_str, tokenizer, train_config)
            data_collator = DataCollatorForSupervisedDataset(train_config, llm_config, tokenizer, model_dtype)
        sampler = RandomSampler(dataset, generator=generator)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=data_collator, shuffle=False, num_workers=32)
        image_batch_list, llm_input_seq_list = run_data_analysis(dataloader)
        data_analysis_result = {"image_batch": image_batch_list, "llm_input_seq": llm_input_seq_list}
        with open(result_path / f"{args.mllm_model_name}_{args.vision_model_name}_{args.llm_model_name}_video.pkl", "wb") as f:
            pickle.dump(data_analysis_result, f)
    else:
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if "a100" in args.device_type:
            l2_cache_size = int(40 * 1024**2)
        elif "6000ada" in args.device_type:
            l2_cache_size = int(96 * 1024**2)
        elif "a6000" in args.device_type:
            l2_cache_size = int(6 * 1024**2)
        else:
            raise ValueError(f"Unknown device_type: {args.device_type}")
        model_profiler(args, global_rank, result_path, model_dtype, args.profile_mode, args.skip_attn, l2_cache_size)
    # os._exit(1)
