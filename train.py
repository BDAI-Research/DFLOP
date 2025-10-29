import os
import queue
import subprocess
import yaml
import json
import pickle
import torch
import time
import numpy as np
import pandas as pd
import transformers
import torch.distributed as dist
from copy import deepcopy
from torch.distributed._composable.replicate import replicate
from torchtune.modules.loss import LinearCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.distributed.tensor.parallel import (
    parallelize_module,
    loss_parallel
)
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.device_mesh import DeviceMesh
from dflop.torchtune_models import flashqwen2, flashllama3
from dflop.parallel import (
    apply_ac,
    init_distributed,
    llava_ov_siglip_tp_plan,
    llavaov_internvit_tp_plan,
    llm_tp_plan,
    prepare_mha_for_tp,
    set_topology_config,
    setup_multinode_distributed_groups,
)
from dflop.pipeline.dflop_pp_schedule_module import Schedule1F1B
from dflop.pipeline.dflop_pp_stage_module import PipelineStage
from dflop.model import (
    DflopPipeStage,
    LLaVAOVInternVitModule,
    LLaVAOVMMConfig,
    LLaVAOVSigLipModule,
    VisionPipeStage,
    llm_configs,
    vision_configs,
)
from dflop.data import (
    LazySupervisedDataset,
    SigLipImageProcessor,
    TrainConfig,
)
from dflop.loader import DataCollator, IndexProducer, QueueBatchSampler
from dflop.config import (
    get_config_path,
    load_config,
    resolve_path,
    reset_config_cache,
)

def get_llm_idx_list(llm_dp_size, idx_dict, global_batch_step, micro_batch_step):
    cur_llm_m_batch = idx_dict[str(global_batch_step)]['llm']
    cur_llm_m_batch_len = [len(cur_llm_m_batch[str(llm_dp_rank)][micro_batch_step]) for llm_dp_rank in range(llm_dp_size)]
    cur_llm_m_batch_idx = []
    start_idx = 0
    for llm_len in cur_llm_m_batch_len:
        cur_llm_m_batch_idx.append(list(range(start_idx, start_idx + llm_len)))
        start_idx += llm_len

    return cur_llm_m_batch_idx

if __name__ == "__main__":
    default_config_path = get_config_path()
    os.environ.setdefault("DFLOP_CONFIG", str(default_config_path))
    reset_config_cache()
    config = load_config()

    paths_cfg = config.get("paths", {})
    model_paths_cfg = config.get("model_paths", {})

    dataset_path = resolve_path(paths_cfg.get("dataset_yaml"))
    image_folder = resolve_path(paths_cfg.get("image_folder"))
    video_folder = resolve_path(paths_cfg.get("video_folder"))
    data_sched_config_path = resolve_path(paths_cfg.get("data_scheduler_config"))
    idx_path = resolve_path(paths_cfg.get("data_index_file"))
    result_path = resolve_path(paths_cfg.get("train_result_dir"))

    required_paths = {
        "paths.dataset_yaml": dataset_path,
        "paths.image_folder": image_folder,
        "paths.video_folder": video_folder,
        "paths.data_scheduler_config": data_sched_config_path,
        "paths.data_index_file": idx_path,
        "paths.train_result_dir": result_path,
    }
    missing_paths = [key for key, value in required_paths.items() if value is None]
    if missing_paths:
        raise ValueError(f"Missing configuration paths: {', '.join(missing_paths)}")

    dataset_path_str = str(dataset_path)
    image_folder_str = str(image_folder)
    video_folder_str = str(video_folder)
    data_sched_config_path_str = str(data_sched_config_path)
    idx_path_str = str(idx_path)
    result_path_str = str(result_path)

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    timeout = 15

    _, global_rank, world_size = init_distributed(timeout)
    torch.cuda.set_device(local_rank)
    if global_rank == 0:
        if os.path.exists(idx_path_str):
            os.remove(idx_path_str)
        command = ['python', './scheduler.py']
        scheduler_env = os.environ.copy()
        scheduler_env["DFLOP_CONFIG"] = os.environ["DFLOP_CONFIG"]
        process = subprocess.Popen(command, env=scheduler_env)
        print(f"Data Scheduling  is running background. (PID: {process.pid})")

    with open(data_sched_config_path_str, "r") as f:
        data_sched_config = yaml.safe_load(f)
    parallel_config = data_sched_config["parallel_config"]

    # Configurations
    num_training_step = data_sched_config["num_training_steps"]
    num_micro_batches = data_sched_config["num_batches"]
    vision_size = data_sched_config["vision_model_size"]
    llm_size = data_sched_config["llm_model_size"]
    vision_model_name = data_sched_config["vision_model_name"]
    llm_model_name = data_sched_config["llm_model_name"]

    model_path_resolved = resolve_path(model_paths_cfg.get(llm_model_name))
    if model_path_resolved is None:
        raise ValueError(f"Missing model path for '{llm_model_name}' in model_paths configuration.")
    model_path = str(model_path_resolved)

    vision_tp_plans = {"siglip" : llava_ov_siglip_tp_plan, "internvit": llavaov_internvit_tp_plan}
    vision_tp_plan = vision_tp_plans[vision_model_name]
    vision_config = vision_configs[vision_model_name][vision_size]
    llm_config = llm_configs[llm_model_name][llm_size]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer_model_max_length = tokenizer.model_max_length
    mm_config = LLaVAOVMMConfig("mlp2x_gelu", vision_config.hidden_size, llm_config["embed_dim"])
    vision_num_layers = vision_config.num_hidden_layers
    llm_num_layers = llm_config["num_layers"]
    vision_hidden_size = vision_config.hidden_size
    llm_hidden_size = llm_config["embed_dim"]    
    model_dtype = torch.bfloat16

    vision_pp_size = parallel_config["vision_pp_size"]
    vision_dp_size = parallel_config["vision_dp_size"]
    vision_tp_size = parallel_config["vision_tp_size"]
    llm_pp_size = parallel_config["llm_pp_size"]
    llm_dp_size = parallel_config["llm_dp_size"]
    llm_tp_size = parallel_config["llm_tp_size"]
    vision_ranks = vision_pp_size * vision_dp_size * vision_tp_size
    llm_ranks = llm_pp_size * llm_dp_size * llm_tp_size

    topology_config = set_topology_config(world_size, vision_ranks, vision_pp_size, vision_dp_size, vision_tp_size,
                                          llm_pp_size, llm_dp_size, llm_tp_size)
    topology = setup_multinode_distributed_groups(topology_config)
    tp_group = topology["tp_group"]
    dp_group = topology["dp_group"]
    pp_group = topology["pp_group"]
    cur_stage_id = torch.distributed.get_rank(pp_group)
    stage_id = cur_stage_id + topology["offset"]
    device_mesh = DeviceMesh.from_group(group=[pp_group, dp_group, tp_group], mesh=topology["module_ranks"].tolist(), device_type="cuda", mesh_dim_names=("pp", "dp", "tp"))
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]
    print(f"[Rank : {global_rank}] TP Mesh : {tp_mesh} DP Mesh : {dp_mesh}")
    if vision_model_name == "siglip":
        vision_module = LLaVAOVSigLipModule(vision_config, mm_config, model_dtype, device)
        image_size = 384
        patch_size = 14
        image_processor = SigLipImageProcessor()
    elif vision_model_name == "internvit":
        vision_module = LLaVAOVInternVitModule(vision_config, mm_config, model_dtype, device)
        image_size = 448
        patch_size = 14
        image_processor = SigLipImageProcessor(size=(image_size, image_size), crop_size={"height": image_size, "width": image_size})
    vision_stages = list(range(vision_pp_size))
    llm_stages = list(range(vision_pp_size, vision_pp_size + llm_pp_size))
    pp_num_stages = sum([pp_size for pp_size in topology["pp_size"].values()])
    if stage_id in vision_stages:
        layers_per_rank = 1
    else:
        layers_per_stage = llm_num_layers // len(llm_stages)
        remain_layers = llm_num_layers % len(llm_stages)
        stage_layer_counts = [layers_per_stage + (1 if i < remain_layers else 0) for i in range(len(llm_stages))]
        stage_idx = llm_stages.index(stage_id)
        layers_per_rank = stage_layer_counts[stage_idx]
    if llm_model_name == "qwen2.5":
        llm_model = flashqwen2(
                        vocab_size=152064,
                        num_layers=layers_per_rank,
                        num_heads=llm_config["num_heads"],
                        num_kv_heads=llm_config["num_kv_heads"],
                        embed_dim=llm_config["embed_dim"],
                        intermediate_dim=llm_config["intermediate_dim"],
                        max_seq_len=32768,
                        attn_dropout=0.0,
                        norm_eps=1e-06,
                        rope_base=1000000.0,
                    )
        vocab_size = 152064
    elif llm_model_name == "llama3":
        llm_model = flashllama3(
                vocab_size=128256,
                num_layers=layers_per_rank,
                num_heads=llm_config["num_heads"],
                num_kv_heads=llm_config["num_kv_heads"],
                embed_dim=llm_config["embed_dim"],
                intermediate_dim=llm_config["intermediate_dim"],
                max_seq_len=32768,
                attn_dropout=0.0,
                norm_eps=1e-06,
                rope_base=1000000.0,
            )
        vocab_size = 128256
    vision_module = vision_module.to(model_dtype)
    llm_model = llm_model.to(model_dtype)
    train_config = TrainConfig(image_folder=image_folder_str,
                video_folder=video_folder_str,
                image_processor=image_processor,
                tokenizer_model_max_length=tokenizer_model_max_length,
                image_size=image_size,
                patch_size=patch_size,
            )
    cur_module = topology["component"]
    v_pp_last_group_ranks = topology["v_pp_last_group_ranks"]
    l_pp_first_group_ranks = topology["l_pp_first_group_ranks"]
    v_tp_first_group_ranks = topology["v_tp_first_group_ranks"]
    l_tp_first_group_ranks = topology["l_tp_first_group_ranks"]
    if stage_id in vision_stages:
        vision_pipe_stage = VisionPipeStage(vision_module, vision_stages, stage_id, device, model_dtype)
    else:
        vision_pipe_stage = None
    stage_model = DflopPipeStage(llm_model, stage_id, vision_stages, llm_stages, vision_pipe_stage, device, train_config, llm_config, model_dtype)
    stage_model = stage_model.to(model_dtype)
    if stage_id < vision_pp_size:
        stage_model = prepare_mha_for_tp(stage_model, tp_mesh) 
        stage_model = parallelize_module(
                    stage_model,
                    device_mesh=tp_mesh,
                    parallelize_plan=vision_tp_plan,
                )
        apply_ac(stage_model.vision_module.encoder)
        
    else:
        stage_model = prepare_mha_for_tp(stage_model, tp_mesh) 
        stage_model= parallelize_module(
                        stage_model,
                        device_mesh=tp_mesh,
                        parallelize_plan=llm_tp_plan)
        apply_ac(stage_model)
    stage_model = stage_model.to(device)
    stage_model = DistributedDataParallel(stage_model, device_ids=[local_rank], device_mesh=dp_mesh)
    stage_model.train()
    print(f"[Rank : {global_rank}] stage_model parallelized {stage_model}")
    dist.barrier()
    stage = PipelineStage(
        stage_model,
        cur_stage_id,
        pp_num_stages,
        device,
        group=pp_group,
        v_pp_last_group_ranks=v_pp_last_group_ranks,
        llm_first_group_ranks=l_pp_first_group_ranks,
        v_tp_first_group_ranks=v_tp_first_group_ranks,
        l_tp_first_group_ranks=l_tp_first_group_ranks,
        l_tp_size=llm_tp_size,
        is_vision=True if cur_module == "vision" else False,
        is_llm=True if cur_module == "llm" else False,
    )
    d_tensor_param = [params for params in stage_model.parameters() if isinstance(params, torch.distributed.tensor.DTensor)]
    local_tensor_param = [params for params in stage_model.parameters() if not isinstance(params, torch.distributed.tensor.DTensor)]
    num_d_tensor_params = sum(params.numel() for params in d_tensor_param)
    num_local_tensor_params = sum(params.numel() for params in local_tensor_param)
    after_model_init = torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024**3
    if len(d_tensor_param) != 0:
        d_tensor_optimizer = torch.optim.AdamW(d_tensor_param, lr=1e-5, weight_decay=0.01)
    else:
        d_tensor_optimizer = None
    if len(local_tensor_param) != 0:
        local_tensor_optimizer = torch.optim.AdamW(local_tensor_param, lr=1e-5, weight_decay=0.01)
    else:
        local_tensor_optimizer = None
    if stage_id == pp_num_stages - 1:
        schedule = Schedule1F1B(stage, num_micro_batches, loss_fn=stage_model.module.loss_fn)
    else:
        schedule = Schedule1F1B(stage, num_micro_batches, loss_fn=LinearCrossEntropyLoss())
    # Setup DataLoader
    dp_rank = torch.distributed.get_rank(dp_group)
    if stage_id in vision_stages:
        module = "vision"
        index_queue = queue.Queue(maxsize=5) 
        producer_thread = IndexProducer(idx_path_str, index_queue, num_training_step, module, dp_rank)
        producer_thread.start()

        data_collator = DataCollator(train_config, llm_config, tokenizer, model_dtype, module, vision_dp_size, dp_rank)
    else:
        module = "llm"
        index_queue = queue.Queue(maxsize=5) 
        producer_thread = IndexProducer(idx_path_str, index_queue, num_training_step, module, dp_rank)
        producer_thread.start()

        data_collator = DataCollator(train_config, llm_config, tokenizer, model_dtype, module)

    dataset = LazySupervisedDataset(dataset_path_str, tokenizer, train_config)
    sampler = QueueBatchSampler(index_queue)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,
        num_workers=2,
    )
    image_feature_dim = train_config.num_patches_per_side ** 2
    torch.distributed.barrier()

    # Start Training
    input_list = []
    input_meta = []
    output_meta = []
    target_list = []
    kwargs_list = []
    for step, batch in enumerate(dataloader):
        global_step = step // num_micro_batches
        micro_step = step % num_micro_batches
        input_args = ()
        if stage_id in vision_stages:
            images = batch.pop("images")
            if stage_id == vision_stages[-1]:
                inp_meta = [(len(images), image_feature_dim, vision_hidden_size), model_dtype]
            if stage_id == vision_stages[0]:
                input_args = (images, )
                inp_meta = [tuple(images.shape), model_dtype]

                out_meta = [(len(images), image_feature_dim, vision_hidden_size), model_dtype]
                kwargs = {
                }
            if stage_id == vision_stages[-1]:
                split_size_list = []
                try:
                    with open(idx_path_str, "r") as f:
                        idx_dict = json.load(f)
                except:
                    time.sleep(5)
                    with open(idx_path_str, "r") as f:
                        idx_dict = json.load(f)
                data_idx = get_llm_idx_list(llm_dp_size, idx_dict, global_step, micro_step)
                for idx_group in data_idx:
                    split_size_list.append([batch["split_sizes"][i] for i in idx_group])
                out_meta = []
                for split_sz in split_size_list:
                    o_meta = [(sum(split_sz), image_feature_dim, llm_config["embed_dim"]), model_dtype]
                    out_meta.append(o_meta)
                kwargs = {
                    'data_idx' : data_idx,
                    'split_sizes': batch["split_sizes"]
                }
        else:
            if stage_id == llm_stages[0]:
                split_sizes = batch["split_sizes"]
                inp_meta = [(sum(split_sizes), image_feature_dim, llm_config["embed_dim"]), model_dtype]
                kwargs = {
                    'split_sizes': split_sizes,
                    'video_idx_in_batch': batch["video_idx_in_batch"],
                    'image_sizes': batch["image_sizes"],
                    'input_ids': batch["input_ids"],
                    'labels': batch["labels"],
                    'text_attention_mask': batch["attention_mask"],
                }
            else:
                inp_meta = [(*batch["new_position_ids"].shape, llm_config["embed_dim"]), model_dtype]
                kwargs = {
                    'input_pos': batch["new_position_ids"],
                }
            out_meta = [(*batch["new_position_ids"].shape, llm_config["embed_dim"]), model_dtype]
            zero_idx = (batch["new_position_ids"] == 0).nonzero(as_tuple=True)[1]
        input_list.append(input_args)
        input_meta.append([inp_meta])
        output_meta.append([out_meta])
        kwargs_list.append(kwargs)
        if stage_id == pp_num_stages - 1:
            new_labels = batch["new_labels"].to(device)
            target_list.append(new_labels)
        if (micro_step + 1) == num_micro_batches:
            with loss_parallel():
                if stage_id == pp_num_stages - 1:
                    output = schedule.step(*input_list, kwargs_list=kwargs_list, input_meta=input_meta, output_meta=output_meta, target=target_list)
                else:
                    output = schedule.step(*input_list, kwargs_list=kwargs_list, input_meta=input_meta, output_meta=output_meta)
            if d_tensor_optimizer is not None:
                d_tensor_optimizer.step()
                d_tensor_optimizer.zero_grad()
            if local_tensor_optimizer is not None:
                local_tensor_optimizer.step()
                local_tensor_optimizer.zero_grad()
            input_list.clear()
            input_meta.clear()
            output_meta.clear()
            target_list.clear()
            kwargs_list.clear()
        if global_step == num_training_step:
            break
    torch.cuda.synchronize()
    torch.distributed.barrier()
    dist.destroy_process_group()
