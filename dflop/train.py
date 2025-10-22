import os
import queue
import subprocess
import yaml
import json
import pickle
import argparse
import torch
import traceback
import time
import numpy as np
import pandas as pd
import transformers
import torch.distributed as dist
from datetime import timedelta
from copy import deepcopy
from torch.distributed._composable.replicate import replicate
from torchtune.modules.loss import LinearCrossEntropyLoss
from torchtitan.distributed import ParallelDims
from torch.utils.data import DataLoader, RandomSampler
from torchtune.modules.loss import LinearCrossEntropyLoss
from torch.distributed.tensor.parallel import (
    parallelize_module,
    loss_parallel
)
from torch.nn.parallel import DistributedDataParallel
from torchtune_models import flashqwen2, flashllama3
from ours_pp_schedule_module import Schedule1F1B
from ours_pp_stage_module import PipelineStage
from utils.utils import prepare_mha_for_tp
from dmllm_utils.utils import llava_ov_siglip_tp_plan, llavaov_internvit_tp_plan, llm_tp_plan, vision_configs, llm_configs, init_distributed, apply_ac
from dmllm_utils.ours_model_utils import VisionPipeStage, OursPipeStage
from dmllm_utils.model_utils import LLaVAOVMMConfig, LLaVAOVSigLipModule, LLaVAOVInternVitModule
from dmllm_utils.data_utils import TrainConfig, LazySupervisedDataset, DataCollatorForSupervisedDataset, SigLipImageProcessor
from dmllm_utils.vision_llm_topo_utils import set_topology_config, setup_multinode_distributed_groups
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.data import DataLoader
from dmllm_utils.data_utils import TrainConfig, LazySupervisedDataset, DataCollatorForSupervisedDataset, SigLipImageProcessor
from dmllm_utils.data_sched_utils import IndexProducer, QueueBatchSampler, DataCollator


if __name__ == "__main__":

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    timeout = 15
    with open(data_sched_config_path, "r") as f:
        data_sched_config = yaml.safe_load(f)
    parallel_config = data_sched_config["parallel_config"]

    # Configurations
    num_training_step = data_sched_config["num_training_steps"]
    num_micro_batches = data_sched_config["num_batches"]
    vision_size = data_sched_config["vision_model_size"]
    llm_size = data_sched_config["llm_model_size"]
    vision_model_name = data_sched_config["vision_model_name"]
    llm_model_name = data_sched_config["llm_model_name"]

    if llm_model_name == "qwen2":
        model_path = "/giant-data/team/4724/models/Qwen2.5-72B-Instruct"
    elif llm_model_name == "llama3":
        model_path = "/giant-data/team/4724/models/Meta-Llama-3-8B"

    vision_tp_plans = {"siglip" : llava_ov_siglip_tp_plan, "internvit": llavaov_internvit_tp_plan}
    vision_tp_plan = vision_tp_plans[vision_model_name]
    vision_config = vision_configs[vision_model_name][vision_size]
    llm_config = llm_configs[llm_model_name][llm_size]
    # llm_config["num_layers"] = 8
    # vision_config.num_hidden_layers = 8
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer_model_max_length = tokenizer.model_max_length
    mm_config = LLaVAOVMMConfig("mlp2x_gelu", vision_config.hidden_size, llm_config["embed_dim"])
    # llm_config["num_layers"] = 8
    # vision_config.num_hidden_layers = 8
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
    if llm_model_name == "qwen2":
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
    train_config = TrainConfig(image_folder=image_folder,
                video_folder=video_folder,
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
    stage_model = OursPipeStage(llm_model, stage_id, vision_stages, llm_stages, vision_pipe_stage, device, train_config, llm_config, model_dtype)
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
    # replicate(stage_model, device_mesh=dp_mesh, bucket_cap_mb=100)
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
        producer_thread = IndexProducer(idx_path, index_queue, num_training_step, module, dp_rank)
        producer_thread.start()

        data_collator = DataCollator(train_config, llm_config, tokenizer, model_dtype, module, vision_dp_size, dp_rank)
    else:
        module = "llm"
        index_queue = queue.Queue(maxsize=5) 
        producer_thread = IndexProducer(idx_path, index_queue, num_training_step, module, dp_rank)
        producer_thread.start()

        data_collator = DataCollator(train_config, llm_config, tokenizer, model_dtype, module)

    dataset = LazySupervisedDataset(dataset_path, tokenizer, train_config)
    sampler = QueueBatchSampler(index_queue)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,
        num_workers=2,
    )
    # Parameters to calculate flops
    ## Vision
    in_channels = 3
    img_h = image_size
    img_w = image_size
    patch_dim = patch_size
    hs = vision_hidden_size
    intermediate_size = vision_config.intermediate_size
    image_feature_dim = train_config.num_patches_per_side ** 2
    ## LLM
    num_kv_heads = llm_config["num_kv_heads"]
    attention_heads = llm_config["num_heads"]
    query_groups = attention_heads // num_kv_heads
    ffn_hidden_size = llm_config["intermediate_dim"]
    torch.distributed.barrier()
    
    flops_list = []
    start = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_step)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_step)]
    # Start Training
    input_list = []
    input_meta = []
    output_meta = []
    target_list = []
    kwargs_list = []
    image_batch_size_list = []
    seq_len_list = []
    time_list = []
    time_dict_list = []
    for step, batch in enumerate(dataloader):
        global_step = step // num_micro_batches
        micro_step = step % num_micro_batches
        # print(f"[Rank : {global_rank}, Global Step : {global_step}, Micro Step : {micro_step}] Batch keys : {batch.keys()} Batch length : {batch['input_ids'].shape[0] if module == 'llm' else len(batch['split_sizes'])}")
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
                    with open(idx_path, "r") as f:
                        idx_dict = json.load(f)
                except:
                    time.sleep(5)
                    with open(idx_path, "r") as f:
                        idx_dict = json.load(f)
                data_idx = get_llm_idx_list(llm_dp_size, idx_dict, global_step, micro_step)
                for idx_group in data_idx:
                    # print(f"[Rank : {global_rank}], idx_group : {idx_group}, split_sizes : {batch['split_sizes']}, idx split_sizes : {[batch['split_sizes'][i] for i in idx_group]}")
                    split_size_list.append([batch["split_sizes"][i] for i in idx_group])
                out_meta = []
                for split_sz in split_size_list:
                    o_meta = [(sum(split_sz), image_feature_dim, llm_config["embed_dim"]), model_dtype]
                    out_meta.append(o_meta)
                # print(f"[Rank : {torch.distributed.get_rank()}, Step : {_}] Images shape : {images.shape}, data_idx : {data_idx}, position_ids shape: {batch['new_position_ids'].shape}, split_sizes : {batch['split_sizes']}")
                kwargs = {
                    'data_idx' : data_idx,
                    'split_sizes': batch["split_sizes"]
                }
            image_batch_size_list.append(images.shape[0])
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
            seq_lens = torch.concat([zero_idx, torch.tensor([batch["new_position_ids"].shape[1]])], dim=0).diff().tolist()
            seq_len_list.append(seq_lens)
        # print(f"[Step : {global_step}, Global rank : {global_rank}] DP Rank {dp_rank}, sequence length : {batch['new_position_ids'].shape[1]}")
        input_list.append(input_args)
        input_meta.append([inp_meta])
        output_meta.append([out_meta])
        kwargs_list.append(kwargs)
        if stage_id == pp_num_stages - 1:
            new_labels = batch["new_labels"].to(device)
            target_list.append(new_labels)
        if (micro_step + 1) == num_micro_batches:
            # time_dict = {"fwd": [], "bwd": []}
            # before_fwd_mem_summary = torch.cuda.memory.memory_summary(device=local_rank, abbreviated=False)
            dist.barrier()
            # print(f"[Rank : {global_rank}] Step : {global_step}, first microstep output meta : {output_meta[0]}")
            start[global_step].record()
            with loss_parallel():
                if stage_id == pp_num_stages - 1:
                    output = schedule.step(*input_list, kwargs_list=kwargs_list, input_meta=input_meta, output_meta=output_meta, target=target_list)
                else:
                    output = schedule.step(*input_list, kwargs_list=kwargs_list, input_meta=input_meta, output_meta=output_meta)
            dist.barrier()
            end[global_step].record()
            torch.cuda.synchronize()
            duration = start[global_step].elapsed_time(end[global_step]) / 1000
            print(f"[Rank : {global_rank}, Step : {global_step}] micro batches: {duration:.2f} s")
            time_list.append(duration)
            time_dict = deepcopy(stage.time_dict)
            time_dict_list.append(time_dict)
            dist.barrier()
            # after_fwd_mem_summary = torch.cuda.memory.memory_summary(device=local_rank, abbreviated=False)
            # time_dict_list.append(time_dict)
            # current_mem = torch.cuda.memory_allocated() / 1024**3
            # peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            # if global_rank == 0:
            #     # print(f"[Rank : {global_rank}, Step : {global_step}] Current GPU memory : {current_mem:.2f} GB, Peak GPU memory : {peak_mem:.2f} GB")
            # dist.barrier()
            # dist.barrier()
            # time.sleep(local_rank)
            # print(f"=============== [Rank : {global_rank}, Step : {global_step}] Memory Summary ===============")
            # print(f"----- Before Fwd -----\n{before_fwd_mem_summary}\n----- After Fwd -----\n{after_fwd_mem_summary}")
            if d_tensor_optimizer is not None:
                d_tensor_optimizer.step()
                d_tensor_optimizer.zero_grad()
            if local_tensor_optimizer is not None:
                local_tensor_optimizer.step()
                local_tensor_optimizer.zero_grad()
            # after_opt_mem_summary = torch.cuda.memory.memory_summary(device=local_rank, abbreviated=False)
            # dist.barrier()
            # time.sleep(local_rank)
            # print(f"=============== [Rank : {global_rank}, Step : {global_step}] Memory Summary ===============")
            # print(f"----- Before Fwd -----\n{before_fwd_mem_summary}\n----- After Fwd -----\n{after_fwd_mem_summary}\n----- After Opt -----\n{after_opt_mem_summary}")
            vision_flops = 0
            llm_flops = 0
            if stage_id in vision_stages:
                for image_batch_size in image_batch_size_list:
                    vision_flops += (vision_module_flops(image_batch_size, in_channels, img_h, img_w, patch_dim, vision_num_layers, hs, intermediate_size, llm_hidden_size) / 1e12)
            else:
                for seq_len in seq_len_list:
                    for s_len in seq_len:
                        llm_flops += (llm_module_flops(s_len, 1, llm_hidden_size, llm_num_layers, query_groups, attention_heads, ffn_hidden_size, vocab_size) / 1e12)
            vision_flops = vision_flops / (vision_tp_size * vision_pp_size)
            llm_flops = llm_flops / (llm_tp_size * llm_pp_size)
            total_flops = vision_flops + llm_flops
            flops_list.append(total_flops)
            input_list.clear()
            input_meta.clear()
            output_meta.clear()
            target_list.clear()
            kwargs_list.clear()
            image_batch_size_list.clear()
            seq_len_list.clear()
        if global_step == num_training_step:
            break
    torch.cuda.synchronize()
    torch.distributed.barrier()
    # time_list = [start[i].elapsed_time(end[i]) for i in range(num_training_step)]
    profile_time_list = time_list[num_training_step // 2:]
    profile_flops_list = flops_list[num_training_step // 2:]
    sum_time = sum(profile_time_list)
    sum_flops = sum(profile_flops_list)
    print(f"[Rank : {global_rank}] Total time: {sum_time:.2f} s, Total FLOPs: {sum_flops:.2f} TFlops, Throughput : {sum_flops / sum_time:.2f} TFlops/s")
    # Convert to tensors for reduction
    sum_time_tensor = torch.tensor(sum_time, device=device)
    sum_flops_tensor = torch.tensor(sum_flops, device=device)

    # Reduce (sum) from all processes to rank 0
    dist.reduce(sum_time_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(sum_flops_tensor, dst=0, op=dist.ReduceOp.SUM)
    time_log_path = f"{result_path}/gpu_{world_size}_{llm_model_name}_{llm_size}_{vision_model_name}_{vision_size}"
    flops_log_path = f"{result_path}/gpu_{world_size}_{llm_model_name}_{llm_size}_{vision_model_name}_{vision_size}_flops"
    if global_rank == 0:
        dist_total_time = sum_time_tensor.item()
        dist_total_flops = sum_flops_tensor.item()
        print(f"[Total] Total time: {dist_total_time:.2f} s, Total FLOPs: {dist_total_flops:.2f} TFlops, Throughput : {dist_total_flops / dist_total_time:.2f} TFlops/s")
        result_dict = {
            "v_tp": vision_tp_size,
            "v_pp": vision_pp_size,
            "v_dp": vision_dp_size,
            "l_tp": llm_tp_size,
            "l_pp": llm_pp_size,
            "l_dp": llm_dp_size,
            "num_micro_batches": num_micro_batches,
            "total_time": sum_time_tensor.item(),
            "total_flops": sum_flops_tensor.item(),
            "throughput": sum_flops_tensor.item() / sum_time_tensor.item(),
            "vision_size": vision_size,
            "llm_size": llm_size,
            "apply_ac": True,
            "llm_model": llm_model_name,
            "vision_model":vision_model_name,
        }
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        df_path = os.path.join(result_path, "ours_llavaov.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            df = pd.DataFrame(columns=list(result_dict.keys()))
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
        df = df.round(2)
        df.to_csv(df_path, index=False)   
        if not os.path.exists(time_log_path):
            os.makedirs(time_log_path)
        if not os.path.exists(flops_log_path):
            os.makedirs(flops_log_path)
    dist.barrier()
    with open(f"{time_log_path}/ours_llava_time_list_{global_rank}.pkl", "wb") as f:
        pickle.dump(time_dict_list, f)
    with open(f"{flops_log_path}/ours_llava_flops_list_{global_rank}.pkl", "wb") as f:
        pickle.dump(profile_flops_list, f)
    dist.destroy_process_group()
    os._exit(1)
# torchrun --nproc-per-node=4 /giant-data/user/1113870/BDAI/dmllm_codes/ours_llavaov_ver3.py