import os
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Any, Dict, Optional
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torchtune.modules.attention import MultiHeadAttention
from torchtune.modules.model_fusion import DeepFusionModel, EarlyFusionModel
from .torchtune_models import FlashMultiHeadAttention
from .internvit_modules import InternAttention
from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention


llava_ov_siglip_tp_plan = {
    "vision_module.encoder.layers.*.self_attn.q_proj": ColwiseParallel(),
    "vision_module.encoder.layers.*.self_attn.k_proj": ColwiseParallel(),
    "vision_module.encoder.layers.*.self_attn.v_proj": ColwiseParallel(),
    "vision_module.encoder.layers.*.self_attn.out_proj": RowwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc1": ColwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc2": RowwiseParallel(),
    "vision_module.mm_projector.0": ColwiseParallel(),
    "vision_module.mm_projector.2": RowwiseParallel(),
}

llavaov_internvit_tp_plan = {
    "vision_module.encoder.layers.*.attn.qkv": ColwiseParallel(),
    "vision_module.encoder.layers.*.attn.proj": RowwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc1": ColwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc2": RowwiseParallel(),
    "vision_module.mm_projector.0": ColwiseParallel(),
    "vision_module.mm_projector.2": RowwiseParallel(),
}

llm_tp_plan = {
    "embed_tokens": RowwiseParallel(input_layouts=Replicate()),
    "layers.*.attn.q_proj": ColwiseParallel(),
    "layers.*.attn.k_proj": ColwiseParallel(),
    "layers.*.attn.v_proj": ColwiseParallel(),
    "layers.*.attn.output_proj": RowwiseParallel(),
    "layers.*.mlp.w1": ColwiseParallel(),
    "layers.*.mlp.w2": RowwiseParallel(),
    "layers.*.mlp.w3": ColwiseParallel(),
    "output": ColwiseParallel(output_layouts=Replicate()),
}

internvl_siglip_tp_plan = {
    "vision_module.encoder.layers.*.self_attn.q_proj": ColwiseParallel(),
    "vision_module.encoder.layers.*.self_attn.k_proj": ColwiseParallel(),
    "vision_module.encoder.layers.*.self_attn.v_proj": ColwiseParallel(),
    "vision_module.encoder.layers.*.self_attn.out_proj": RowwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc1": ColwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc2": RowwiseParallel(),
    "vision_module.mm_projector.1": ColwiseParallel(),
    "vision_module.mm_projector.3": RowwiseParallel(),
}

internvl_internvit_tp_plan = {
    "vision_module.encoder.layers.*.attn.qkv": ColwiseParallel(),
    "vision_module.encoder.layers.*.attn.proj": RowwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc1": ColwiseParallel(),
    "vision_module.encoder.layers.*.mlp.fc2": RowwiseParallel(),
    "vision_module.mm_projector.1": ColwiseParallel(),
    "vision_module.mm_projector.3": RowwiseParallel(),
}


def init_distributed(timeout: Optional[int] = None):
    local_rank = int(os.environ["LOCAL_RANK"])
    if timeout is not None:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    else:
        dist.init_process_group(backend="nccl")
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, global_rank, world_size


def apply_ac(model: nn.Module) -> None:
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        wrapped_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.layers.register_module(layer_id, wrapped_block)


def prepare_mha_for_tp(model: nn.Module, tp_mesh: DeviceMesh) -> nn.Module:
    """
    Scale MultiHeadAttention parameters across tensor-parallel devices so each rank
    handles its local shard of attention computation.
    """

    is_fusion_model = isinstance(model, (DeepFusionModel, EarlyFusionModel))
    decoder = model.decoder if is_fusion_model else model
    tp_size = tp_mesh.size()
    for module in decoder.modules():
        if isinstance(module, (MultiHeadAttention, FlashMultiHeadAttention)):
            if module.num_heads % tp_size != 0:
                raise ValueError(
                    f"Number of attention heads ({module.num_heads}) must be divisible by tensor parallel size ({tp_size})."
                )
            if module.num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"Number of KV heads ({module.num_kv_heads}) must be divisible by tensor parallel size ({tp_size})."
                )
            if module.embed_dim % tp_size != 0:
                raise ValueError(
                    f"Embedding dimension ({module.embed_dim}) must be divisible by tensor parallel size ({tp_size})."
                )
            module.num_heads //= tp_size
            module.num_kv_heads //= tp_size
            module.embed_dim //= tp_size
        if isinstance(module, (SigLipAttention, InternAttention)):
            if module.num_heads % tp_size != 0:
                raise ValueError(
                    f"Number of attention heads ({module.num_heads}) must be divisible by tensor parallel size ({tp_size})."
                )
            if module.embed_dim % tp_size != 0:
                raise ValueError(
                    f"Embedding dimension ({module.embed_dim}) must be divisible by tensor parallel size ({tp_size})."
                )
            module.num_heads //= tp_size
            module.embed_dim //= tp_size
        if isinstance(module, InternAttention) and module.qk_normalization:
            module.q_norm.weight = nn.Parameter(
                torch.ones(module.q_norm.weight.shape[0] // tp_size, device=module.q_norm.weight.device)
            )
            module.k_norm.weight = nn.Parameter(
                torch.ones(module.k_norm.weight.shape[0] // tp_size, device=module.k_norm.weight.device)
            )
    if is_fusion_model:
        model.decoder = decoder
    return model


def set_topology_config(
    world_size: int,
    vision_ranks: int,
    vision_pp_size: int,
    vision_dp_size: int,
    vision_tp_size: int,
    llm_pp_size: int,
    llm_dp_size: int,
    llm_tp_size: int,
) -> Dict[str, Dict[str, Any]]:
    """Dynamically generate topology configuration."""
    return {
        "vision": {
            "ranks": range(vision_ranks),
            "pp_size": vision_pp_size,
            "dp_size": vision_dp_size,
            "tp_size": vision_tp_size,
        },
        "llm": {
            "ranks": range(vision_ranks, world_size),
            "pp_size": llm_pp_size,
            "dp_size": llm_dp_size,
            "tp_size": llm_tp_size,
        },
    }


def _validate_topology(config: Dict[str, Dict[str, Any]], world_size: int) -> None:
    all_ranks_in_config = set()
    for name, component_config in config.items():
        num_ranks = len(component_config["ranks"])
        expected_ranks = (
            component_config["pp_size"] * component_config["dp_size"] * component_config["tp_size"]
        )
        if num_ranks != expected_ranks:
            raise ValueError(
                f"'{name}' component configuration error: pp*dp*tp ({expected_ranks}) "
                f"does not match the assigned number of ranks ({num_ranks})."
            )
        component_ranks = set(component_config["ranks"])
        if not component_ranks.isdisjoint(all_ranks_in_config):
            raise ValueError("Configuration error: There are duplicate ranks between components.")
        all_ranks_in_config.update(component_ranks)
    expected_all_ranks = set(range(world_size))
    if all_ranks_in_config != expected_all_ranks:
        raise ValueError(
            f"Configuration error: Configured ranks ({all_ranks_in_config}) "
            f"do not match actual world_size ({world_size})."
        )


def setup_multinode_distributed_groups(topology_config: Dict[str, Dict[str, Any]]):
    """
    Create parallel processing groups based on flexible topology configuration for multi-node environments.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if rank == 0:
        _validate_topology(topology_config, world_size)
    rank_details = [{} for _ in range(world_size)]
    tp_group_ranks, dp_group_ranks, pp_group_ranks = [], [], []
    cur_module_ranks = None
    for name, config in topology_config.items():
        pp, dp, tp = config["pp_size"], config["dp_size"], config["tp_size"]
        module_ranks = np.arange(config["ranks"][0], config["ranks"][-1] + 1).reshape(pp, dp, tp)
        if rank in config["ranks"]:
            cur_module_ranks = module_ranks
        for p in range(pp):
            for d in range(dp):
                tp_group_ranks.append(module_ranks[p, d, :].tolist())
        for t in range(tp):
            for p in range(pp):
                dp_group_ranks.append(module_ranks[p, :, t].tolist())
        for t in range(tp):
            for d in range(dp):
                pp_group_ranks.append(module_ranks[:, d, t].tolist())
        offset = config["ranks"][0]
        for r in config["ranks"]:
            local_r = r - offset
            rank_details[r] = {
                "component": name,
                "local_rank_in_comp": local_r,
                "pp_rank": (local_r // (tp * dp)) % pp,
                "dp_rank": (local_r // tp) % dp,
                "tp_rank": local_r % tp,
            }
    module_tp_group = None
    module_dp_group = None
    module_pp_group = None
    for ranks_in_tp_group in tp_group_ranks:
        group = dist.new_group(ranks_in_tp_group)
        if rank in ranks_in_tp_group:
            module_tp_group = group
    for ranks_in_dp_group in dp_group_ranks:
        group = dist.new_group(ranks_in_dp_group)
        if rank in ranks_in_dp_group:
            module_dp_group = group
    for ranks_in_pp_group in pp_group_ranks:
        group = dist.new_group(ranks_in_pp_group)
        if rank in ranks_in_pp_group:
            module_pp_group = group
    cur_rank_info = rank_details[rank]
    cur_component = cur_rank_info["component"]

    v_pp_size = topology_config["vision"]["pp_size"]
    v_pp_last_group_ranks = []
    l_pp_first_group_ranks = []
    v_tp_first_group_ranks = []
    l_tp_first_group_ranks = []

    for i in range(world_size):
        i_details = rank_details[i]
        i_tp_rank = i_details["tp_rank"]
        i_pp_rank = i_details["pp_rank"]
        if (i_details["component"] == "vision") and (i_pp_rank == v_pp_size - 1):
            v_pp_last_group_ranks.append(i)
            if i_tp_rank == 0:
                v_tp_first_group_ranks.append(i)
        if (i_details["component"] == "llm") and (i_pp_rank == 0):
            l_pp_first_group_ranks.append(i)
            if i_tp_rank == 0:
                l_tp_first_group_ranks.append(i)

    dist.barrier()

    if cur_module_ranks is None:
        raise RuntimeError("Failed to determine module ranks for current process.")
    if module_tp_group is None or module_dp_group is None or module_pp_group is None:
        raise RuntimeError("Failed to initialize distributed process groups.")

    return {
        "pp_group": module_pp_group,
        "dp_group": module_dp_group,
        "tp_group": module_tp_group,
        "v_pp_last_group_ranks": v_pp_last_group_ranks,
        "l_pp_first_group_ranks": l_pp_first_group_ranks,
        "v_tp_first_group_ranks": v_tp_first_group_ranks,
        "l_tp_first_group_ranks": l_tp_first_group_ranks,
        "pp_size": {
            "vision": topology_config["vision"]["pp_size"],
            "llm": topology_config["llm"]["pp_size"],
        },
        "offset": 0 if cur_component == "vision" else topology_config["vision"]["pp_size"],
        "component": cur_component,
        "module_ranks": cur_module_ranks,
        "details": cur_rank_info,
    }
