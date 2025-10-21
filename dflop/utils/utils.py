import os
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from datetime import timedelta
from torch.distributed._composable.replicate import replicate
from llava_utils import init_qwen_loader_model, BaseLLaVAOVPipeStage
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig
from torchtune.modules.loss import LinearCrossEntropyLoss
from torchtitan.distributed import ParallelDims
from torch.utils.data import DataLoader, RandomSampler
from torchtune.modules.loss import LinearCrossEntropyLoss
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    loss_parallel
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,)
from dmllm_utils.internvit_modules import InternVisionConfig

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

llm_tp_plan = {
        "embed_tokens": RowwiseParallel(
            input_layouts=Replicate()
        ),
        "layers.*.attn.q_proj": ColwiseParallel(),
        "layers.*.attn.k_proj": ColwiseParallel(),
        "layers.*.attn.v_proj": ColwiseParallel(),
        "layers.*.attn.output_proj": RowwiseParallel(),
        "layers.*.mlp.w1": ColwiseParallel(),
        "layers.*.mlp.w2": RowwiseParallel(),
        "layers.*.mlp.w3": ColwiseParallel(),
        "output": ColwiseParallel(output_layouts=Replicate()),
    }

sig_lip_configs = {"400m": SigLipVisionConfig(hidden_size=1152,
                                                intermediate_size=4304,
                                                num_hidden_layers=27,
                                                num_attention_heads=16,
                                                ),
                    "1b": SigLipVisionConfig(hidden_size=1728,
                                        intermediate_size=6912,
                                        num_hidden_layers=27,
                                        num_attention_heads=32,),
                    "6b":SigLipVisionConfig(hidden_size=2304,
                                    intermediate_size=8608,
                                    num_hidden_layers=40,
                                    num_attention_heads=48,
                                    )}

internvit_configs = {"6b":InternVisionConfig(hidden_size=3200,
                                            intermediate_size=12800,
                                            num_hidden_layers=30,
                                            num_attention_heads=16,
                                            image_size=448,
                                            hidden_act="gelu",
                                            norm_type="rms_norm",
                                            num_channels=3,
                                            patch_size=14,
                                            qk_normalization=True,
                                            qkv_bias=False,
                                            torch_dtype="float16",
                                            use_bfloat16=False,
                                            use_flash_attn=True),
                    "1b": InternVisionConfig(hidden_size=1792,
                                            intermediate_size=7168,
                                            num_hidden_layers=28,
                                            num_attention_heads=28,
                                            image_size=448,
                                            hidden_act="gelu",
                                            norm_type="rms_norm",
                                            num_channels=3,
                                            patch_size=14,
                                            qk_normalization=True,
                                            qkv_bias=False,
                                            torch_dtype="float16",
                                            use_bfloat16=False,
                                            use_flash_attn=True),
                                                                
                    "300m":InternVisionConfig(hidden_size=1024,
                                            intermediate_size=4096,
                                            num_hidden_layers=24,
                                            image_size=448,
                                            num_attention_heads=16,
                                            hidden_act="gelu",
                                            norm_type="layer_norm",
                                            num_channels=3,
                                            patch_size=14,
                                            qk_normalization=False,
                                            qkv_bias=True,
                                            torch_dtype="float16",
                                            use_bfloat16=False,
                                            use_flash_attn=True
                                            )}

qwen_configs = {"7b" : {"num_layers" : 28,
                        "num_heads" : 28,
                        "num_kv_heads" : 4,
                        "embed_dim" : 3584,
                        "intermediate_dim" : 18944
                        },
                "32b" : {"num_layers" : 64,
                        "num_heads" : 40,
                        "num_kv_heads" : 8,
                        "embed_dim" : 5120,
                        "intermediate_dim" : 27648
                        },
                "72b" : {"num_layers" : 80,
                        "num_heads" : 64,
                        "num_kv_heads" : 8,
                        "embed_dim" : 8192,
                        "intermediate_dim" : 29568
                        }
                }

llama_configs = {"8b" : {"num_layers" : 32,
                        "num_heads" : 32,
                        "num_kv_heads" : 8,
                        "embed_dim" : 4096,
                        "intermediate_dim" : 14336
                        },
                "70b" : {"num_layers" : 80,
                        "num_heads" : 64,
                        "num_kv_heads" : 8,
                        "embed_dim" : 8192,
                        "intermediate_dim" : 28672
                        }
                }

                
vision_configs = {"siglip":sig_lip_configs, 
                 "internvit":internvit_configs}
llm_configs = {"qwen2":qwen_configs, "llama3":llama_configs}

def init_distributed(timeout=None):
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if timeout is not None:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    else:
        dist.init_process_group(backend="nccl")
    return local_rank, global_rank, world_size

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.layers.register_module(layer_id, transformer_block)
