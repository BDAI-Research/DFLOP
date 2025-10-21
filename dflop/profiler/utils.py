import torch
from datasets import Dataset
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader
from torchtune.modules.model_fusion import DeepFusionModel, EarlyFusionModel
from torch.distributed.device_mesh import DeviceMesh
from torchtune.modules.attention import MultiHeadAttention
from torchtune_models import FlashMultiHeadAttention
from dmllm_utils.internvit_modules import InternAttention
from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention
from dmllm_utils.qwenvit_modules import Qwen2_5_VLVisionFlashAttention2
from typing import Callable, Union

def flush_cache(l2_cache_size):
    x = torch.empty(l2_cache_size, dtype=torch.int8, device='cuda')
    x.zero_()
    return x

def generate_hf_data(num_samples, seq_len, vocab_size):
    data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (seq_len,)).tolist()
        data.append({"input_ids": input_ids})
    return Dataset.from_list(data)

def tokenize_hf_data(examples):
    return {
        "input_ids": examples["input_ids"],
        "attention_mask": [1] * len(examples["input_ids"]),
        "labels": examples["input_ids"][1:] + [-100]
    }

def generate_torchtune_data(num_samples, seq_len, vocab_size):
    data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (seq_len,)).tolist()
        data.append({"input_ids": input_ids})
    return Dataset.from_list(data)

def tokenize_torchtune_data(examples):
    return {
        "tokens": examples["input_ids"],
        "labels": examples["input_ids"][1:] + [-100]
    }

def prepare_mha_for_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
) -> nn.Module:
    """
    Utility to scale MultiHeadAttention parameters(num_heads, num_kv_heads, embed_dim) across
    tensor parallel devices. Each device will handle a portion of the attention computations.

    Args:
        model (nn.Module): Model whose attention parameters will be scaled by TP size.
        tp_mesh (DeviceMesh): Tensor parallel device mesh.

    Returns:
        nn.Module: The model with scaled MultiHeadAttention parameters.

    Raises:
        ValueError: If attention heads, kv heads, or embed dimension is not divisible by TP size.

    Examples:
        >>> from torchtune.modules import TransformerDecoder
        >>> from torch.distributed.device_mesh import DeviceMesh
        >>> model = TransformerDecoder(
                num_heads=32,
                num_kv_heads=32,
                embed_dim=4096,
            )
        >>> tp_mesh = DeviceMesh("cuda", torch.arange(2))  # 2 GPUs
        >>> model = prepare_mha_for_tp(model, tp_mesh)
        >>> # Now each GPU has:
        >>> # num_heads = 16 (32/2)
        >>> # num_kv_heads = 16 (32/2)
        >>> # embed_dim = 2048 (4096/2)
    """
    # Handle fusion models by extracting decoder
    is_fusion_model = isinstance(model, (DeepFusionModel, EarlyFusionModel))
    decoder = model.decoder if is_fusion_model else model
    tp_size = tp_mesh.size()
    for m in list(decoder.modules()):
        if isinstance(m, MultiHeadAttention) or isinstance(m, FlashMultiHeadAttention):
            # Adjust attention module to use the local number of heads
            if m.num_heads % tp_size != 0:
                raise ValueError(
                    f"Number of attention heads ({m.num_heads}) must be divisible by "
                    f"tensor parallel size ({tp_size})."
                )
            if m.num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"Number of KV heads ({m.num_kv_heads}) must be divisible by "
                    f"tensor parallel size ({tp_size})."
                )
            if m.embed_dim % tp_size != 0:
                raise ValueError(
                    f"Embedding dimension ({m.embed_dim}) must be divisible by "
                    f"tensor parallel size ({tp_size})."
                )
            m.num_heads = m.num_heads // tp_size
            m.num_kv_heads = m.num_kv_heads // tp_size
            m.embed_dim = m.embed_dim // tp_size
        if isinstance(m, SigLipAttention) or isinstance(m, InternAttention):
            if m.num_heads % tp_size != 0:
                raise ValueError(
                    f"Number of attention heads ({m.num_heads}) must be divisible by "
                    f"tensor parallel size ({tp_size})."
                )
            if m.embed_dim % tp_size != 0:
                raise ValueError(
                    f"Embedding dimension ({m.embed_dim}) must be divisible by "
                    f"tensor parallel size ({tp_size})."
                )
            m.num_heads = m.num_heads // tp_size
            m.embed_dim = m.embed_dim // tp_size
        if isinstance(m, InternAttention):
            if m.qk_normalization:
                m.q_norm.weight = nn.Parameter(torch.ones(m.q_norm.weight.shape[0] // tp_size))
                m.k_norm.weight = nn.Parameter(torch.ones(m.k_norm.weight.shape[0] // tp_size))
        if isinstance(m, Qwen2_5_VLVisionFlashAttention2):
            if m.num_heads % tp_size != 0:
                raise ValueError(
                    f"Number of attention heads ({m.num_heads}) must be divisible by "
                    f"tensor parallel size ({tp_size})."
                )
            m.num_heads = m.num_heads // tp_size
    if is_fusion_model:
        model.decoder = decoder
    return model


def torchtune_loader(num_training_steps, batch_size, sequence_length, vocab_size):
    num_samples_for_dummy_data = num_training_steps * batch_size * 2
    dummy_dataset = generate_torchtune_data(num_samples_for_dummy_data, sequence_length, vocab_size)
    tokenized_dataset = dummy_dataset.map(tokenize_torchtune_data, batched=False)
    tokenized_dataset.set_format(type="torch", columns=["tokens", "labels"])
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)
    return data_iter



ACWrapPolicyType = Union[set[type], Callable[[nn.Module, bool, int], bool]]

def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy: ACWrapPolicyType, **kwargs
) -> None:
    """Utility to apply activation checkpointing to the passed-in model.

    Args:
        model (nn.Module): Model to apply activation checkpointing to.
        auto_wrap_policy (ACWrapPolicyType): Policy to wrap module.
            This can either be a set of ``nn.Module`` types, in which case, modules of the specified type(s)
            will be wrapped individually with activation checkpointing, or a ``callable`` policy describing
            how to wrap the model with activation checkpointing. For more information on authoring custom
            policies, please see this tutorial:
            https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html#transformer-wrapping-policy.
        **kwargs: additional arguments to pass to ``torch.distributed`` activation checkpointing.
    """
    if isinstance(auto_wrap_policy, set):
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy)
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)