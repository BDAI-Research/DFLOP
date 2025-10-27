from typing import Callable, Union

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from .parallel import prepare_mha_for_tp  # noqa: F401
from .profile import flush_cache, torchtune_loader  # noqa: F401


ACWrapPolicyType = Union[set[type], Callable[[nn.Module, bool, int], bool]]


def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy: ACWrapPolicyType, **kwargs
) -> None:
    """Apply activation checkpointing using the provided wrap policy."""
    if isinstance(auto_wrap_policy, set):
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy)
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
