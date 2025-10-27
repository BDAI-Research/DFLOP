
import os
import torch
from torchtune import training
from torchtune.modules.loss import LinearCrossEntropyLoss
from torchtune.utils import batch_to_device
from .parallel import prepare_mha_for_tp
from .profile import torchtune_loader
from .profiler import AllocatedMemContext
from torchtune_models import flashqwen2
from torchtune.models.qwen2 import qwen2, qwen2_tokenizer
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel
)
layer_tp_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
        "layers.*.attn": PrepareModuleInput(
            input_layouts=(Shard(1), Shard(1)),
            desired_input_layouts=(Replicate(), Replicate()),
        ),
        "layers.*.mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "layers.*.sa_norm": SequenceParallel(),
        "layers.*.mlp_norm": SequenceParallel(),
        "layers.*.attn.q_proj": ColwiseParallel(),
        "layers.*.attn.k_proj": ColwiseParallel(),
        "layers.*.attn.v_proj": ColwiseParallel(),
        "layers.*.attn.output_proj": RowwiseParallel(
            output_layouts=Shard(1)
        ),
        "layers.*.mlp.w1": ColwiseParallel(),
        "layers.*.mlp.w2": RowwiseParallel(output_layouts=Shard(1)),
        "layers.*.mlp.w3": ColwiseParallel(),
    }

def parallelize_model(model):
    _world_size = int(os.environ["WORLD_SIZE"])
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
    model = prepare_mha_for_tp(model, device_mesh)
    model = parallelize_module(
            module=model,
            device_mesh=device_mesh,
            parallelize_plan=layer_tp_plan
        )
    return model

def init_qwen(model_dtype, num_hidden_layers, attn_impl, is_compiled):
    assert attn_impl in ["sdpa", "flash_attention_2"], "attn_impl must be either 'sdpa' or 'flash_attention_2'"
    model_cls = qwen2 if attn_impl == "sdpa" else flashqwen2
    with training.set_default_dtype(model_dtype):
        model = model_cls(
                vocab_size=152064,
                num_layers=num_hidden_layers,
                num_heads=28,
                num_kv_heads=4,
                embed_dim=3584,
                intermediate_dim=18944,
                max_seq_len=32768,
                attn_dropout=0.0,
                norm_eps=1e-06,
                rope_base=1000000.0,
            )
        loss_fn = LinearCrossEntropyLoss()
        loss_fn.set_model_output(model)
    model = model.to(torch.cuda.current_device())
    with AllocatedMemContext() as model_tp_mem:
        if is_compiled:
            backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
            training.compile_model(model)
            # loss_fn = torch.compile(loss_fn, backend=backend)
        model = parallelize_model(model)
    model_init_peak = model_tp_mem.after["peak"]
    model_cur_mem = model_tp_mem.after["current"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler()
    return model, loss_fn, optimizer, scaler, model_init_peak, model_cur_mem

def profile_model_mem(model_dtype, model, loss_fn, optimizer, scaler, data_iter):
    torch.cuda.reset_peak_memory_stats()
    batch = next(data_iter)
    batch_to_device(batch, torch.cuda.current_device())
    labels = batch.pop("labels")

    # Forward calculation
    with AllocatedMemContext() as fwd_mem:
        if model_dtype == torch.float32:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(**batch)
        else:
            output = model(**batch)
        with loss_parallel():
            loss = loss_fn(output, labels)
    fwd_peak = fwd_mem.delta["peak"]
    # Backward calculation
    with AllocatedMemContext() as bwd_mem:
        if model_dtype == torch.float32:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    bwd_peak = bwd_mem.delta["peak"]
    # Optimizer step
    with AllocatedMemContext() as opt_mem:
        if model_dtype == torch.float32:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    optimizer.zero_grad()
    opt_peak = opt_mem.delta["peak"]
    return fwd_peak, bwd_peak, opt_peak

def model_search_space(model, loss_fn, optimizer, scaler, model_init_peak, 
                       model_cur_mem, model_dtype, vocab_size, seq_search_space, mem_limit):
    if torch.distributed.get_rank() == 0:
        print(f"Searching space between {seq_search_space[0]} and {seq_search_space[-1]} with memory limit {mem_limit} GB")
    batch_size = 1
    seq_len = 512
    num_training_steps = 1
    data_iter = torchtune_loader(num_training_steps, batch_size, seq_len, vocab_size)
    fwd_peak, bwd_peak, opt_peak = profile_model_mem(model_dtype, model, loss_fn, optimizer, scaler, data_iter)
    # print(f"Initial model peak memory: {model_init_peak/(1024**3):.2f} GB, TP peak memory: {model_tp_peak/(1024**3):.2f} GB, Forward peak memory: {fwd_peak/(1024**3):.2f} GB, Backward peak memory: {bwd_peak/(1024**3):.2f} GB, Optimizer peak memory: {opt_peak/(1024**3):.2f} GB")
    if model_dtype == torch.float32:
        # For float32, we need to profile with two different batch sizes to calculate alpha and constant
        batch_size *= 2
        data_iter = torchtune_loader(num_training_steps, batch_size, seq_len, vocab_size)
        fwd_peak_double, _, _ = profile_model_mem(model_dtype, model, loss_fn, optimizer, scaler, data_iter)
        var_val = fwd_peak_double - fwd_peak
        constan_val = fwd_peak - var_val
        alpha = var_val / (batch_size * seq_len)
    else:
        alpha = fwd_peak / (batch_size * seq_len)
    model_state_peak = model_cur_mem + fwd_peak + bwd_peak + opt_peak
    print(f"Model state peak : {model_state_peak/(1024**3):.2f} GB, Alpha : {alpha:.2f}")
    batch_size_list = [1, 2, 4, 8, 16, 32, 64]
    batch_seq_list = []
    for seq_len in seq_search_space:
        for batch_size in batch_size_list:
            if model_dtype == torch.float32:
                fwd_peak = constan_val + alpha * seq_len * batch_size
            else:
                fwd_peak = alpha * seq_len * batch_size
            alloc_mem = (model_state_peak + fwd_peak) / (1024 ** 3)  # Convert to GB
            # if torch.distributed.get_rank() == 0:
            #     print(f"seq_len: {seq_len}, batch_size: {batch_size}, alloc_mem: {alloc_mem:.2f} GB, fwd_peak: {fwd_peak/(1024**3):.2f} GB, model_state_peak: {model_state_peak/(1024**3):.2f} GB")
            if alloc_mem > (mem_limit * 0.9):
                break
            else:
                batch_seq_list.append((batch_size, seq_len))
    return batch_seq_list
