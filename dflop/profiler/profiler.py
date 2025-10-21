import time
import torch
from dflop.github.DFLOP.dflop.profiler.utils import flush_cache, torchtune_loader
from typing import Any, Iterable, Optional, Union
from torchtune.utils import batch_to_device
from torch.distributed.tensor.parallel import loss_parallel


class CUDATimer: # Context manager to measure CUDA execution time
    def __init__(self, timers_list):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.timers_list = timers_list

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize()
        self.timers_list.append(self.start_event.elapsed_time(self.end_event))

class AllocatedMemContext: # Context manager to track CUDA memory allocation changes
    def __init__(self) -> None:
        # Ensure CUDA libraries are loaded:
        torch.cuda.current_blas_handle()

        self.before: dict[str, int] = {}
        self.after: dict[str, int] = {}
        self.delta: dict[str, int] = {}

    def _get_mem_dict(self) -> dict[str, int]:
        # Only need `allocated_bytes.all`-prefixed keys here
        key_prefix = "allocated_bytes.all."
        return {
            k.replace(key_prefix, ""): v
            for k, v in torch.cuda.memory_stats().items()
            if key_prefix in k
        }

    def __enter__(self) -> "AllocatedMemContext":
        torch.cuda.reset_peak_memory_stats() # Reset peak memory stats before measuring
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}
 
class SavedTensorContext: # Context manager to track saved activation tensors in autograd
    def __init__(
        self,
        ignored_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> None:
        self._ignored_data_ptrs = (
            set()
            if ignored_tensors is None
            else {t.untyped_storage().data_ptr() for t in ignored_tensors}
        )

        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = saved_tensor.untyped_storage().data_ptr()
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[saved_tensor] = data_ptr
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook, unpack_hook
        )

    def __enter__(self) -> "SavedTensorContext":
        self._saved_tensors_hook.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_tensors_hook.__exit__(*args, **kwargs)

    @property
    def saved_tensor_mem(self) -> int:
        """
        The memory in bytes of all saved tensors, accounting for views into the same storage.
        """
        accounted_for = self._ignored_data_ptrs.copy()
        total_bytes = 0
        for t in self.saved_tensor_dict:
            data_ptr = t.untyped_storage().data_ptr()
            if data_ptr not in accounted_for:
                total_bytes += t.untyped_storage().nbytes()
                accounted_for.add(data_ptr)
        return total_bytes
    
class TPSavedTensorContext: # Context manager to track saved activation tensors in autograd
    def __init__(
        self,
        _ignored_data_ptrs: Optional[Iterable[torch.Tensor]] = None,
    ) -> None:
        self._ignored_data_ptrs = (
            set()
            if _ignored_data_ptrs is None
            else _ignored_data_ptrs
        )

        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = saved_tensor.untyped_storage().data_ptr()
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[saved_tensor] = data_ptr
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook, unpack_hook
        )

    def __enter__(self) -> "TPSavedTensorContext":
        self._saved_tensors_hook.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_tensors_hook.__exit__(*args, **kwargs)

    @property
    def saved_tensor_mem(self) -> int:
        """
        The memory in bytes of all saved tensors, accounting for views into the same storage.
        """
        accounted_for = self._ignored_data_ptrs.copy()
        total_bytes = 0
        for t in self.saved_tensor_dict:
            data_ptr = t.untyped_storage().data_ptr()
            if data_ptr not in accounted_for:
                total_bytes += t.untyped_storage().nbytes()
                accounted_for.add(data_ptr)
        return total_bytes
    
def warmup_model(model, loss_fn, scaler, optimizer, data_iter, num_training_steps, model_dtype):
    for i in range(num_training_steps):
        batch = next(data_iter)
        batch_to_device(batch, torch.cuda.current_device())
        labels = batch.pop("labels")

        if model_dtype == torch.float32:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(**batch)
            with loss_parallel():
                loss = loss_fn(output, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            output = model(**batch)
            with loss_parallel():
                loss = loss_fn(output, labels)    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def profile_model(model, loss_fn, scaler, optimizer, data_iter, num_training_steps, model_dtype, l2_cache_size, batch_size, sequence_length, torch_profiler=False):
    # Prevent resiudal  throttling
    tp_size = torch.distributed.get_world_size()
    time.sleep(5)
    if torch_profiler:
        prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=5, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log_dir/prof_qwen2_bsz{batch_size}_seqlen{sequence_length}_tp{tp_size}'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
                )
        prof.start()
        for i in range(num_training_steps):
            _ = flush_cache(l2_cache_size)
            torch.cuda._sleep(1_000_000)
            batch = next(data_iter)
            batch_to_device(batch, torch.cuda.current_device())
            labels = batch.pop("labels")
            if model_dtype == torch.float32:
                with torch.profiler.record_function("Forward Pass"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        output = model(**batch)
                    with loss_parallel():
                        loss = loss_fn(output, labels)
                with torch.profiler.record_function("Backward Pass"):
                    scaler.scale(loss).backward()
                with torch.profiler.record_function("Optimizer Step"):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                with torch.profiler.record_function("Forward Pass"):
                    output = model(**batch)
                    with loss_parallel():
                        loss = loss_fn(output, labels)    
                with torch.profiler.record_function("Backward Pass"):
                    loss.backward()
                with torch.profiler.record_function("Optimizer Step"):
                    optimizer.step()
                    optimizer.zero_grad()
            prof.step()
        prof.stop()
        if torch.distributed.get_rank() == 0:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        fwd_avg_time = []
        bwd_avg_time = []
        opt_avg_time = []
    else:
        # Warmup the model
        warmup_model(model, loss_fn, scaler, optimizer, data_iter, num_training_steps, model_dtype)
        start_fwd_times = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
        end_fwd_times = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
        start_bwd_times = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
        end_bwd_times = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
        start_opt_times = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
        end_opt_times = [torch.cuda.Event(enable_timing=True) for _ in range(num_training_steps)]
        for i in range(num_training_steps):
            _ = flush_cache(l2_cache_size)
            torch.cuda._sleep(1_000_000)
            batch = next(data_iter)
            batch_to_device(batch, torch.cuda.current_device())
            labels = batch.pop("labels")
            if model_dtype == torch.float32:
                start_fwd_times[i].record()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        output = model(**batch)
                with loss_parallel():
                    loss = loss_fn(output, labels)
                end_fwd_times[i].record()
                start_bwd_times[i].record()
                scaler.scale(loss).backward()
                end_bwd_times[i].record()
                start_opt_times[i].record()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                end_opt_times[i].record()
            else:    
                start_fwd_times[i].record()
                output = model(**batch)
                with loss_parallel():
                    loss = loss_fn(output, labels)    
                end_fwd_times[i].record()
                start_bwd_times[i].record()
                loss.backward()
                end_bwd_times[i].record()
                start_opt_times[i].record()
                optimizer.step()
                optimizer.zero_grad()
                end_opt_times[i].record()
            
        torch.cuda.synchronize()
        fwd_times = [start.elapsed_time(end) for start, end in zip(start_fwd_times, end_fwd_times)]
        bwd_times = [start.elapsed_time(end) for start, end in zip(start_bwd_times, end_bwd_times)]
        opt_times = [start.elapsed_time(end) for start, end in zip(start_opt_times, end_opt_times)]
        fwd_avg_time = round(sum(fwd_times) / num_training_steps, 2)
        bwd_avg_time = round(sum(bwd_times) / num_training_steps, 2)
        opt_avg_time = round(sum(opt_times) / num_training_steps, 2)
        if torch.distributed.get_rank() == 0:
            print(f"Forward Avg Time: {fwd_avg_time} ms, Backward Avg Time: {bwd_avg_time} ms, Optimizer Avg Time: {opt_avg_time} ms")
    return fwd_avg_time, bwd_avg_time, opt_avg_time