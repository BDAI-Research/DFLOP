import subprocess
import torch.distributed as dist
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from dflop.config import load_config, resolve_path

def main():
    BASE_DIR = Path(__file__).resolve().parent
    config = load_config()
    paths_cfg = config.get("paths", {})
    models_cfg = config.get("models", {})
    vision_cfg = models_cfg.get("vision", {})
    llm_cfg = models_cfg.get("llm", {})
    settings = SimpleNamespace(
        mllm_model_name=models_cfg.get("mllm"),
        vision_model_name=vision_cfg.get("name"),
        vision_model_size=vision_cfg.get("size"),
        llm_model_name=llm_cfg.get("name"),
        llm_model_size=llm_cfg.get("size"),
    )
    script_path = (BASE_DIR / "scripts").resolve()
    profiler_path = (BASE_DIR / "prof_engine.py").resolve()
    data_analysis_task = ["bash", f"{script_path}/run_dataset_analysis.sh", profiler_path, settings.mllm_model_name, settings.vision_model_name, settings.llm_model_name]
    vision_mem_task = ["bash", f"{script_path}/run_mem_vision.sh", profiler_path, settings.mllm_model_name, settings.vision_model_name, settings.vision_model_size]
    llm_mem_task = ["bash", f"{script_path}/run_mem_llm.sh", profiler_path, settings.mllm_model_name, settings.llm_model_name, settings.llm_model_size]
    vision_thr_task = ["bash", f"{script_path}/run_thr_vision.sh", profiler_path, settings.mllm_model_name, settings.vision_model_name, settings.vision_model_size]
    llm_thr_full_task = ["bash", f"{script_path}/run_thr_llm_full.sh", profiler_path, settings.mllm_model_name, settings.llm_model_name, settings.llm_model_size]
    llm_thr_skip_attn_task = ["bash", f"{script_path}/run_thr_llm_skip_attn.sh", profiler_path, settings.mllm_model_name, settings.llm_model_name, settings.llm_model_size]
    result_dir = paths_cfg.get("result_dir")
    result_path = resolve_path(result_dir)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    background_process = None
    try:
        if rank == 0:
            if not result_path.exists():
                result_path.mkdir(parents=True, exist_ok=True)
            print("Rank 0: Starting data analysis task in the background.")
            background_process = subprocess.Popen(data_analysis_task)
        # Assign and run tasks
        profiling_tasks = [vision_mem_task, llm_mem_task, vision_thr_task, llm_thr_full_task, llm_thr_skip_attn_task]
        world_size = dist.get_world_size()
        assigned_tasks = []

        for i in range(len(profiling_tasks)):
            assigned_rank = i % world_size
            if assigned_rank == rank:
                assigned_tasks.append(profiling_tasks[i])

        if not assigned_tasks:
            print(f"Rank {rank}: No profiling task assigned, waiting.")
        else:
            for task_command in assigned_tasks:
                print(f"Rank {rank}: Running profiling task: {task_command}")
                subprocess.run(task_command, check=True)
        dist.barrier()
        
        if rank == 0 and background_process:
            print("Rank 0: Main tasks finished. Waiting for background process to complete.")
            background_process.wait()

    except Exception as e:
        print(f"An error occurred on Rank {rank}: {e}. Initiating cleanup.")
    
    finally:
        if rank == 0 and background_process:
            if background_process.poll() is None:
                print("Rank 0: Main script is exiting. Terminating background data analysis task.")
                background_process.terminate()
                background_process.wait()

        print(f"Rank {rank}: Destroying process group.")
        dist.destroy_process_group()
if __name__ == "__main__":
    main()