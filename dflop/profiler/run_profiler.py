import time
import subprocess
import argparse
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Launcher for distributed profiling tasks")
    parser.add_argument('--mllm_model_name', type=str, default="llavaov", help='MLLM model name')
    parser.add_argument('--vision_model_name', type=str, default="siglip", help='Vision model name')
    parser.add_argument('--vision_model_size', type=str, default="400m", help='Vision model size')
    parser.add_argument('--llm_model_name', type=str, default="qwen2", help='LLM model name')
    parser.add_argument('--llm_model_size', type=str, default="7b", help='LLM model size')
    args = parser.parse_args()
    
    script_path = "/giant-data/user/1113870/BDAI/dmllm_codes/profile_scripts"
    profiler_path = "/giant-data/user/1113870/BDAI/dmllm_codes/dmllm_profiler.py"

    # Task definitions
    data_analysis_task = ["bash", f"{script_path}/run_dataset_analysis.sh", profiler_path, args.mllm_model_name, args.vision_model_name, args.llm_model_name]
    vision_mem_task = ["bash", f"{script_path}/run_mem_vision.sh", profiler_path, args.mllm_model_name, args.vision_model_name, args.vision_model_size]
    llm_mem_task = ["bash", f"{script_path}/run_mem_llm.sh", profiler_path, args.mllm_model_name, args.llm_model_name, args.llm_model_size]
    vision_thr_task = ["bash", f"{script_path}/run_thr_vision.sh", profiler_path, args.mllm_model_name, args.vision_model_name, args.vision_model_size]
    llm_thr_full_task = ["bash", f"{script_path}/run_thr_llm_full.sh", profiler_path, args.mllm_model_name, args.llm_model_name, args.llm_model_size]
    llm_thr_skip_attn_task = ["bash", f"{script_path}/run_thr_llm_skip_attn.sh", profiler_path, args.mllm_model_name, args.llm_model_name, args.llm_model_size]
    
    # Initialize distributed environment
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # Initialize background process handle outside try-block
    background_process = None
    
    try:
        if rank == 0:
            print("Rank 0: Starting data analysis task in the background...")
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
            print(f"Rank {rank}: No profiling task assigned. Waiting...")
        else:
            for task_command in assigned_tasks:
                print(f"Rank {rank}: Running profiling task: {task_command}")
                # If an error occurs here, it will be caught by the except block
                subprocess.run(task_command, check=True)
        
        # Synchronize all ranks
        dist.barrier()
        
        # On normal exit, rank 0 waits for the background process to finish
        if rank == 0 and background_process:
            print("Rank 0: Main tasks finished. Waiting for background process to complete...")
            background_process.wait()

    except Exception as e:
        # On error, print message from every rank
        print(f"!!! An error occurred on Rank {rank}: {e}. Initiating cleanup. !!!")
    
    finally:
        # Always executed on normal or error exit
        if rank == 0 and background_process:
            # poll() returns exit code if finished, or None if still running
            if background_process.poll() is None:
                print("Rank 0: Main script is exiting. Terminating background data analysis task...")
                background_process.terminate()  # Send SIGTERM
                background_process.wait()       # Wait for complete termination

        print(f"Rank {rank}: Destroying process group.")
        dist.destroy_process_group()

if __name__ == "__main__":
    start_profile = time.time()
    main()
    end_profile = time.time()
    profile_duration = end_profile - start_profile
    print(f"Profiling Duration : {profile_duration:.2f}")