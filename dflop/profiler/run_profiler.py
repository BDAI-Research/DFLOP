import time
import subprocess
import argparse
import torch.distributed as dist

def main():
    # --- (기존 ArgumentParser 및 task list 정의 부분은 동일) ---
    parser = argparse.ArgumentParser(description="Launcher for distributed profiling tasks")
    parser.add_argument('--mllm_model_name', type=str, default="llavaov", help='MLLM model name')
    parser.add_argument('--vision_model_name', type=str, default="siglip", help='Vision model name')
    parser.add_argument('--vision_model_size', type=str, default="400m", help='Vision model size')
    parser.add_argument('--llm_model_name', type=str, default="qwen2", help='LLM model name')
    parser.add_argument('--llm_model_size', type=str, default="7b", help='LLM model size')
    args = parser.parse_args()
    
    script_path = "/giant-data/user/1113870/BDAI/dmllm_codes/profile_scripts"
    profiler_path = "/giant-data/user/1113870/BDAI/dmllm_codes/dmllm_profiler.py"

    # --- (task list 정의 부분도 동일) ---
    data_analysis_task = ["bash", f"{script_path}/run_dataset_analysis.sh", profiler_path, args.mllm_model_name, args.vision_model_name, args.llm_model_name]
    vision_mem_task = ["bash", f"{script_path}/run_mem_vision.sh", profiler_path, args.mllm_model_name, args.vision_model_name, args.vision_model_size]
    llm_mem_task = ["bash", f"{script_path}/run_mem_llm.sh", profiler_path, args.mllm_model_name, args.llm_model_name, args.llm_model_size]
    vision_thr_task = ["bash", f"{script_path}/run_thr_vision.sh", profiler_path, args.mllm_model_name, args.vision_model_name, args.vision_model_size]
    llm_thr_full_task = ["bash", f"{script_path}/run_thr_llm_full.sh", profiler_path, args.mllm_model_name, args.llm_model_name, args.llm_model_size]
    llm_thr_skip_attn_task = ["bash", f"{script_path}/run_thr_llm_skip_attn.sh", profiler_path, args.mllm_model_name, args.llm_model_name, args.llm_model_size]
    
    # 분산 환경 초기화
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # 💡 1. 백그라운드 프로세스 핸들을 try 블록 외부에서 초기화
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
            print(f"Rank {rank}: No profiling task assigned, waiting...")
        else:
            for task_command in assigned_tasks:
                print(f"Rank {rank}: Running profiling task: {task_command}")
                # 이 부분에서 에러가 발생하면 except 블록으로 이동
                subprocess.run(task_command, check=True)
        
        # 모든 랭크가 동기화될 때까지 대기
        dist.barrier()
        
        # 정상 종료 시, Rank 0은 백그라운드 프로세스가 끝날 때까지 기다림
        if rank == 0 and background_process:
            print("Rank 0: Main tasks finished. Waiting for background process to complete...")
            background_process.wait()

    except Exception as e:
        # 에러 발생 시 모든 랭크에서 에러 메시지 출력
        print(f"!!! An error occurred on Rank {rank}: {e}. Initiating cleanup. !!!")
    
    finally:
        # 💡 2. 스크립트가 정상 종료되거나 에러로 종료될 때 항상 실행
        if rank == 0 and background_process:
            # poll()은 프로세스가 종료되었으면 exit code를, 실행 중이면 None을 반환
            if background_process.poll() is None:
                print("Rank 0: Main script is exiting. Terminating background data analysis task...")
                background_process.terminate()  # 프로세스에 종료 신호(SIGTERM) 전송
                background_process.wait()       # 프로세스가 완전히 종료될 때까지 대기

        print(f"Rank {rank}: Destroying process group.")
        dist.destroy_process_group()

if __name__ == "__main__":
    start_profile = time.time()
    main()
    end_profile = time.time()
    profile_duration = end_profile - start_profile
    print(f"Profiling Duration : {profile_duration:.2f}")