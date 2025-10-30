# DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization

DFLOP is a **data-driven optimization framework** designed to improve distributed training efficiency for **Multimodal Large Language Models (MLLMs)**.  
Unlike existing data-agnostic frameworks that parallelize computation blindly, DFLOP adapts parallelism and scheduling to the **real data characteristics**, mitigating computation imbalance and input-dependent performance variance.


## Overview

DFLOP consists of three core components:

1. **Profiling Engine**  
   - Profiles both model and data workloads.  
   - Builds predictive models for memory and throughput across input shapes.  
   - Analyzes the empirical input-shape distribution from real datasets.

2. **Data-aware 3D Parallelism Optimizer**  
   - Uses profiling results to determine optimal 3D parallelism configurations  
     (Tensor / Pipeline / Data Parallelism) for each module independently.  
   - Minimizes expected makespan under memory and hardware constraints.

3. **Online Microbatch Scheduler**  
   - Dynamically partitions each training batch using **Integer Linear Programming (ILP)**.  
   - Balances computation load across pipeline stages in real time.  
   - Reduces GPU idle time caused by pipeline bubbles.

<div align="center">
  <img src="figure/DFLOP_overview.png" alt="DFLOP System Overview" width="30%">
</div>

## Getting Started

### Installation

1. Install package
```bash
conda create -n dflop python=3.10 -y
conda activate dflop
pip install --upgrade pip  # enable PEP 660 support
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124
```

2. Install additional packages
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

### Download dataset

- [Single Image Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)
- [Multiple Image Dataset](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data)
- [Video Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)

After downloading:
	•	Place Single Image Dataset and Multiple Image Dataset inside the image_folder
(e.g., data/image_folder/)
	•	Place Video Dataset inside the video_folder
(e.g., data/video_folder/)

Set these dataset paths in the dataset paths in [`configs/dataset_config.yaml`](configs/dataset_config.yaml).


## How to Use DFLOP

- `mllm_model_name` can be selected from the following options:
  - **llavaov**
  - **internvl**

- `llm_model_name` can be selected from:
  - **qwen2.5**
  - **llama3**

DFLOP uses a separate configuration file, [`configs/dflop_config.yaml`](configs/dflop_config.yaml),
to define model selection, dataset paths, hardware resources, and training parameters.

You must fill in the commented sections before running DFLOP.

### Running the DFLOP Profiling Engine
Navigate to [scripts](scripts) folder.

The run_profiling_engine.sh script launches the Profiling Engine of DFLOP across multiple nodes.

Each node must have a unique rank_number, assigned sequentially (e.g., 0, 1, 2, 3, ...),
so that every node can correctly identify its role in the distributed profiling job.

```bash
bash run_profiling_engine.sh <num_nodes> <rank_number> <master_addr>
```

### Running the Data-aware 3D Parallelism Optimizer
After completing the profiling stage, run the **Data-aware 3D Parallelism Optimizer** to automatically search for optimal parallel configurations based on the profiling results.

```bash
bash run_data_aware_optimization.sh
```

### Training with the Online Microbatch Scheduler
Once the optimized configuration is generated, you can start the training phase.
During this stage, DFLOP’s Online Microbatch Scheduler runs asynchronously to dynamically balance workloads across GPU pipeline stages in real time. 

Each node must have a unique rank_number, assigned sequentially (e.g., 0, 1, 2, 3, ...),
so that every node can correctly identify its role in the distributed profiling job.

```bash
bash run_training.sh <num_nodes> <rank_number> <master_addr>
```