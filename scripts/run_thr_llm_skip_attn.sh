#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <script_path> <mllm_model_name> <llm_model_name> <llm_model_size>"
    exit 1
fi

SCRIPT_PATH=$1
MLLM_MODEL_NAME=$2
LLM_MODEL_NAME=$3
LLM_MODEL_SIZE=$4


echo "Running throughput profiler with MLLM: ${MLLM_MODEL_NAME}, LLM: ${LLM_MODEL_NAME}, Size: ${LLM_MODEL_SIZE}"
for TP_SIZE in 1 2 4 8; do
    for LAYERS in 4; do
        echo "=== Running with TP size=${TP_SIZE} num_hidden_layers=${LAYERS} ==="
        torchrun --nproc-per-node=$TP_SIZE "${SCRIPT_PATH}" \
            --mllm_model_name "${MLLM_MODEL_NAME}" \
            --llm_model_name "${LLM_MODEL_NAME}" \
            --llm_model_size "${LLM_MODEL_SIZE}" \
            --num_hidden_layers "${LAYERS}" \
            --profile_mode thr \
            --skip_attn
    done
done

