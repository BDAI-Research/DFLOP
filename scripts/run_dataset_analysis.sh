#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <script_path> <mllm_model_name> <vision_model_name> <llm_model_name>"
    exit 1
fi

SCRIPT_PATH=$1
MLLM_MODEL_NAME=$2
VISION_MODEL_NAME=$3
LLM_MODEL_NAME=$4

echo "Running data analysis with MLLM: ${MLLM_MODEL_NAME}, Vision: ${VISION_MODEL_NAME}, LLM: ${LLM_MODEL_NAME}"
python "${SCRIPT_PATH}" \
    --mllm_model_name "${MLLM_MODEL_NAME}" \
    --vision_model_name "${VISION_MODEL_NAME}" \
    --llm_model_name "${LLM_MODEL_NAME}" \
    --profile_mode data