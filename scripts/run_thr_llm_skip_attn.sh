#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <profiler_script_path> <config_path>"
    exit 1
fi

SCRIPT_PATH="$1"
CONFIG_PATH="$2"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: profiler script not found at $SCRIPT_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config file not found at $CONFIG_PATH"
    exit 1
fi

CONFIG_PATH="$(realpath "$CONFIG_PATH")"
PROFILE_MODE="${PROFILE_MODE:-thr}"

echo "Running LLM throughput profiling (skip attention, mode=${PROFILE_MODE}) using config: ${CONFIG_PATH}"
for TP_SIZE in 1 2 4 8; do
    for LAYERS in 4; do
        echo "=== Running with TP size=${TP_SIZE} num_hidden_layers=${LAYERS} ==="
        DFLOP_CONFIG="$CONFIG_PATH" PROFILE_MODE="$PROFILE_MODE" NUM_HIDDEN_LAYERS="$LAYERS" SKIP_ATTN=1 \
            torchrun --nproc-per-node="$TP_SIZE" "$SCRIPT_PATH"
    done
done
