#!/bin/bash

# --- Argument check ---
if [ "$#" -lt 4 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <num_nodes> <rank_number> <python_script_path> <master_addr>"
    echo "Example: $0 5 0 /path/to/train.py xxx.xx.xx.xx"
    exit 1
fi

NNODES="$1"
RANKNUM="$2"
PYTHON_FILE="$3"
MASTER_ADDR="$4"

# --- Display configuration ---
echo "========================================="
echo "Launching DFLOP Profiling Engine with Parameters:"
echo "  Number of Nodes : $NNODES"
echo "  Node Rank       : $RANKNUM"
echo "  Python Script   : $PYTHON_FILE"
echo "  Master Address  : $MASTER_ADDR"
echo "========================================="

# --- Launch torchrun ---
torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node=1 \
  --master_addr="$MASTER_ADDR" \
  --master_port=25000 \
  --node-rank="$RANKNUM" \
  "$PYTHON_FILE" \