#!/bin/bash
set -euo pipefail

# --- Argument check ---
if [ "$#" -lt 3 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <num_nodes> <rank_number> <master_addr>"
    echo "Example: $0 4 0 xxx.xx.xx.xx"
    exit 1
fi

NNODES="$1"
RANKNUM="$2"
MASTER_ADDR="$3"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

PYTHON_FILE="${SCRIPT_DIR}/../run_profile.py"

command -v torchrun >/dev/null 2>&1 || {
  echo "Error: torchrun not found in PATH"; exit 1;
}

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
  "$PYTHON_FILE"