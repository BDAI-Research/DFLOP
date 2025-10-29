#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_FILE="${SCRIPT_DIR}/../data_aware_optimizer.py"

# --- Display configuration ---
echo "========================================="
echo "Launching DFLOP Data-aware Optimization with Parameters:"
echo "  Python Script   : $PYTHON_FILE"
echo "========================================="

# --- Launch torchrun ---
python "$PYTHON_FILE"