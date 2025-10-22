#!/bin/bash

# --- Argument check ---
if [ "$#" -lt 1 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <python_script_path>"
    echo "Example: $0 /path/to/data_aware_optimization.py"
    exit 1
fi
PYTHON_FILE="$1"

# --- Display configuration ---
echo "========================================="
echo "Launching DFLOP Data-aware Optimization with Parameters:"
echo "  Python Script   : $PYTHON_FILE"
echo "========================================="

# --- Launch torchrun ---
python "$PYTHON_FILE"