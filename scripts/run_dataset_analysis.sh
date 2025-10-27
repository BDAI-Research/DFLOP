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
PROFILE_MODE="${PROFILE_MODE:-data}"

echo "Running data analysis (mode=${PROFILE_MODE}) using config: ${CONFIG_PATH}"
DFLOP_CONFIG="$CONFIG_PATH" PROFILE_MODE="$PROFILE_MODE" python "$SCRIPT_PATH"
