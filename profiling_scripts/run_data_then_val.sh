#!/bin/bash
#
# run_data_then_val.sh
# Run data collection first, then validation, even if data collection fails.
#
# Usage (from project root):
#   EPOCHS=1 POLL_INTERVAL=1 GRANULARITY=operation DATASET_NAME=gsm8k bash profiling_scripts/run_data_then_val.sh
#

set -o pipefail

PROJECT_DIR="/home/cathxhou/projects/verl_research"
SCRIPT="${PROJECT_DIR}/profiling_scripts/policy_size_model_nnodes.sh"

EPOCHS="${EPOCHS:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-1}"
GRANULARITY="${GRANULARITY:-operation}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: missing script: $SCRIPT"
    exit 1
fi

echo "========================================"
echo "Step 1: Data collection (USE_VALIDATION=0)"
echo "========================================"

DATA_EXIT=0
EPOCHS="$EPOCHS" \
POLL_INTERVAL="$POLL_INTERVAL" \
GRANULARITY="$GRANULARITY" \
DATASET_NAME="$DATASET_NAME" \
USE_VALIDATION=0 \
bash "$SCRIPT" || DATA_EXIT=$?

if [ "$DATA_EXIT" -ne 0 ]; then
    echo "WARNING: data collection failed (exit code: $DATA_EXIT). Continuing to validation."
fi

echo ""
echo "========================================"
echo "Step 2: Validation (USE_VALIDATION=1)"
echo "========================================"

VAL_EXIT=0
EPOCHS="$EPOCHS" \
POLL_INTERVAL="$POLL_INTERVAL" \
GRANULARITY="$GRANULARITY" \
DATASET_NAME="$DATASET_NAME" \
USE_VALIDATION=1 \
bash "$SCRIPT" || VAL_EXIT=$?

if [ "$VAL_EXIT" -ne 0 ]; then
    echo "ERROR: validation failed (exit code: $VAL_EXIT)."
fi

if [ "$DATA_EXIT" -ne 0 ]; then
    exit "$DATA_EXIT"
fi

exit "$VAL_EXIT"
