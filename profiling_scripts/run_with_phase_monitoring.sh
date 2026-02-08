#!/bin/bash
#
# run_with_phase_monitoring.sh
# Runs verl training with phase-level GPU monitoring

set -e

# -------------------- Arguments --------------------
BASE_EXPERIMENT_NAME="${1:-gsm8k_phased}"
EPOCHS="${2:-1}"
POLL_INTERVAL="${3:-1}"
GRANULARITY="${4:-phase}"  # 'phase' or 'operation'
MODEL_NAME="${5:-Qwen/Qwen2.5-0.5B-Instruct}"
POLICY="${6:-ppo}"  # ppo | remax
NNODES="${7:-1}"
N_GPUS_PER_NODE="${8:-1}"
DATASET_NAME="${9:-gsm8k}"
GPU_ID=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${TIMESTAMP}"

# -------------------- Paths --------------------
PROJECT_DIR="/home/cathxhou/projects/verl_research"
PROFILING_DIR="${PROJECT_DIR}/profiling_scripts"
MONITORING_DIR="${PROJECT_DIR}/monitoring"
if [ "${USE_VALIDATION:-0}" = "1" ]; then
    MONITORING_DIR="${PROJECT_DIR}/monitoring_val"
fi

mkdir -p "$MONITORING_DIR"
cd "$PROJECT_DIR"

export EXPERIMENT_NAME
export MONITORING_DIR

echo "=========================================="
echo "verl Phase/Subphase Profiling"
echo "=========================================="
echo "Experiment (canonical): $EXPERIMENT_NAME"
echo "Epochs: $EPOCHS"
echo "GPU: $GPU_ID"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Monitoring Dir: $MONITORING_DIR"
echo "Granularity: $GRANULARITY (phase | operation for subphase timings)"
echo "Model: $MODEL_NAME"
echo "Policy: $POLICY"
echo "Nodes: $NNODES (gpus per node: $N_GPUS_PER_NODE)"
echo "Dataset: $DATASET_NAME"
echo "=========================================="

# -------------------- Cleanup --------------------
cleanup() {
    echo ""
    echo "Cleaning up..."

    if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "Stopping monitor (PID: $MONITOR_PID)..."
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    # Remove phase state file ONLY (CSV + JSONL are data)
    rm -f "${MONITORING_DIR}/phase_state_${EXPERIMENT_NAME}.json"

    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# -------------------- Start GPU Monitor --------------------
echo "Starting GPU monitor..."

MONITORING_DIR="$MONITORING_DIR" bash "${PROFILING_DIR}/monitor_nvidia_smi_phased.sh" \
    "$EXPERIMENT_NAME" \
    "$GPU_ID" \
    "$POLL_INTERVAL" \
    "$NNODES" \
    "$N_GPUS_PER_NODE" &

MONITOR_PID=$!
echo "Monitor started (PID: $MONITOR_PID)"

sleep 2
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "ERROR: Monitor failed to start"
    exit 1
fi

# -------------------- Start Training --------------------
echo ""
echo "Starting training..."
echo ""

TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_verl_train_nonval.sh}"

bash "${PROFILING_DIR}/${TRAIN_SCRIPT}" \
    "$EXPERIMENT_NAME" \
    "$EPOCHS" \
    "$GPU_ID" \
    "$GRANULARITY" \
    "$MODEL_NAME" \
    "$POLICY" \
    "$NNODES" \
    "$N_GPUS_PER_NODE" \
    "$DATASET_NAME"

echo ""
echo "Training complete."
