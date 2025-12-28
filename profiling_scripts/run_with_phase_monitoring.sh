#!/bin/bash
#
# run_with_phase_monitoring.sh
# Runs verl training with phase-level GPU monitoring

set -e

# -------------------- Arguments --------------------
BASE_EXPERIMENT_NAME="${1:-gsm8k_phased}"
EPOCHS="${2:-1}"
GPU_ID="${3:-1}"
POLL_INTERVAL="${4:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${TIMESTAMP}"

# -------------------- Paths --------------------
PROJECT_DIR="/home/cathxhou/projects/verl_research"
PROFILING_DIR="${PROJECT_DIR}/profiling_scripts"
MONITORING_DIR="${PROJECT_DIR}/monitoring"

mkdir -p "$MONITORING_DIR"
cd "$PROJECT_DIR"

export EXPERIMENT_NAME

echo "=========================================="
echo "verl Phase-Level Profiling (FIXED)"
echo "=========================================="
echo "Experiment (canonical): $EXPERIMENT_NAME"
echo "Epochs: $EPOCHS"
echo "GPU: $GPU_ID"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Monitoring Dir: $MONITORING_DIR"
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

bash "${PROFILING_DIR}/monitor_nvidia_smi_phased.sh" \
    "$EXPERIMENT_NAME" \
    "$GPU_ID" \
    "$POLL_INTERVAL" &

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

bash "${PROFILING_DIR}/run_verl_gsm8k_light.sh" \
    "$EXPERIMENT_NAME" \
    "$EPOCHS" \
    "$GPU_ID"

echo ""
echo "Training complete."
