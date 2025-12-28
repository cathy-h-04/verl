#!/bin/bash
#
# run_verl_with_subphase_monitoring.sh
# Runs verl training with sub-phase level GPU monitoring
#
# Usage: ./run_verl_with_subphase_monitoring.sh [experiment_name] [epochs] [gpu_id] [poll_interval] [granularity]
# Example: ./run_verl_with_subphase_monitoring.sh gsm8k_subphase 1 1 1 operation

set -e

EXPERIMENT_NAME="${1:-gsm8k_subphase}"
EPOCHS="${2:-1}"
GPU_ID="${3:-1}"
POLL_INTERVAL="${4:-1}"
GRANULARITY="${5:-operation}"  # 'phase' or 'operation'

PROJECT_DIR="/home/cathxhou/projects/thesis_research/verl_research"
cd "$PROJECT_DIR"

mkdir -p monitoring

echo "=========================================="
echo "verl Sub-Phase Level Profiling"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Epochs: $EPOCHS"
echo "GPU: $GPU_ID"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Granularity: $GRANULARITY"
echo "=========================================="
echo ""
if [ "$GRANULARITY" = "operation" ]; then
    echo "NOTE: Operation-level profiling will create timing log"
elif [ "$GRANULARITY" = "phase" ]; then
    echo "NOTE: Phase-level profiling only (no timing log)"
else
    echo "WARNING: Invalid granularity '$GRANULARITY', using 'phase'"
    GRANULARITY="phase"
fi
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "Stopping monitor (PID: $MONITOR_PID)..."
        kill $MONITOR_PID 2>/dev/null || true
        wait $MONITOR_PID 2>/dev/null || true
    fi
    
    # Clean up phase state file (timing log is kept as valuable data)
    rm -f monitoring/phase_state_${EXPERIMENT_NAME}*.json
    
    echo "Done!"
    echo ""
    echo "=========================================="
    echo "Output Files Generated:"
    echo "=========================================="
    
    # Find the actual generated files (with timestamp)
    CSV_FILE=$(ls -t monitoring/${EXPERIMENT_NAME}*_phased.csv 2>/dev/null | head -1)
    if [ -n "$CSV_FILE" ]; then
        echo "GPU Metrics CSV: $CSV_FILE"
    else
        echo "GPU Metrics CSV: monitoring/${EXPERIMENT_NAME}*_phased.csv (not found)"
    fi
    
    if [ "$GRANULARITY" = "operation" ]; then
        TIMING_FILE=$(ls -t monitoring/phase_timings_${EXPERIMENT_NAME}*.jsonl 2>/dev/null | head -1)
        if [ -n "$TIMING_FILE" ]; then
            echo "Timing Log: $TIMING_FILE"
            echo ""
            echo "To analyze sub-phase metrics, run:"
            echo "  python analyze_subphase_metrics.py \\"
            echo "    --gpu-csv $CSV_FILE \\"
            echo "    --timing-log $TIMING_FILE"
        else
            echo "Timing Log: monitoring/phase_timings_${EXPERIMENT_NAME}*.jsonl (not found)"
        fi
    else
        echo ""
        echo "To analyze phase-level metrics, run:"
        echo "  python analyze_phase_metrics.py $CSV_FILE"
    fi
    echo "=========================================="
}

trap cleanup EXIT INT TERM

# Start monitoring in background
# We need to find the actual experiment name with timestamp AFTER training starts
echo "Starting GPU monitor (will auto-detect timestamped experiment name)..."

# Start a wrapper script that monitors for the phase state file
cat > monitoring/monitor_wrapper_${EXPERIMENT_NAME}.sh << 'EOF_WRAPPER'
#!/bin/bash
EXPERIMENT_BASE="$1"
GPU_ID="$2"
POLL_INTERVAL="$3"
MONITOR_SCRIPT="$4"

# Wait for phase state file to appear (with any timestamp)
echo "Waiting for phase state file to appear..."
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    STATE_FILE=$(ls -t /home/cathxhou/projects/thesis_research/verl_research/monitoring/phase_state_${EXPERIMENT_BASE}*.json 2>/dev/null | head -1)
    if [ -n "$STATE_FILE" ]; then
        # Extract the full experiment name from the file
        FULL_EXP_NAME=$(basename "$STATE_FILE" | sed 's/phase_state_//' | sed 's/.json$//')
        echo "Found experiment: $FULL_EXP_NAME"
        echo "Starting monitor for: $FULL_EXP_NAME"
        exec "$MONITOR_SCRIPT" "$FULL_EXP_NAME" "$GPU_ID" "$POLL_INTERVAL"
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done
echo "ERROR: Timeout waiting for phase state file"
exit 1
EOF_WRAPPER

chmod +x monitoring/monitor_wrapper_${EXPERIMENT_NAME}.sh

./monitoring/monitor_wrapper_${EXPERIMENT_NAME}.sh \
  "$EXPERIMENT_NAME" \
  "$GPU_ID" \
  "$POLL_INTERVAL" \
  "./profiling_scripts/monitor_nvidia_smi_phased.sh" &

MONITOR_PID=$!
echo "Monitor wrapper started (PID: $MONITOR_PID)"
sleep 3

# Check monitor is running
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "ERROR: Monitor failed to start!"
    exit 1
fi

echo ""
echo "Starting training with profiling..."
echo ""
echo "IMPORTANT: Make sure you've copied ray_trainer_subphase.py to your verl installation:"
echo "  cp ray_trainer_subphase.py verl/verl/trainer/ppo/ray_trainer.py"
echo ""

# Run training with granularity flag
./profiling_scripts/run_verl_gsm8k_light.sh "$EXPERIMENT_NAME" "$EPOCHS" "$GPU_ID" "$GRANULARITY"

echo ""
echo "Training complete!"

# Cleanup will be called by trap