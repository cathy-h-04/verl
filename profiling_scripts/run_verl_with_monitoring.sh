#!/bin/bash
# ================================================================
# Verl Training with GPU Monitoring
# Launches training and monitoring in parallel processes
# ================================================================

set -euo pipefail

# -------------------- Configuration --------------------
PROJECT_DIR=~/projects/verl_research
EXPERIMENT_NAME="${1:-gsm8k_experiment}"
TOTAL_EPOCHS="${2:-3}"
GPU_ID="${3:-1}"
POLL_INTERVAL="${4:-1}"  # nvidia-smi polling interval in seconds

# -------------------- Setup --------------------
cd "$PROJECT_DIR"

TRAINING_SCRIPT="${5:-${PROJECT_DIR}/run_verl_gsm8k_light.sh}"  # Allow custom script as 5th arg
MONITOR_SCRIPT="${PROJECT_DIR}/monitor_nvidia_smi.sh"

# Check if scripts exist
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo "ERROR: Monitor script not found: $MONITOR_SCRIPT"
    exit 1
fi

# Make scripts executable
chmod +x "$TRAINING_SCRIPT" "$MONITOR_SCRIPT"

# -------------------- Display Configuration --------------------
echo "========================================"
echo "Verl Training + GPU Monitoring"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Epochs: $TOTAL_EPOCHS"
echo "GPU ID: $GPU_ID"
echo "Monitor Interval: ${POLL_INTERVAL}s"
echo "========================================"
echo ""

# -------------------- Launch Monitoring --------------------
echo "Starting GPU monitor in background..."
"$MONITOR_SCRIPT" "$EXPERIMENT_NAME" "$POLL_INTERVAL" "$GPU_ID" &
MONITOR_PID=$!
echo "✓ Monitor started (PID: $MONITOR_PID)"
echo ""

# -------------------- Launch Training --------------------
echo "Starting Verl training immediately..."
echo ""

# Run training and capture exit code
set +e
"$TRAINING_SCRIPT" "$EXPERIMENT_NAME" "$TOTAL_EPOCHS" "$GPU_ID"
TRAINING_EXIT_CODE=$?
set -e

# -------------------- Cleanup --------------------
echo ""
echo "========================================"
echo "Training completed. Stopping monitor..."
echo "========================================"

# Stop the monitor process
if kill -0 $MONITOR_PID 2>/dev/null; then
    kill -INT $MONITOR_PID
    wait $MONITOR_PID 2>/dev/null || true
    echo "✓ Monitor stopped"
fi

# -------------------- Summary --------------------
echo ""
echo "========================================"
echo "Experiment Complete"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Training Exit Code: $TRAINING_EXIT_CODE"
echo ""
echo "Output Locations:"
echo "  - Training checkpoints: ${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}/"
echo "  - Training logs: ${PROJECT_DIR}/logs/"
echo "  - GPU metrics: ${PROJECT_DIR}/monitoring/"
echo "========================================"
echo ""

# Find the most recent monitoring CSV
LATEST_CSV=$(ls -t "${PROJECT_DIR}/monitoring/${EXPERIMENT_NAME}_gpu_metrics_"*.csv 2>/dev/null | head -1)
if [ -n "$LATEST_CSV" ]; then
    echo "GPU Metrics Summary:"
    echo "-------------------"
    
    # Calculate statistics using awk
    awk -F',' 'NR>1 {
        sum_gpu+=$11; sum_mem+=$10; sum_power+=$5; sum_temp+=$4; count++
    } 
    END {
        if(count>0) {
            printf "  Samples collected: %d\n", count
            printf "  Avg GPU Utilization: %.1f%%\n", sum_gpu/count
            printf "  Avg Memory Utilization: %.1f%%\n", sum_mem/count
            printf "  Avg Power Draw: %.1f W\n", sum_power/count
            printf "  Avg Temperature: %.1f°C\n", sum_temp/count
        }
    }' "$LATEST_CSV"
    
    echo ""
    echo "Full metrics saved to: $LATEST_CSV"
fi

echo ""
exit $TRAINING_EXIT_CODE