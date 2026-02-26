#!/bin/bash
#
# monitor_nvidia_smi_phased.sh
# Enhanced GPU monitoring with phase tracking
#
# Usage: ./monitor_nvidia_smi_phased.sh <experiment_name> <gpu_id> <poll_interval_seconds> [nnodes] [gpus_per_node]

set -e

# Ensure SCRATCH_DIR is set for the cluster
export SCRATCH_DIR="${SCRATCH_DIR:-/n/netscratch/yu_lab/Lab/chou}"
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"
if [ -n "${RAY_ADDRESS:-}" ]; then
    export RAY_ADDRESS
fi

EXPERIMENT_NAME="${1:-default_experiment}"
GPU_ID_ARG="${2:-0}"
POLL_INTERVAL="${3:-1}"
NNODES="${4:-na}"
N_GPUS_PER_NODE="${5:-na}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export VERL_PROFILER_DIR="${VERL_PROFILER_DIR:-$SCRIPT_DIR}"
MONITOR_DIR="${MONITORING_DIR:-$SCRATCH_DIR/monitoring}"
if [[ "$MONITOR_DIR" == */"$EXPERIMENT_NAME" ]]; then
    EXPERIMENT_MONITOR_DIR="$MONITOR_DIR"
else
    EXPERIMENT_MONITOR_DIR="${MONITOR_DIR}/${EXPERIMENT_NAME}"
fi
mkdir -p "$EXPERIMENT_MONITOR_DIR"
OUTPUT_FILE="$EXPERIMENT_MONITOR_DIR/${EXPERIMENT_NAME}_phased_${TIMESTAMP}.csv"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-$SCRATCH_DIR/logs/${EXPERIMENT_NAME}.log}"


# Create Python helper to read phase state
cat > /tmp/read_phase_state_${EXPERIMENT_NAME}.py << 'PYTHON_HELPER'
#!/usr/bin/env python3
import sys
import os
project_dir = os.environ.get("PROJECT_DIR", "")
if project_dir:
    sys.path.insert(0, project_dir)

profiler_dir = os.environ.get("VERL_PROFILER_DIR", "")
if profiler_dir:
    sys.path.insert(0, profiler_dir)

from verl_subphase_profiler import PhaseReader

def main():
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "default_experiment"
    
    try:
        reader = PhaseReader(experiment_name=experiment_name)
        state = reader.get_current_phase()
        # Output: phase_id,phase_name,iteration
        print(f"{state['phase_id']},{state['phase_name']},{state['iteration']}")
    except Exception:
        # If can't read phase state, output idle
        print("0,idle,0")

if __name__ == "__main__":
    main()
PYTHON_HELPER

chmod +x /tmp/read_phase_state_${EXPERIMENT_NAME}.py

echo "=== Phase-Aware GPU Monitor Started ==="
echo "Experiment: $EXPERIMENT_NAME"
GPU_IDS=""
GPU_ID_SOURCE="CUDA_VISIBLE_DEVICES"

parse_cvd_list() {
    python3 - <<'PY'
import os
import re
import sys

s = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not s:
    sys.exit(1)

parts = re.split(r"[,\s]+", s)
ids = []
for part in parts:
    if part:
        ids.append(part)

print(" ".join(ids))
PY
}

GPU_IDS="$(parse_cvd_list)" || true
if [ -z "$GPU_IDS" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty. Refusing to profile unknown GPUs."
    exit 1
fi

echo "GPU ID(s): $GPU_IDS (source: $GPU_ID_SOURCE)"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Nodes: $NNODES (gpus per node: $N_GPUS_PER_NODE)"
echo "Output: $OUTPUT_FILE"
echo "Training log: $TRAIN_LOG_FILE"
echo "=========================================="

# Detect whether utilization.sm is supported by this nvidia-smi
SM_UTIL_SUPPORTED=1
TEST_GPU_ID=""
for TEST_ID in $GPU_IDS; do
    TEST_GPU_ID="$TEST_ID"
    break
done
if [ -n "$TEST_GPU_ID" ]; then
    if ! nvidia-smi --query-gpu=utilization.sm --format=csv,noheader,nounits -i "$TEST_GPU_ID" >/dev/null 2>&1; then
        SM_UTIL_SUPPORTED=0
    fi
fi

# Write CSV header (conditionally include sm_util_percent)
if [ "$SM_UTIL_SUPPORTED" -eq 1 ]; then
    echo "timestamp,elapsed_seconds,phase_id,phase_name,iteration,nnodes,gpus_per_node,gpu_id,gpu_name,temperature_c,power_draw_w,power_limit_w,enforced_power_limit_w,pstate,clocks_throttle_active,clocks_throttle_gpu_idle,clocks_throttle_sw_power_cap,memory_used_mb,memory_total_mb,memory_util_percent,gpu_util_percent,sm_util_percent,sm_clock_mhz,mem_clock_mhz" > "$OUTPUT_FILE"
else
    echo "timestamp,elapsed_seconds,phase_id,phase_name,iteration,nnodes,gpus_per_node,gpu_id,gpu_name,temperature_c,power_draw_w,power_limit_w,enforced_power_limit_w,pstate,clocks_throttle_active,clocks_throttle_gpu_idle,clocks_throttle_sw_power_cap,memory_used_mb,memory_total_mb,memory_util_percent,gpu_util_percent,sm_clock_mhz,mem_clock_mhz" > "$OUTPUT_FILE"
fi

START_TIME=$(date +%s)

# Wait for training log to exist (helps sync on Lustre)
if [ ! -f "$TRAIN_LOG_FILE" ]; then
    echo "Waiting for training log..."
    while [ ! -f "$TRAIN_LOG_FILE" ]; do
        sleep 1
    done
fi

# Main monitoring loop
while true; do
    TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
    ELAPSED=$(($(date +%s) - START_TIME))
    
    # Read current phase state
    PHASE_STATE=$(python3 /tmp/read_phase_state_${EXPERIMENT_NAME}.py "$EXPERIMENT_NAME" 2>/dev/null || echo "0,idle,0")
    PHASE_ID=$(echo "$PHASE_STATE" | cut -d',' -f1)
    PHASE_NAME=$(echo "$PHASE_STATE" | cut -d',' -f2)
    ITERATION=$(echo "$PHASE_STATE" | cut -d',' -f3)
    
    for GPU_ID in $GPU_IDS; do
        # Query nvidia-smi
        if [ "$SM_UTIL_SUPPORTED" -eq 1 ]; then
            GPU_STATS=$(nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,enforced.power.limit,pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_power_cap,memory.used,memory.total,utilization.memory,utilization.gpu,utilization.sm,clocks.current.sm,clocks.current.memory \
                --format=csv,noheader,nounits \
                -i "$GPU_ID" 2>/dev/null || echo "$GPU_ID,N/A,0,0,0,0,N/A,0,0,0,0,0,0,0,0,0,0")
        else
            GPU_STATS=$(nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,enforced.power.limit,pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_power_cap,memory.used,memory.total,utilization.memory,utilization.gpu,clocks.current.sm,clocks.current.memory \
                --format=csv,noheader,nounits \
                -i "$GPU_ID" 2>/dev/null || echo "$GPU_ID,N/A,0,0,0,0,N/A,0,0,0,0,0,0,0,0,0")
        fi

        # Parse GPU stats
        if [ "$SM_UTIL_SUPPORTED" -eq 1 ]; then
            IFS=',' read -r gpu_idx gpu_name temp power_draw power_limit enforced_power_limit pstate throttle_active throttle_gpu_idle throttle_sw_power_cap mem_used mem_total mem_util gpu_util sm_util sm_clock mem_clock <<< "$GPU_STATS"
        else
            IFS=',' read -r gpu_idx gpu_name temp power_draw power_limit enforced_power_limit pstate throttle_active throttle_gpu_idle throttle_sw_power_cap mem_used mem_total mem_util gpu_util sm_clock mem_clock <<< "$GPU_STATS"
            sm_util=""
        fi
        
        # Trim whitespace
        gpu_name=$(echo "$gpu_name" | xargs)
        temp=$(echo "$temp" | xargs)
        power_draw=$(echo "$power_draw" | xargs)
        power_limit=$(echo "$power_limit" | xargs)
        enforced_power_limit=$(echo "$enforced_power_limit" | xargs)
        pstate=$(echo "$pstate" | xargs)
        throttle_active=$(echo "$throttle_active" | xargs)
        throttle_gpu_idle=$(echo "$throttle_gpu_idle" | xargs)
        throttle_sw_power_cap=$(echo "$throttle_sw_power_cap" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        mem_util=$(echo "$mem_util" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        sm_util=$(echo "$sm_util" | xargs)
        sm_clock=$(echo "$sm_clock" | xargs)
        mem_clock=$(echo "$mem_clock" | xargs)

        # Write to CSV
        if [ "$SM_UTIL_SUPPORTED" -eq 1 ]; then
            echo "$TIMESTAMP,$ELAPSED,$PHASE_ID,$PHASE_NAME,$ITERATION,$NNODES,$N_GPUS_PER_NODE,$GPU_ID,$gpu_name,$temp,$power_draw,$power_limit,$enforced_power_limit,$pstate,$throttle_active,$throttle_gpu_idle,$throttle_sw_power_cap,$mem_used,$mem_total,$mem_util,$gpu_util,$sm_util,$sm_clock,$mem_clock" >> "$OUTPUT_FILE"
        else
            echo "$TIMESTAMP,$ELAPSED,$PHASE_ID,$PHASE_NAME,$ITERATION,$NNODES,$N_GPUS_PER_NODE,$GPU_ID,$gpu_name,$temp,$power_draw,$power_limit,$enforced_power_limit,$pstate,$throttle_active,$throttle_gpu_idle,$throttle_sw_power_cap,$mem_used,$mem_total,$mem_util,$gpu_util,$sm_clock,$mem_clock" >> "$OUTPUT_FILE"
        fi
    done
    
    sleep "$POLL_INTERVAL"
done
