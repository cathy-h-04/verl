#!/bin/bash
#
# monitor_nvidia_smi_phased.sh
# Enhanced GPU monitoring with phase tracking
#
# Usage: ./monitor_nvidia_smi_phased.sh <experiment_name> <gpu_id> <poll_interval_seconds>

set -e

EXPERIMENT_NAME="${1:-default_experiment}"
GPU_ID="${2:-0}"
POLL_INTERVAL="${3:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/home/cathxhou/projects/verl_research"
MONITOR_DIR="$PROJECT_DIR/monitoring"
mkdir -p "$MONITOR_DIR"
OUTPUT_FILE="$MONITOR_DIR/${EXPERIMENT_NAME}_phased_${TIMESTAMP}.csv"


# Create Python helper to read phase state
cat > /tmp/read_phase_state_${EXPERIMENT_NAME}.py << 'PYTHON_HELPER'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/cathxhou/projects/verl_research')

from profiling_scripts.verl_phase_profiler import PhaseReader

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
echo "GPU ID: $GPU_ID"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Write CSV header
echo "timestamp,elapsed_seconds,phase_id,phase_name,iteration,gpu_id,gpu_name,temperature_c,power_draw_w,power_limit_w,memory_used_mb,memory_total_mb,memory_util_percent,gpu_util_percent,sm_clock_mhz,mem_clock_mhz" > "$OUTPUT_FILE"

START_TIME=$(date +%s)

# Main monitoring loop
while true; do
    TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
    ELAPSED=$(($(date +%s) - START_TIME))
    
    # Read current phase state
    PHASE_STATE=$(python3 /tmp/read_phase_state_${EXPERIMENT_NAME}.py "$EXPERIMENT_NAME" 2>/dev/null || echo "0,idle,0")
    PHASE_ID=$(echo "$PHASE_STATE" | cut -d',' -f1)
    PHASE_NAME=$(echo "$PHASE_STATE" | cut -d',' -f2)
    ITERATION=$(echo "$PHASE_STATE" | cut -d',' -f3)
    
    # Query nvidia-smi
    GPU_STATS=$(nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.memory,utilization.gpu,clocks.current.sm,clocks.current.memory \
        --format=csv,noheader,nounits \
        -i "$GPU_ID" 2>/dev/null || echo "$GPU_ID,N/A,0,0,0,0,0,0,0,0,0")
    
    # Parse GPU stats
    IFS=',' read -r gpu_idx gpu_name temp power_draw power_limit mem_used mem_total mem_util gpu_util sm_clock mem_clock <<< "$GPU_STATS"
    
    # Trim whitespace
    gpu_name=$(echo "$gpu_name" | xargs)
    temp=$(echo "$temp" | xargs)
    power_draw=$(echo "$power_draw" | xargs)
    power_limit=$(echo "$power_limit" | xargs)
    mem_used=$(echo "$mem_used" | xargs)
    mem_total=$(echo "$mem_total" | xargs)
    mem_util=$(echo "$mem_util" | xargs)
    gpu_util=$(echo "$gpu_util" | xargs)
    sm_clock=$(echo "$sm_clock" | xargs)
    mem_clock=$(echo "$mem_clock" | xargs)
    
    # Write to CSV
    echo "$TIMESTAMP,$ELAPSED,$PHASE_ID,$PHASE_NAME,$ITERATION,$GPU_ID,$gpu_name,$temp,$power_draw,$power_limit,$mem_used,$mem_total,$mem_util,$gpu_util,$sm_clock,$mem_clock" >> "$OUTPUT_FILE"
    
    sleep "$POLL_INTERVAL"
done