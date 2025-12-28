#!/bin/bash

PROJECT_DIR=~/projects/verl_research
EXPERIMENT_NAME="${1:-test}"
POLL_INTERVAL="${2:-1}"
GPU_ID="${3:-1}"

MONITOR_DIR="${PROJECT_DIR}/monitoring"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${MONITOR_DIR}/${EXPERIMENT_NAME}_gpu_metrics_${TIMESTAMP}.csv"

mkdir -p "$MONITOR_DIR"

# CSV Header
echo "timestamp,elapsed_seconds,gpu_id,gpu_name,temperature_c,power_draw_w,power_limit_w,memory_used_mb,memory_total_mb,memory_util_percent,gpu_util_percent,sm_clock_mhz,mem_clock_mhz" > "$OUTPUT_CSV"

echo "========================================"
echo "NVIDIA GPU Monitor"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPU ID: $GPU_ID"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Output: $OUTPUT_CSV"
echo "========================================"
echo ""

START_TIME=$(date +%s)
COUNT=0

trap 'echo ""; echo "✓ Stopped. Collected $COUNT samples in $OUTPUT_CSV"; exit 0' INT TERM

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    TS=$(date +%Y-%m-%d_%H:%M:%S)
    
    # Query all metrics in one call
    METRICS=$(nvidia-smi -i "$GPU_ID" \
        --query-gpu=name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.memory,utilization.gpu,clocks.sm,clocks.mem \
        --format=csv,noheader,nounits)
    
    # Parse using awk and trim whitespace
    GPU_NAME=$(echo "$METRICS" | awk -F',' '{print $1}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    TEMP=$(echo "$METRICS" | awk -F',' '{print $2}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    POWER_DRAW=$(echo "$METRICS" | awk -F',' '{print $3}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    POWER_LIMIT=$(echo "$METRICS" | awk -F',' '{print $4}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    MEM_USED=$(echo "$METRICS" | awk -F',' '{print $5}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    MEM_TOTAL=$(echo "$METRICS" | awk -F',' '{print $6}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    MEM_UTIL=$(echo "$METRICS" | awk -F',' '{print $7}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    GPU_UTIL=$(echo "$METRICS" | awk -F',' '{print $8}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    SM_CLOCK=$(echo "$METRICS" | awk -F',' '{print $9}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    MEM_CLOCK=$(echo "$METRICS" | awk -F',' '{print $10}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # Write to CSV
    echo "$TS,$ELAPSED,$GPU_ID,$GPU_NAME,$TEMP,$POWER_DRAW,$POWER_LIMIT,$MEM_USED,$MEM_TOTAL,$MEM_UTIL,$GPU_UTIL,$SM_CLOCK,$MEM_CLOCK" >> "$OUTPUT_CSV"
    
    COUNT=$((COUNT + 1))
    
    # Progress every 10 samples
    if [ $((COUNT % 10)) -eq 0 ]; then
        echo "[Sample $COUNT @ ${ELAPSED}s] GPU: ${GPU_UTIL}% | Mem: ${MEM_USED}MB (${MEM_UTIL}%) | Power: ${POWER_DRAW}W | Temp: ${TEMP}°C"
    fi
    
    sleep "$POLL_INTERVAL"
done