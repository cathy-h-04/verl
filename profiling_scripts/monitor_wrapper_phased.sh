#!/bin/bash
EXPERIMENT_BASE="$1"
GPU_ID="$2"
POLL_INTERVAL="$3"
MONITOR_SCRIPT="$4"

echo "Waiting for phase state file..."
MAX_WAIT=60
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    STATE_FILE=$(ls -t /home/cathxhou/projects/verl_research/monitoring/phase_state_${EXPERIMENT_BASE}*.json 2>/dev/null | head -1)
    if [ -n "$STATE_FILE" ]; then
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
