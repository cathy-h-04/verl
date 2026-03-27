#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-/n/home08/chou/verl_research/logs}"

# Override these if your cluster uses different partition names.
PARTITION_H200="${PARTITION_H200:-gpu_h200}"
PARTITION_A100="${PARTITION_A100:-gpu}"

TIME_LIMIT="${TIME_LIMIT:-00:05:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-4G}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"

mkdir -p "$LOG_ROOT"

submit_probe_job() {
    local gpu_label="$1"
    local partition="$2"
    local job_name="probe_${gpu_label}"

    sbatch \
        --job-name="$job_name" \
        --nodes=1 \
        --ntasks=1 \
        --gres="gpu:${GPUS_PER_JOB}" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEMORY" \
        --time="$TIME_LIMIT" \
        --partition="$partition" \
        --output="${LOG_ROOT}/${job_name}_%j.out" \
        --error="${LOG_ROOT}/${job_name}_%j.err" <<'EOF'
#!/bin/bash
set -euo pipefail

echo "========== Fabric Probe Start =========="
echo "date: $(date)"
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo

echo "$ nvidia-smi -L"
nvidia-smi -L
echo

echo "$ nvidia-smi topo -m"
nvidia-smi topo -m
echo

echo "$ nvidia-smi nvlink --status"
nvidia-smi nvlink --status
echo

echo "$ nvidia-smi -q | sed -n '/Fabric/,/Processes/p'"
nvidia-smi -q | sed -n "/Fabric/,/Processes/p"
echo

echo "========== Fabric Probe End =========="
EOF
}

echo "Submitting H200 probe job on partition: ${PARTITION_H200}"
submit_probe_job "h200" "$PARTITION_H200"

echo "Submitting A100 probe job on partition: ${PARTITION_A100}"
submit_probe_job "a100" "$PARTITION_A100"

echo "Submitted both probe jobs."
echo "Logs will be written under: ${LOG_ROOT}"
