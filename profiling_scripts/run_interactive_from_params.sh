#!/usr/bin/env bash
set -euo pipefail

# Interactive helper:
# - Intended to be run inside an existing salloc allocation.
# - Reads one run from a JSON runs file and launches the run directly (no sbatch).
#
# Usage:
#   run_interactive_from_params.sh --config <runs.json> --line <n>
#   run_interactive_from_params.sh --config <runs.json> --name <run_name>
#
# Notes:
# - Intended for single-node interactive runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_FILE=""
LINE=""
NAME_FILTER=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|--runs)
            RUNS_FILE="$2"
            shift 2
            ;;
        --line)
            LINE="$2"
            shift 2
            ;;
        --name)
            NAME_FILTER="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ -z "$RUNS_FILE" ]; then
    RUNS_FILE="${SCRIPT_DIR}/runs.json"
fi

if [ ! -f "$RUNS_FILE" ]; then
    echo "ERROR: Runs file not found: $RUNS_FILE"
    exit 1
fi
if [[ "$RUNS_FILE" != *.json ]]; then
    echo "ERROR: Runs file must be JSON: $RUNS_FILE"
    exit 1
fi

if [ -z "$LINE" ] && [ -z "$NAME_FILTER" ]; then
    RUN_COUNT=$(python3 "${SCRIPT_DIR}/config_utils.py" count --config "$RUNS_FILE")
    if [ "$RUN_COUNT" -eq 1 ]; then
        LINE=1
    else
        echo "ERROR: Provide --line or --name to select a run (found $RUN_COUNT runs)."
        exit 1
    fi
fi

CONFIG_EXPORTS=$(python3 "${SCRIPT_DIR}/config_utils.py" export --config "$RUNS_FILE" ${LINE:+--index "$LINE"} ${NAME_FILTER:+--name "$NAME_FILTER"})
eval "$CONFIG_EXPORTS"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-${RUN_CONFIG_NAME:-}}"
NAME="${BASE_EXPERIMENT_NAME}"

# Environment setup (mirrors ray_on_slurm.slurm without sbatch/srun)
source /n/home08/chou/verl_research/verl-env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:/n/home08/chou/verl_research"
export SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/n/netscratch}/yu_lab/Lab/chou}"
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"
# Prefer node-local scratch for temp/cache to avoid NFS stale file handles.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-${SLURM_TMPDIR:-/scratch/${USER}}}"
if [ ! -d "$LOCAL_SCRATCH" ]; then
    LOCAL_SCRATCH="/tmp/${USER}"
fi
mkdir -p "$LOCAL_SCRATCH"
export TMPDIR="${TMPDIR:-${LOCAL_SCRATCH}/tmp}"
mkdir -p "$TMPDIR"

export RAY_TMPDIR="${RAY_TMPDIR:-${LOCAL_SCRATCH}/ray}"
mkdir -p "$RAY_TMPDIR"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_SCRATCH}/xdg_cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${LOCAL_SCRATCH}/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_SCRATCH}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_SCRATCH}/triton}"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export HOME_STORAGE="/n/home08/chou/verl_research/results"
mkdir -p "$HOME_STORAGE"
export HOME_LOGS="/n/home08/chou/verl_research/logs"
mkdir -p "$HOME_LOGS"
export RAY_MANAGED_BY_SLURM=0


echo "=== Preflight Checks ==="
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "SLURM_JOB_ID detected: ${SLURM_JOB_ID} (interactive allocation)"
else
    echo "WARNING: SLURM_JOB_ID not set. Ensure you are already on a compute node."
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. GPU driver not available in this environment."
    exit 1
fi
if [ ! -d "/n/home08/chou/verl_research/verl-env" ]; then
    echo "ERROR: venv missing at /n/home08/chou/verl_research/verl-env"
    exit 1
fi
if ! python -c "import torch; print(torch.cuda.is_available())" >/dev/null 2>&1; then
    echo "ERROR: Python or torch not functional in venv."
    exit 1
fi
if ! python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
print(torch.cuda.get_device_name(0))
PY
then
    echo "ERROR: CUDA not available in this allocation."
    exit 1
fi
echo "Preflight OK."

ROLLOUT_N="${ROLLOUT_N:-4}"

if [ "${SAVE_FREQ:-}" = "-1" ]; then
    MOVE_CHECKPOINTS="false"
else
    MOVE_CHECKPOINTS="true"
fi
export MOVE_CHECKPOINTS

POLICY_NORMALIZED="$(echo "${POLICY:-ppo}" | tr '[:upper:]' '[:lower:]')"
TRAIN_SCRIPT="run_verl_train_nonval.sh"
if [ "$POLICY_NORMALIZED" = "sft" ]; then
    TRAIN_SCRIPT="run_verl_train_sft_nonval.sh"
fi
case "${USE_VALIDATION:-}" in
    1|true|TRUE|yes|YES)
        if [ "$POLICY_NORMALIZED" = "sft" ]; then
            TRAIN_SCRIPT="run_verl_train_sft_val.sh"
        else
            TRAIN_SCRIPT="run_verl_train_val.sh"
        fi
        ;;
esac
export TRAIN_SCRIPT

NAME="${BASE_EXPERIMENT_NAME:-${NAME:-}}"
if [ -z "$NAME" ]; then
    echo "ERROR: Missing base experiment name."
    exit 1
fi

# Ensure CUDA_VISIBLE_DEVICES is set for interactive allocations.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="0"
    export CUDA_VISIBLE_DEVICES
fi

echo "Running Slurm-path dry-run check..."
TASK_ID="${RUN_CONFIG_INDEX:-1}"
SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-$(hostname)}"
SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$(nproc)}"
SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${N_GPUS_PER_NODE:-1}}"
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SUBMIT_SCRIPT_DIR="$SCRIPT_DIR" \
RUNS_FILE="$RUNS_FILE" \
SLURM_ARRAY_TASK_ID="$TASK_ID" \
SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST" \
SLURM_JOB_NUM_NODES="$SLURM_JOB_NUM_NODES" \
SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
SLURM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE" \
SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR" \
DRY_RUN=1 \
PREFLIGHT_ONLY=0 \
bash "${SCRIPT_DIR}/ray_on_slurm.slurm"

echo "Launching interactive run..."
# Mark start time to find the exact monitoring dir created by this run.
START_MARKER="$(mktemp)"
touch "$START_MARKER"
bash "${SCRIPT_DIR}/run_with_phase_monitoring.sh"

# Post-run migration (best-effort) to HOME_STORAGE, mirroring ray_on_slurm.slurm
case "${USE_VALIDATION:-}" in
    1|true|TRUE|yes|YES)
        MONITOR_ROOT="${SCRATCH_DIR}/monitoring_val"
        DEST_ROOT="${HOME_STORAGE}/monitoring_val"
        ;;
    *)
        MONITOR_ROOT="${SCRATCH_DIR}/monitoring"
        DEST_ROOT="${HOME_STORAGE}/monitoring"
        ;;
esac

LATEST_MONITOR_DIR=""
EXPERIMENT_NAME=""
if [ -d "$MONITOR_ROOT" ]; then
    EXP_FILE=$(
        find "$MONITOR_ROOT" -type f -name experiment_name.txt -newer "$START_MARKER" -printf '%T@ %p\n' 2>/dev/null \
            | sort -n \
            | tail -1 \
            | awk '{print $2}'
    )
    if [ -n "$EXP_FILE" ] && [ -f "$EXP_FILE" ]; then
        LATEST_MONITOR_DIR="$(dirname "$EXP_FILE")"
        EXPERIMENT_NAME="$(cat "$EXP_FILE" 2>/dev/null || true)"
    else
        LATEST_MONITOR_DIR=$(ls -td "${MONITOR_ROOT}/${NAME}_"* 2>/dev/null | head -1 || true)
        if [ -n "$LATEST_MONITOR_DIR" ]; then
            EXPERIMENT_NAME="$(basename "$LATEST_MONITOR_DIR")"
        fi
    fi
fi
rm -f "$START_MARKER"

if [ -n "$LATEST_MONITOR_DIR" ] && [ -d "$LATEST_MONITOR_DIR" ] && [ -n "$EXPERIMENT_NAME" ]; then
    echo "Migrating results for ${EXPERIMENT_NAME}..."
    mkdir -p "$DEST_ROOT"

    # Ensure config metadata is present in the monitoring dir.
    if [ -n "${RUN_CONFIG_JSON:-}" ] && [ ! -f "${LATEST_MONITOR_DIR}/run_config.json" ]; then
        LATEST_MONITOR_DIR="$LATEST_MONITOR_DIR" python3 - <<'PY'
import json
import os

raw = os.environ.get("RUN_CONFIG_JSON", "")
out_path = os.environ.get("LATEST_MONITOR_DIR", "") + "/run_config.json"
if raw and out_path:
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {"raw": raw}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
PY
    elif [ -n "${RUN_CONFIG_PATH:-}" ] && [ -f "${RUN_CONFIG_PATH:-}" ] && [ ! -f "${LATEST_MONITOR_DIR}/run_config.json" ]; then
        cp "${RUN_CONFIG_PATH}" "${LATEST_MONITOR_DIR}/run_config.json"
    fi
    if [ -n "${SLURM_CONFIG_PATH:-}" ] && [ -f "${SLURM_CONFIG_PATH:-}" ] && [ ! -f "${LATEST_MONITOR_DIR}/slurm_config.json" ]; then
        cp "${SLURM_CONFIG_PATH}" "${LATEST_MONITOR_DIR}/slurm_config.json"
    fi

    rsync -avz "${LATEST_MONITOR_DIR}/" "${DEST_ROOT}/${EXPERIMENT_NAME}/" || true

    if [ "${MOVE_CHECKPOINTS}" = "true" ] && [ -d "${SCRATCH_DIR}/checkpoints/${EXPERIMENT_NAME}" ]; then
        rsync -avz "${SCRATCH_DIR}/checkpoints/${EXPERIMENT_NAME}/" "${DEST_ROOT}/${EXPERIMENT_NAME}/checkpoints/" || true
    fi

    if [ -f "${SCRATCH_DIR}/logs/${EXPERIMENT_NAME}.log" ]; then
        rsync -avz "${SCRATCH_DIR}/logs/${EXPERIMENT_NAME}.log" "$HOME_LOGS/" || true
    fi
else
    echo "WARNING: No monitoring directory found under ${MONITOR_ROOT} for ${NAME}_*"
fi
