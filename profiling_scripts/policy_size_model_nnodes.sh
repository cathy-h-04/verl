#!/bin/bash
#
# policy_size_model_nnodes.sh
# Single-run entry point for Slurm array tasks.
#

# Ensure SCRATCH_DIR is set for the cluster
export SCRATCH_DIR="${SCRATCH_DIR:-/n/netscratch/yu_lab/Lab/chou}"
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"
if [ -n "${RAY_ADDRESS:-}" ]; then
    export RAY_ADDRESS
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PROJECT_DIR

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/verl-env}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "WARNING: venv not found at $VENV_PATH (continuing)"
fi
set -o pipefail


PROFILING_DIR="${PROJECT_DIR}/profiling_scripts"
BASE_EXPERIMENT_PREFIX="${BASE_EXPERIMENT_PREFIX:-sweep}"
EPOCHS="${EPOCHS:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-1}"
GRANULARITY="${GRANULARITY:-phase}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
USE_VALIDATION="${USE_VALIDATION:-1}"
VAL_FREQ="${VAL_FREQ:-20}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
RESUME_FROM_CHECKPOINT=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume_path)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    export RESUME_FROM_CHECKPOINT
    echo "Resume from checkpoint: $RESUME_FROM_CHECKPOINT"
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ $# -lt 15 ]; then
    echo "ERROR: Slurm array task requires 15 params."
    echo "Expected: NAME MODEL EPOCHS POLL GRANULARITY POLICY NODES GPUS DATASET VAL VAL_FREQ VAL_SAMPLES TOTAL_STEPS SAVE_FREQ RESUME_PATH"
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen2.5_7b}"
POLICY="${POLICY:-ppo}"

if [ $# -ge 15 ]; then
    BASE_EXPERIMENT_NAME="$1"
    MODEL_NAME="$2"
    EPOCHS="$3"
    POLL_INTERVAL="$4"
    GRANULARITY="$5"
    POLICY="$6"
    NNODES="$7"
    N_GPUS_PER_NODE="$8"
    DATASET_NAME="$9"
    USE_VALIDATION="${10}"
    VAL_FREQ="${11}"
    VAL_MAX_SAMPLES="${12}"
    TOTAL_STEPS="${13}"
    SAVE_FREQ="${14}"
    RESUME_FROM_CHECKPOINT="${15}"
else
    BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_PREFIX}_${DATASET_NAME}_${POLICY}_${MODEL_TAG}_${N_GPUS_PER_NODE}gpn"
fi

if [ "${USE_VALIDATION}" != "1" ] && [ "${USE_VALIDATION}" != "true" ] && [ "${USE_VALIDATION}" != "TRUE" ] && [ "${USE_VALIDATION}" != "yes" ] && [ "${USE_VALIDATION}" != "YES" ]; then
    echo "WARNING: Forcing validation-only runs. Overriding USE_VALIDATION=$USE_VALIDATION to 1."
    USE_VALIDATION=1
fi
TRAIN_SCRIPT="run_verl_train_val.sh"

echo ""
echo "=== Running: ${BASE_EXPERIMENT_NAME} ==="
echo "Model: $MODEL_NAME"
echo "Policy: $POLICY"
echo "Nodes: $NNODES (gpus per node: $N_GPUS_PER_NODE)"
echo "Validation: $USE_VALIDATION"
echo "Val freq: $VAL_FREQ"
echo "Val max samples: ${VAL_MAX_SAMPLES:-auto}"
echo "Dataset: $DATASET_NAME"

TRAIN_SCRIPT="$TRAIN_SCRIPT" USE_VALIDATION="$USE_VALIDATION" VAL_FREQ="$VAL_FREQ" VAL_MAX_SAMPLES="$VAL_MAX_SAMPLES" TOTAL_STEPS="$TOTAL_STEPS" SAVE_FREQ="$SAVE_FREQ" RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
    bash "${PROFILING_DIR}/run_with_phase_monitoring.sh" \
    "$BASE_EXPERIMENT_NAME" \
    "$EPOCHS" \
    "$POLL_INTERVAL" \
    "$GRANULARITY" \
    "$MODEL_NAME" \
    "$POLICY" \
    "$NNODES" \
    "$N_GPUS_PER_NODE" \
    "$DATASET_NAME"
