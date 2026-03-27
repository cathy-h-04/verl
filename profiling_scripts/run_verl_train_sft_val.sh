#!/bin/bash
# ================================================================
# Verl SFT Training Script - Validation mode (Env-driven)
# Mirrors run_verl_train_val.sh with SFT trainer entrypoint.
# ================================================================

set -euo pipefail

# Ensure SCRATCH_DIR is set for the cluster
export SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/n/netscratch}/yu_lab/Lab/chou}"
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"

# Prefer node-local scratch for temp/cache to avoid NFS stale file handles.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-${SLURM_TMPDIR:-/tmp/${USER}}}"
mkdir -p "$LOCAL_SCRATCH"
export TMPDIR="${TMPDIR:-${LOCAL_SCRATCH}/tmp}"
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_SCRATCH}/xdg_cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${LOCAL_SCRATCH}/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_SCRATCH}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_SCRATCH}/triton}"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# -------------------- CLI Overrides --------------------
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
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
    if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
        echo "FATAL: resume_path does not exist: $RESUME_FROM_CHECKPOINT"
        exit 1
    fi
fi

# -------------------- Configuration --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PROJECT_DIR
EXPERIMENT_NAME="${1:-sft_val_profile}"
TOTAL_EPOCHS="${2:-1}"
GRANULARITY="${3:-phase}"  # 'phase' or 'operation'
MODEL_NAME="${4:-Qwen/Qwen2.5-7B-Instruct}"
POLICY="${5:-sft}"
NNODES="${6:-1}"
N_GPUS_PER_NODE="${7:-4}"
DATASET_NAME="${8:-gsm8k}"
VAL_FREQ="${VAL_FREQ:-${9:-20}}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-${10:-}}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
export EXPERIMENT_NAME

if [ "${POLICY,,}" != "sft" ]; then
    echo "ERROR: run_verl_train_sft_val.sh expects POLICY=sft (got '$POLICY')"
    exit 1
fi
if [ "$NNODES" != "1" ]; then
    echo "ERROR: SFT profiling launcher currently supports single-node runs only (NNODES=$NNODES)."
    exit 1
fi
if [[ "$GRANULARITY" != "phase" && "$GRANULARITY" != "operation" ]]; then
    echo "ERROR: GRANULARITY must be 'phase' or 'operation' (got '$GRANULARITY')."
    exit 1
fi

if ! [[ "$VAL_FREQ" =~ ^[0-9]+$ ]] || [ "$VAL_FREQ" -le 0 ]; then
    echo "ERROR: VAL_FREQ must be a positive integer for validation runs (got '$VAL_FREQ')."
    exit 1
fi

# SFT knobs; map to existing profiling env where possible.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
ENABLE_GRAD_CHECKPOINTING="${ENABLE_GRAD_CHECKPOINTING:-true}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-${ROLLOUT_MAX_MODEL_LEN:-2048}}"
SFT_TRUNCATION="${SFT_TRUNCATION:-right}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"

# -------------------- Environment Setup --------------------
cd "$PROJECT_DIR"

if [ ! -d "verl-env" ]; then
    echo "ERROR: verl-env not found. Please create it first."
    exit 1
fi
source verl-env/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}/verl"
export PYTHONUNBUFFERED=1

TOKEN_SCRIPT="${SCRIPT_DIR}/token.sh"
if [ -f "$TOKEN_SCRIPT" ]; then
    if ! bash "$TOKEN_SCRIPT"; then
        echo "WARNING: token script failed: $TOKEN_SCRIPT (continuing)"
    fi
else
    echo "WARNING: token script not found: $TOKEN_SCRIPT (continuing)"
fi

MONITORING_DIR="${MONITORING_DIR:-${SCRATCH_DIR}/monitoring_val/${EXPERIMENT_NAME}}"
export VERL_FILE_LOGGER_ROOT="$(dirname "$MONITORING_DIR")"
export VERL_FILE_LOGGER_PATH="${MONITORING_DIR}/${EXPERIMENT_NAME}.jsonl"
mkdir -p "$MONITORING_DIR"

# -------------------- Dataset Setup --------------------
SFT_DATA_ARGS=()
case "$DATASET_NAME" in
    gsm8k)
        DATA_DIR="${SCRATCH_DIR}/data/gsm8k"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_FILE="${DATA_DIR}/test.parquet"
        PREPROCESS_CMD=(python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR")
        # Reuse existing gsm8k parquet schema through dict-key extraction.
        SFT_DATA_ARGS+=("data.prompt_key=extra_info")
        SFT_DATA_ARGS+=("data.prompt_dict_keys=['question']")
        SFT_DATA_ARGS+=("data.response_key=extra_info")
        SFT_DATA_ARGS+=("+data.response_dict_keys=['answer']")
        ;;
    rlhf-ff)
        RLHF_ROOT="${SCRATCH_DIR}/data/full_hh_rlhf"
        DATA_DIR="${RLHF_ROOT}/sft"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_FILE="${DATA_DIR}/test.parquet"
        PREPROCESS_CMD=(python3 examples/data_preprocess/full_hh_rlhf.py --split sft --local_save_dir "$RLHF_ROOT")
        # full_hh_rlhf SFT parquet schema uses prompt/response columns.
        SFT_DATA_ARGS+=("data.prompt_key=prompt")
        SFT_DATA_ARGS+=("data.response_key=response")
        ;;
    *)
        echo "ERROR: Unsupported dataset '$DATASET_NAME' (supported: gsm8k, rlhf-ff)"
        exit 1
        ;;
esac

# -------------------- Directory Setup --------------------
OUTPUT_DIR="${SCRATCH_DIR}/checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="${SCRATCH_DIR}/logs"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# -------------------- Data Preparation --------------------
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Preparing dataset ($DATASET_NAME)..."
    "${PREPROCESS_CMD[@]}"
fi
if [ "$DATASET_NAME" = "rlhf-ff" ] && [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: ${VAL_FILE} is required for rlhf-ff SFT validation. Please regenerate dataset split."
    exit 1
fi
if [ "$DATASET_NAME" != "rlhf-ff" ] && [ ! -f "$VAL_FILE" ] && [ -f "$TRAIN_FILE" ]; then
    echo "WARNING: ${VAL_FILE} not found; falling back to ${TRAIN_FILE}."
    VAL_FILE="$TRAIN_FILE"
fi
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: Missing required dataset files: train=$TRAIN_FILE val=$VAL_FILE"
    exit 1
fi

# -------------------- Training --------------------
LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"

echo "========================================"
echo "Starting SFT Training (validation)"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Validation Frequency: $VAL_FREQ"
echo "Nodes: $NNODES (gpus per node: $N_GPUS_PER_NODE)"
echo "Logs: $LOG_FILE"
echo "========================================"

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    RESUME_ARGS+=("trainer.resume_mode=resume_path")
    RESUME_ARGS+=("trainer.resume_from_path=$RESUME_FROM_CHECKPOINT")
fi

SAVE_ARGS=("trainer.save_freq=-1")
if [ -n "${SAVE_FREQ:-}" ] && [ "$SAVE_FREQ" -gt 0 ]; then
    SAVE_ARGS=("trainer.save_freq=$SAVE_FREQ")
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  data.max_length=$SFT_MAX_LENGTH \
  data.truncation=$SFT_TRUNCATION \
  ${VAL_MAX_SAMPLES:+data.val_max_samples=$VAL_MAX_SAMPLES} \
  model.partial_pretrain="$MODEL_NAME" \
  model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
  model.fsdp_config.model_dtype=bfloat16 \
  ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
  use_remove_padding=$USE_REMOVE_PADDING \
  trainer.logger=[console,file] \
  trainer.project_name="$EXPERIMENT_NAME" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  +trainer.enable_phase_profiling=True \
  +trainer.phase_profiling_granularity="$GRANULARITY" \
  trainer.test_freq=$VAL_FREQ \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$NNODES \
  "${SAVE_ARGS[@]}" \
  trainer.total_epochs=$TOTAL_EPOCHS \
  ${TOTAL_STEPS:+trainer.total_training_steps=$TOTAL_STEPS} \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUTPUT_DIR" \
  "${RESUME_ARGS[@]}" \
  "${SFT_DATA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed (exit code: $EXIT_CODE)."
fi
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoints: $OUTPUT_DIR"
echo "Logs: $LOG_FILE"
echo "========================================"

exit $EXIT_CODE
