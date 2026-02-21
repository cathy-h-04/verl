#!/bin/bash
# ================================================================
# Verl PPO Training Script - GSM8K (Working Version)
# Fast profiling with smaller batches
# ================================================================

set -euo pipefail

# Ensure SCRATCH_DIR is set for the cluster
export SCRATCH_DIR="${SCRATCH_DIR:-/n/netscratch/yu_lab/Lab/chou}"
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"
if [ -n "${RAY_ADDRESS:-}" ]; then
    export RAY_ADDRESS
    echo "Using existing Ray cluster at $RAY_ADDRESS"
fi

# -------------------- Configuration --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
EXPERIMENT_NAME="${1:-gsm8k_profile}"
MODEL_NAME="${5:-Qwen/Qwen2.5-7B-Instruct}"
TOTAL_EPOCHS="${2:-1}"
GPU_ID="${3:-1}"
GRANULARITY="${4:-phase}"  # 'phase' or 'operation'
POLICY="${6:-ppo}"  # ppo | remax
NNODES="${7:-1}"
N_GPUS_PER_NODE="${8:-4}"
DATASET_NAME="${9:-gsm8k}"
export EXPERIMENT_NAME

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
fi

# Rollout tensor parallel size
TENSOR_PARALLEL_SIZE=1
# Batch size configuration - SMALL for fast profiling
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=4
MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE=4
GPU_MEMORY_UTIL=0.50
ROLLOUT_MAX_BATCHED_TOKENS=8192
ROLLOUT_MAX_MODEL_LEN=2048
ROLLOUT_MAX_NUM_SEQS=64
ROLLOUT_N=4
ENABLE_GRAD_CHECKPOINTING=true

# -------------------- Environment Setup --------------------
cd "$PROJECT_DIR"

if [ ! -d "verl-env" ]; then
    echo "ERROR: verl-env not found. Please create it first."
    exit 1
fi
source verl-env/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}/verl"
export PYTHONUNBUFFERED=1
# Structured file logging (JSONL) goes to monitoring/<project>/<experiment>.jsonl
MONITORING_DIR="${MONITORING_DIR:-${SCRATCH_DIR}/logs}"
export VERL_FILE_LOGGER_ROOT="$MONITORING_DIR"

# Flash attention enabled (requires compatible flash-attn install)
# export VLLM_DISABLE_FLASHINFER=1  # uncomment if flashinfer causes issues

# -------------------- Dataset Setup --------------------
case "$DATASET_NAME" in
    gsm8k)
        DATA_DIR="${SCRATCH_DIR}/data/gsm8k"
        TRAIN_FILE="${DATA_DIR}/train.parquet"
        VAL_FILE="${DATA_DIR}/test.parquet"
        PREPROCESS_CMD="python3 examples/data_preprocess/gsm8k.py --local_save_dir \"$DATA_DIR\""
        ;;
    *)
        echo "ERROR: Unsupported dataset '$DATASET_NAME' (only gsm8k supported here)"
        exit 1
        ;;
esac

# -------------------- Directory Setup --------------------
OUTPUT_DIR="${SCRATCH_DIR}/checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="${SCRATCH_DIR}/logs"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# -------------------- System Check --------------------
echo "========================================"
echo "Verl PPO Training - Profiling Mode"
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $MODEL_NAME"
echo "Epochs: $TOTAL_EPOCHS"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "GPU: $GPU_ID"
echo "Profiling Granularity: $GRANULARITY"
echo "Policy: $POLICY"
echo "Nodes: $NNODES (gpus per node: $N_GPUS_PER_NODE)"
echo "Dataset: $DATASET_NAME"
echo "Python: $(python --version)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "Verl Version: $(python -c 'import verl; print(verl.__version__)' 2>/dev/null || echo 'Not found')"
echo "========================================"
echo ""

# -------------------- Data Preparation --------------------
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Preparing dataset ($DATASET_NAME)..."
    eval "$PREPROCESS_CMD"
    echo "Dataset prepared"
else
    echo "Dataset already exists"
fi
echo ""

# -------------------- Training --------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"

echo "========================================"
echo "Starting PPO Training"
echo "========================================"
echo "Logs will be saved to: $LOG_FILE"
echo ""

POLICY_ARGS=()
case "$POLICY" in
    ppo)
        ;;
    remax)
        POLICY_ARGS+=("algorithm.adv_estimator=remax")
        ;;
    grpo)
        POLICY_ARGS+=("algorithm.adv_estimator=grpo")
        POLICY_ARGS+=("algorithm.use_kl_in_reward=False")
        POLICY_ARGS+=("actor_rollout_ref.actor.use_kl_loss=False")
        ;;
    *)
        echo "ERROR: Unsupported policy '$POLICY' (use ppo, remax, or grpo)"
        exit 1
        ;;
esac

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    RESUME_ARGS+=("trainer.resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
fi

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL_NAME" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED_TOKENS \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
  actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  critic.model.path="$MODEL_NAME" \
  critic.optim.lr=1e-5 \
  critic.model.enable_gradient_checkpointing=$ENABLE_GRAD_CHECKPOINTING \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=[console,file] \
  trainer.project_name="$EXPERIMENT_NAME" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  +trainer.enable_phase_profiling=True \
  +trainer.phase_profiling_granularity="$GRANULARITY" \
  trainer.val_before_train=False \
  trainer.test_freq=0 \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$NNODES \
  trainer.save_freq=-1 \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUTPUT_DIR" \
  +critic.model.override_config.attn_implementation=flash_attention_2 \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  "${RESUME_ARGS[@]}" \
  "${POLICY_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

# -------------------- Post-Training --------------------
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "âœ“ Training Completed Successfully!"
else
    echo "âœ— Training Failed (Exit Code: $EXIT_CODE)"
fi
echo "========================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoints: $OUTPUT_DIR"
echo "Logs: $LOG_FILE"
echo "========================================"
echo ""

echo "Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -1
echo ""

exit $EXIT_CODE
