#!/bin/bash
# ================================================================
# Verl PPO Training Script - GSM8K (Working Version)
# Fast profiling with smaller batches
# ================================================================

set -euo pipefail

# -------------------- Configuration --------------------
PROJECT_DIR=/home/cathxhou/projects/verl_research
EXPERIMENT_NAME="${1:-gsm8k_profile}"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
TOTAL_EPOCHS="${2:-1}"
GPU_ID="${3:-1}"
GRANULARITY="${4:-phase}"  # 'phase' or 'operation'

# Batch size configuration - SMALL for fast profiling
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=8
MICRO_BATCH_SIZE_PER_GPU=2
LOG_PROB_MICRO_BATCH_SIZE=4
GPU_MEMORY_UTIL=0.6

# -------------------- Environment Setup --------------------
cd "$PROJECT_DIR"

if [ ! -d "verl-env" ]; then
    echo "ERROR: verl-env not found. Please create it first."
    exit 1
fi
source verl-env/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}/verl"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
# Structured file logging (JSONL) goes to monitoring/<project>/<experiment>.jsonl
export VERL_FILE_LOGGER_ROOT="${PROJECT_DIR}/monitoring"

# Flash attention enabled (requires compatible flash-attn install)
# export VLLM_DISABLE_FLASHINFER=1  # uncomment if flashinfer causes issues

# -------------------- Directory Setup --------------------
DATA_DIR="${PROJECT_DIR}/data/gsm8k"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}"
LOG_DIR="${PROJECT_DIR}/logs"

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
echo "Python: $(python --version)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "Verl Version: $(python -c 'import verl; print(verl.__version__)' 2>/dev/null || echo 'Not found')"
echo "========================================"
echo ""

# -------------------- Data Preparation --------------------
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ]; then
    echo "Preparing GSM8K dataset..."
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
    echo "Dataset prepared"
else
    echo "Dataset already exists"
fi
echo ""

# -------------------- Training --------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo "========================================"
echo "Starting PPO Training"
echo "========================================"
echo "Logs will be saved to: $LOG_FILE"
echo ""

python3 -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path="$MODEL_NAME" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_model_len=1536 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  critic.model.path="$MODEL_NAME" \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=[console,file] \
  trainer.project_name="$EXPERIMENT_NAME" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  +trainer.enable_phase_profiling=True \
  +trainer.phase_profiling_granularity="$GRANULARITY" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUTPUT_DIR" \
  +critic.model.override_config.attn_implementation=flash_attention_2 \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
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
