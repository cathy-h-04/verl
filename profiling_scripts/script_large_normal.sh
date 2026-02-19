#!/bin/bash
#
# policy_size_model_nnodes.sh
# Sweep model family, size, policy, and n_gpus_per_node for non-validation runs.
#

source ~/projects/verl_research/verl-env/bin/activate
set -euo pipefail


PROJECT_DIR="/home/cathxhou/projects/verl_research"
PROFILING_DIR="${PROJECT_DIR}/profiling_scripts"

TOKEN_SCRIPT="${PROFILING_DIR}/token.sh"
if [ -f "$TOKEN_SCRIPT" ]; then
    bash "$TOKEN_SCRIPT"
else
    echo "ERROR: token script not found: $TOKEN_SCRIPT"
    exit 1
fi

BASE_EXPERIMENT_PREFIX="${BASE_EXPERIMENT_PREFIX:-sweep}"
EPOCHS="${EPOCHS:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-1}"
GRANULARITY="${GRANULARITY:-phase}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
NNODES="1"
USE_VALIDATION="${USE_VALIDATION:-0}"

# Model IDs (override as needed).
QWEN_SMALL="${QWEN_SMALL:-Qwen/Qwen2.5-1.5B-Instruct}"
QWEN_MEDIUM="${QWEN_MEDIUM:-Qwen/Qwen2.5-1.5B-Instruct}"
QWEN_LARGE="${QWEN_LARGE:-Qwen/Qwen2.5-3B-Instruct}"
LLAMA_SMALL="${LLAMA_SMALL:-meta-llama/Llama-3.1-8B-Instruct}"
LLAMA_LARGE="${LLAMA_LARGE:-meta-llama/Llama-3.1-70B-Instruct}"
MISTRAL_SMALL="${MISTRAL_SMALL:-mistralai/Mistral-7B-Instruct-v0.3}"
MISTRAL_LARGE="${MISTRAL_LARGE:-mistralai/Mistral-7B-Instruct-v0.3}"

# MODELS=("qwen2.5" "llama3.1" "mistral")
# SIZES=("small" "medium" "large")
# N_GPUS_PER_NODE_LIST=("1" "2")
# POLICIES=("ppo" "remax")

MODELS=("qwen2.5")
SIZES=("large")
N_GPUS_PER_NODE_LIST=("1" "2")
echo "using $N_GPUS_PER_NODE_LIST total nodes"
POLICIES=("ppo" "remax")

for model in "${MODELS[@]}"; do
    for size in "${SIZES[@]}"; do
        case "${model}_${size}" in
            qwen2.5_small) MODEL_NAME="$QWEN_SMALL" ;;
            qwen2.5_medium) MODEL_NAME="$QWEN_MEDIUM" ;;
            qwen2.5_large) MODEL_NAME="$QWEN_LARGE" ;;
            llama3.1_small) MODEL_NAME="$LLAMA_SMALL" ;;
            llama3.1_large) MODEL_NAME="$LLAMA_LARGE" ;;
            mistral_small) MODEL_NAME="$MISTRAL_SMALL" ;;
            mistral_large) MODEL_NAME="$MISTRAL_LARGE" ;;
            *)
                echo "ERROR: Unknown model/size combo ${model}_${size}"
                exit 1
                ;;
        esac

        for n_gpus_per_node in "${N_GPUS_PER_NODE_LIST[@]}"; do
            for policy in "${POLICIES[@]}"; do
                if [ "$size" = "medium" ] && [ "$policy" = "ppo" ] && [ "$n_gpus_per_node" = "1" ]; then
                    echo "Skipping ${model}_${size} ${policy} ${n_gpus_per_node}gpn (already collected)."
                    continue
                fi
                TRAIN_SCRIPT="run_verl_train_nonval.sh"
                case "$USE_VALIDATION" in
                    1|true|TRUE|yes|YES)
                        TRAIN_SCRIPT="run_verl_train_val.sh"
                        ;;
                esac
                BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_PREFIX}_${DATASET_NAME}_${policy}_${model}_${size}_${n_gpus_per_node}gpn"
                echo ""
                echo "=== Running: ${BASE_EXPERIMENT_NAME} ==="
                echo "Model: $MODEL_NAME"
                echo "Policy: $policy"
                echo "Nodes: $NNODES (gpus per node: $n_gpus_per_node)"
                echo "Validation: $USE_VALIDATION"
                echo "Dataset: $DATASET_NAME"

                if ! TRAIN_SCRIPT="$TRAIN_SCRIPT" bash "${PROFILING_DIR}/run_with_phase_monitoring.sh" \
                    "$BASE_EXPERIMENT_NAME" \
                    "$EPOCHS" \
                    "$POLL_INTERVAL" \
                    "$GRANULARITY" \
                    "$MODEL_NAME" \
                    "$policy" \
                    "$NNODES" \
                    "$n_gpus_per_node" \
                    "$DATASET_NAME"; then
                    EXIT_CODE=$?
                    echo "WARNING: Run failed (exit code: $EXIT_CODE). Continuing sweep..."
                fi
            done
        done
    done
done
