#!/bin/bash
#SBATCH --job-name=verl-ray-integrity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH -p gpu_requeue
#SBATCH --time=00:25:00        # 20m smoke-test window to reach training
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
# NOTE: Use profiling_scripts/submit_sweep.sh to set --array based on PARAM_FILE.
#SBATCH --output=/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.out
#SBATCH --error=/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.err

# 1. Environment Setup (Native)
source /n/home08/chou/verl_research/verl-env/bin/activate
export PYTHONPATH=$PYTHONPATH:/n/home08/chou/verl_research
export SCRATCH_DIR=/n/netscratch/yu_lab/Lab/chou
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/checkpoints" "$SCRATCH_DIR/data"
export RAY_TMPDIR="${SCRATCH_DIR}/ray_tmp"
mkdir -p "$RAY_TMPDIR"
export RAY_MANAGED_BY_SLURM=1
export HOME_STORAGE="/n/home08/chou/verl_research/results"
mkdir -p "$HOME_STORAGE"
MOVE_CHECKPOINTS="${MOVE_CHECKPOINTS:-false}"
PARAM_FILE="${PARAM_FILE:-/n/home08/chou/verl_research/profiling_scripts/sweep_params.txt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
export HOME_LOGS="/n/home08/chou/verl_research/logs"
mkdir -p "$HOME_LOGS"
REQUESTED_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [ -z "$REQUESTED_GPUS_PER_NODE" ]; then
    echo "WARNING: SLURM_GPUS_PER_NODE not set; defaulting to 1 for Ray startup."
    REQUESTED_GPUS_PER_NODE=1
fi

# Extract the line corresponding to this specific array task
PARAM_HEADER=$(awk 'NR==1 {print $0}' "$PARAM_FILE")
PARAM_LINE=$(awk -F, -v line="$TASK_ID" 'NR==line+1 {print $0}' "$PARAM_FILE")
if [ -z "$PARAM_LINE" ]; then
    echo "ERROR: No params found for SLURM_ARRAY_TASK_ID=${TASK_ID}"
    echo "Check ${PARAM_FILE} and --array range."
    exit 1
fi
export PARAM_HEADER
export PARAM_LINE

IFS=',' read -r NAME MODEL EPOCHS POLL GRANULARITY POLICY NODES GPUS DATASET VAL VAL_FREQ VAL_SAMPLES TOTAL_STEPS SAVE_FREQ RESUME_PATH <<< "$PARAM_LINE"
BASE_EXPERIMENT_NAME="$NAME"
USE_VALIDATION="$VAL"
if [ "${SAVE_FREQ:-}" = "-1" ]; then
    MOVE_CHECKPOINTS="false"
else
    MOVE_CHECKPOINTS="true"
fi

MAX_PARAM_GPUS=$(awk -F, 'NR>1 && $8 ~ /^[0-9]+$/ {if ($8>m) m=$8} END{print m+0}' "$PARAM_FILE")
if [ "$REQUESTED_GPUS_PER_NODE" -lt "$MAX_PARAM_GPUS" ]; then
    echo "ERROR: Requested GPUs per node ($REQUESTED_GPUS_PER_NODE) is less than max GPUS in PARAM_FILE ($MAX_PARAM_GPUS)."
    echo "Fix: increase Slurm GPU request or lower GPUS column in $PARAM_FILE."
    exit 1
fi
if [ -n "${GPUS:-}" ] && [ "$REQUESTED_GPUS_PER_NODE" -gt "$GPUS" ]; then
    echo "WARNING: Slurm requested $REQUESTED_GPUS_PER_NODE GPU(s) but params request only $GPUS GPU(s)."
    echo "This will leave GPUs idle. Set GPUS in $PARAM_FILE to match or lower --gres."
    if [ "${STRICT_GPU_MATCH:-0}" = "1" ]; then
        echo "ERROR: STRICT_GPU_MATCH=1 and GPU mismatch detected."
        exit 1
    fi
fi

# 2. Get the list of nodes assigned by Slurm
# Using readarray to safely handle the newline-separated list from scontrol
readarray -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | awk '{print $1}')

export RAY_PORT=$((6379 + TASK_ID))
port=$RAY_PORT
export RAY_ADDRESS="$head_node_ip:$port"
echo "Starting Ray Head Node at: $head_node ($head_node_ip)"

# 3. Cleanup Logic (Ensures no zombie processes on exit)
cleanup() {
    echo "Starting post-run data migration..."
    DEST_DIR="${HOME_STORAGE}"
    mkdir -p "$DEST_DIR"

    EXPERIMENT_NAME_FILE="${SCRATCH_DIR}/logs/.experiment_${SLURM_JOB_ID}_${TASK_ID}.txt"
    if [ -f "$EXPERIMENT_NAME_FILE" ]; then
        EXPERIMENT_NAME=$(cat "$EXPERIMENT_NAME_FILE")
    else
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${SLURM_JOB_ID}_${TASK_ID}"
    fi

    case "${USE_VALIDATION}" in
        1|true|TRUE|yes|YES)
            MONITOR_ROOT="${SCRATCH_DIR}/monitoring_val"
            DEST_ROOT="${HOME_STORAGE}/monitoring_val"
            ;;
        *)
            MONITOR_ROOT="${SCRATCH_DIR}/monitoring"
            DEST_ROOT="${HOME_STORAGE}/monitoring"
            ;;
    esac
    MONITOR_DIR="${MONITOR_ROOT}/${EXPERIMENT_NAME}"
    mkdir -p "$MONITOR_DIR"
    echo "$EXPERIMENT_NAME" > "${MONITOR_DIR}/experiment_name.txt"
    if [ -n "${PARAM_HEADER:-}" ] && [ -n "${PARAM_LINE:-}" ]; then
        {
            echo "$PARAM_HEADER"
            echo "$PARAM_LINE"
        } > "${MONITOR_DIR}/params.txt"
    fi

    mkdir -p "$DEST_ROOT"
    DEST_DIR="${DEST_ROOT}/${EXPERIMENT_NAME}"
    mkdir -p "$DEST_DIR"

    LOGS_OK=0
    CKPT_OK=0

    if rsync -avz "$MONITOR_DIR/" "$DEST_DIR/"; then
        LOGS_OK=0
    else
        LOGS_OK=$?
    fi

    # Flatten nested phase_timings directory if present.
    NESTED_DIR="${DEST_DIR}/${EXPERIMENT_NAME}"
    NESTED_TIMINGS="${NESTED_DIR}/phase_timings_${EXPERIMENT_NAME}.jsonl"
    DEST_TIMINGS="${DEST_DIR}/phase_timings_${EXPERIMENT_NAME}.jsonl"
    if [ -f "$NESTED_TIMINGS" ]; then
        if [ ! -f "$DEST_TIMINGS" ]; then
            mv "$NESTED_TIMINGS" "$DEST_TIMINGS"
        else
            rm -f "$NESTED_TIMINGS"
        fi
        rmdir "$NESTED_DIR" 2>/dev/null || true
    fi

    if [ "${MOVE_CHECKPOINTS}" = "true" ] && [ -d "$SCRATCH_DIR/checkpoints/$EXPERIMENT_NAME" ]; then
        if rsync -avz "$SCRATCH_DIR/checkpoints/$EXPERIMENT_NAME/" "$DEST_DIR/checkpoints/"; then
            CKPT_OK=0
        else
            CKPT_OK=$?
        fi
    fi

    if [ "${LOGS_OK}" -eq 0 ] && { [ "${MOVE_CHECKPOINTS}" != "true" ] || [ "${CKPT_OK}" -eq 0 ]; }; then
        echo "Migration successful. Purging scratch directory..."
        rm -rf "$MONITOR_ROOT/$EXPERIMENT_NAME"
        if [ "${MOVE_CHECKPOINTS}" = "true" ]; then
            rm -rf "$SCRATCH_DIR/checkpoints/$EXPERIMENT_NAME"
        fi
        rm -rf "$RAY_TMPDIR"
    else
        echo "WARNING: Migration failed. Data remains in scratch for manual recovery."
    fi

    if [ -f "$SCRATCH_DIR/logs/${EXPERIMENT_NAME}.log" ]; then
        rsync -avz "$SCRATCH_DIR/logs/${EXPERIMENT_NAME}.log" "$HOME_LOGS/" || true
    fi
    if [ -f "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.out" ]; then
        rsync -avz "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.out" "$HOME_LOGS/" || true
    fi
    if [ -f "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.err" ]; then
        rsync -avz "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.err" "$HOME_LOGS/" || true
    fi

    echo "Stopping Ray on all nodes..."
    srun ray stop || true
}
trap cleanup EXIT INT TERM

# 4. Start Ray Head Node
# --block keeps the process alive; & puts it in background so we can continue
ray start --head --node-ip-address="$head_node_ip" --port=$RAY_PORT \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$REQUESTED_GPUS_PER_NODE" --block &
sleep 15

# 5. Start Ray Worker Nodes (Only if nodes > 1)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [[ "$SLURM_JOB_NUM_NODES" -gt 1 ]]; then
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Joining Worker Node: $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$RAY_ADDRESS" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$REQUESTED_GPUS_PER_NODE" --block &
        sleep 5
    done
fi

# 6. Call your existing Profiling Entry Point
# "$@" forwards all command line arguments from sbatch to your script
echo "Launching Research Suite..."
bash /n/home08/chou/verl_research/profiling_scripts/policy_size_model_nnodes.sh \
    "$NAME" \
    "$MODEL" \
    "$EPOCHS" \
    "$POLL" \
    "$GRANULARITY" \
    "$POLICY" \
    "$NODES" \
    "$GPUS" \
    "$DATASET" \
    "$VAL" \
    "$VAL_FREQ" \
    "$VAL_SAMPLES" \
    "$TOTAL_STEPS" \
    "$SAVE_FREQ" \
    "$RESUME_PATH"
