#!/bin/bash
#SBATCH --job-name=verl-ray-integrity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -p gpu
#SBATCH --time=04:00:00        # Increased to 4h for a research-grade run
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --array=1-6
# NOTE: Keep --array in sync with the number of data rows in sweep_params.csv.
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

# Extract the line corresponding to this specific array task
PARAMS=$(awk -F, -v line="$SLURM_ARRAY_TASK_ID" 'NR==line+1 {print $0}' /n/home08/chou/verl_research/profiling_scripts/sweep_params.csv)
if [ -z "$PARAMS" ]; then
    echo "ERROR: No params found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    echo "Check /n/home08/chou/verl_research/profiling_scripts/sweep_params.csv and --array range."
    exit 1
fi
PARAMS="${PARAMS//,/ }"
BASE_EXPERIMENT_NAME=$(echo "$PARAMS" | awk '{print $1}')

# 2. Get the list of nodes assigned by Slurm
# Using readarray to safely handle the newline-separated list from scontrol
readarray -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | awk '{print $1}')

export RAY_PORT=$((6379 + SLURM_ARRAY_TASK_ID))
port=$RAY_PORT
export RAY_ADDRESS="$head_node_ip:$port"
echo "Starting Ray Head Node at: $head_node ($head_node_ip)"

# 3. Cleanup Logic (Ensures no zombie processes on exit)
cleanup() {
    echo "Starting post-run data migration..."
    DEST_DIR="${HOME_STORAGE}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$DEST_DIR"

    EXPERIMENT_NAME_FILE="${SCRATCH_DIR}/logs/.experiment_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
    if [ -f "$EXPERIMENT_NAME_FILE" ]; then
        EXPERIMENT_NAME=$(cat "$EXPERIMENT_NAME_FILE")
    else
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}"
    fi

    # Make experiment name visible in results
    DEST_DIR="${HOME_STORAGE}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${EXPERIMENT_NAME}"
    mkdir -p "$DEST_DIR"
    echo "$EXPERIMENT_NAME" > "${DEST_DIR}/experiment_name.txt"
    echo "$PARAMS" > "${DEST_DIR}/params.txt"

    LOGS_OK=0
    CKPT_OK=0

    if [ -d "$SCRATCH_DIR/logs/$EXPERIMENT_NAME" ]; then
        rsync -avz "$SCRATCH_DIR/logs/$EXPERIMENT_NAME/" "$DEST_DIR/logs/"
        LOGS_OK=$?
    fi

    if [ "${MOVE_CHECKPOINTS}" = "true" ] && [ -d "$SCRATCH_DIR/checkpoints/$EXPERIMENT_NAME" ]; then
        rsync -avz "$SCRATCH_DIR/checkpoints/$EXPERIMENT_NAME/" "$DEST_DIR/checkpoints/"
        CKPT_OK=$?
    fi

    if [ "${LOGS_OK}" -eq 0 ] && { [ "${MOVE_CHECKPOINTS}" != "true" ] || [ "${CKPT_OK}" -eq 0 ]; }; then
        echo "Migration successful. Purging scratch directory..."
        rm -rf "$SCRATCH_DIR/logs/$EXPERIMENT_NAME"
        if [ "${MOVE_CHECKPOINTS}" = "true" ]; then
            rm -rf "$SCRATCH_DIR/checkpoints/$EXPERIMENT_NAME"
        fi
        rm -rf "$RAY_TMPDIR"
    else
        echo "WARNING: Migration failed. Data remains in scratch for manual recovery."
    fi

    echo "Stopping Ray on all nodes..."
    srun ray stop || true
}
trap cleanup EXIT INT TERM

# 4. Start Ray Head Node
# --block keeps the process alive; & puts it in background so we can continue
ray start --head --node-ip-address="$head_node_ip" --port=$RAY_PORT \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --block &
sleep 15

# 5. Start Ray Worker Nodes (Only if nodes > 1)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [[ "$SLURM_JOB_NUM_NODES" -gt 1 ]]; then
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Joining Worker Node: $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$RAY_ADDRESS" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --block &
        sleep 5
    done
fi

# 6. Call your existing Profiling Entry Point
# "$@" forwards all command line arguments from sbatch to your script
echo "Launching Research Suite..."
bash /n/home08/chou/verl_research/profiling_scripts/policy_size_model_nnodes.sh $PARAMS
