#!/bin/bash
#SBATCH --job-name=verl-run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH -p gpu_h200
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# NOTE: Submit with submit_sweep.sh to override resources at runtime.
#SBATCH --output=/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.out
#SBATCH --error=/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.err

# 1. Environment Setup (Native)
source /n/home08/chou/verl_research/verl-env/bin/activate
export PYTHONPATH=$PYTHONPATH:/n/home08/chou/verl_research
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
export RAY_MANAGED_BY_SLURM=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer the submit-time script dir to avoid Slurm spool paths.
if [ -n "${SUBMIT_SCRIPT_DIR:-}" ] && [ -f "${SUBMIT_SCRIPT_DIR}/config_utils.py" ]; then
    SCRIPT_DIR="${SUBMIT_SCRIPT_DIR}"
fi
# When running under Slurm, the script is copied to a spool dir.
# Fall back to the submit directory if needed.
if [ ! -f "${SCRIPT_DIR}/config_utils.py" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/profiling_scripts/config_utils.py" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/profiling_scripts"
    fi
fi
# Temporary fallback while transitions settle.
if [ ! -f "${SCRIPT_DIR}/config_utils.py" ] && [ -f "/n/home08/chou/verl_research/profiling_scripts/config_utils.py" ]; then
    SCRIPT_DIR="/n/home08/chou/verl_research/profiling_scripts"
fi
if [ ! -f "${SCRIPT_DIR}/config_utils.py" ]; then
    echo "ERROR: config_utils.py not found. Set SUBMIT_SCRIPT_DIR or SLURM_SUBMIT_DIR."
    exit 1
fi
export HOME_STORAGE="/n/home08/chou/verl_research/results"
mkdir -p "$HOME_STORAGE"
MOVE_CHECKPOINTS="${MOVE_CHECKPOINTS:-false}"
RUNS_FILE="${RUNS_FILE:-}"
if [ -z "${RUNS_FILE:-}" ]; then
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
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
export HOME_LOGS="/n/home08/chou/verl_research/logs"
mkdir -p "$HOME_LOGS"
REQUESTED_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [ -z "$REQUESTED_GPUS_PER_NODE" ]; then
    echo "WARNING: SLURM_GPUS_PER_NODE not set; defaulting to 1 for Ray startup."
    REQUESTED_GPUS_PER_NODE=1
fi


# Extract the run corresponding to this specific array task
CONFIG_EXPORTS=$(python3 "${SCRIPT_DIR}/config_utils.py" export --config "$RUNS_FILE" --index "$TASK_ID")
eval "$CONFIG_EXPORTS"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-${RUN_CONFIG_NAME:-}}"
if [ -z "$BASE_EXPERIMENT_NAME" ]; then
    echo "ERROR: Missing base experiment name in config."
    exit 1
fi

if [ "${SAVE_FREQ:-}" = "-1" ]; then
    MOVE_CHECKPOINTS="false"
else
    MOVE_CHECKPOINTS="true"
fi

MAX_PARAM_GPUS=$(python3 "${SCRIPT_DIR}/config_utils.py" max-gpus --config "$RUNS_FILE")

if [ "$REQUESTED_GPUS_PER_NODE" -lt "$MAX_PARAM_GPUS" ]; then
    echo "ERROR: Requested GPUs per node ($REQUESTED_GPUS_PER_NODE) is less than max GPUS in RUNS_FILE ($MAX_PARAM_GPUS)."
    echo "Fix: increase Slurm GPU request or lower GPUS in $RUNS_FILE."
    exit 1
fi
if [ -n "${N_GPUS_PER_NODE:-}" ] && [ "$REQUESTED_GPUS_PER_NODE" -gt "$N_GPUS_PER_NODE" ]; then
    echo "WARNING: Slurm requested $REQUESTED_GPUS_PER_NODE GPU(s) but run requests only $N_GPUS_PER_NODE GPU(s)."
    echo "This will leave GPUs idle. Set gpus_per_node to match or lower --gres."
    if [ "${STRICT_GPU_MATCH:-0}" = "1" ]; then
        echo "ERROR: STRICT_GPU_MATCH=1 and GPU mismatch detected."
        exit 1
    fi
fi

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "PREFLIGHT_ONLY=1: config checks passed; exiting before runtime."
    exit 0
fi

# 2. Get the list of nodes assigned by Slurm
if [ "${DRY_RUN:-0}" = "1" ]; then
    if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
        SLURM_JOB_NODELIST="$(hostname)"
    fi
    head_node="$(hostname)"
    head_node_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "$head_node_ip" ]; then
        head_node_ip="$(hostname -i 2>/dev/null | awk '{print $1}')"
    fi
else
    # Using readarray to safely handle the newline-separated list from scontrol
    readarray -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | awk '{print $1}')
fi

export RAY_PORT=$((6379 + TASK_ID))
port=$RAY_PORT
export RAY_ADDRESS="$head_node_ip:$port"
echo "Starting Ray Head Node at: $head_node ($head_node_ip)"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: exiting after Slurm-path checks."
    echo "SCRIPT_DIR=${SCRIPT_DIR}"
    echo "RUNS_FILE=${RUNS_FILE}"
    echo "RUN_CONFIG_NAME=${RUN_CONFIG_NAME:-}"
    echo "BASE_EXPERIMENT_NAME=${BASE_EXPERIMENT_NAME:-}"
    echo "REQUESTED_GPUS_PER_NODE=${REQUESTED_GPUS_PER_NODE}"
    echo "MAX_PARAM_GPUS=${MAX_PARAM_GPUS}"
    echo "HEAD_NODE=${head_node}"
    echo "HEAD_NODE_IP=${head_node_ip}"
    echo "RAY_PORT=${RAY_PORT}"
    echo "RAY_ADDRESS=${RAY_ADDRESS}"
    exit 0
fi

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
    export MONITOR_DIR
    echo "$EXPERIMENT_NAME" > "${MONITOR_DIR}/experiment_name.txt"
    {
        echo "timestamp: $(date +%Y%m%d_%H%M%S)"
        echo "slurm_job_id: ${SLURM_JOB_ID}"
        echo "slurm_array_task_id: ${TASK_ID}"
        echo "slurm_job_name: ${SLURM_JOB_NAME}"
    } >> "${MONITOR_DIR}/slurm_job_ids.txt"
    if [ -n "${RUN_CONFIG_JSON:-}" ]; then
        python3 - <<'PY'
import json
import os

raw = os.environ.get("RUN_CONFIG_JSON", "")
out_path = os.environ.get("MONITOR_DIR", "") + "/run_config.json"
if raw and out_path:
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {"raw": raw}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
PY
    elif [ -n "${RUN_CONFIG_PATH:-}" ] && [ -f "${RUN_CONFIG_PATH:-}" ]; then
        cp "${RUN_CONFIG_PATH}" "${MONITOR_DIR}/run_config.json"
    fi

    if [ -n "${SLURM_CONFIG_PATH:-}" ] && [ -f "${SLURM_CONFIG_PATH:-}" ]; then
        cp "${SLURM_CONFIG_PATH}" "${MONITOR_DIR}/slurm_config.json"
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
        rm -rf "$RAY_TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    else
        echo "WARNING: Migration failed. Data remains in scratch for manual recovery."
    fi

    LOG_COPY_OK=0
    if [ -f "$SCRATCH_DIR/logs/${EXPERIMENT_NAME}.log" ]; then
        if rsync -avz "$SCRATCH_DIR/logs/${EXPERIMENT_NAME}.log" "$HOME_LOGS/"; then
            rm -f "$SCRATCH_DIR/logs/${EXPERIMENT_NAME}.log"
            LOG_COPY_OK=1
        fi
    fi
    if [ -f "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.out" ]; then
        if rsync -avz "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.out" "$HOME_LOGS/"; then
            rm -f "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.out"
            LOG_COPY_OK=1
        fi
    fi
    if [ -f "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.err" ]; then
        if rsync -avz "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.err" "$HOME_LOGS/"; then
            rm -f "$SCRATCH_DIR/logs/${SLURM_JOB_ID}_${TASK_ID}.err"
            LOG_COPY_OK=1
        fi
    fi

    # Copy Slurm stdout/stderr (from --output/--error) into HOME_LOGS if configured.
    SLURM_OUT_PATH=""
    SLURM_ERR_PATH=""
    if [ -n "${SLURM_CONFIG_PATH:-}" ] && [ -f "${SLURM_CONFIG_PATH:-}" ]; then
        SLURM_OUT_PATH=$(python3 - <<'PY'
import json
import os

cfg_path = os.environ.get("SLURM_CONFIG_PATH", "")
job_id = os.environ.get("SLURM_JOB_ID", "")
task_id = os.environ.get("TASK_ID", "")
job_name = os.environ.get("SLURM_JOB_NAME", "")
submit_dir = os.environ.get("SLURM_SUBMIT_DIR", "")

def expand(p):
    if not p:
        return ""
    out = p
    out = out.replace("%A", job_id)
    out = out.replace("%a", task_id)
    out = out.replace("%j", job_id)
    out = out.replace("%x", job_name)
    if not out.startswith("/"):
        out = os.path.join(submit_dir or os.getcwd(), out)
    return out

try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(expand(cfg.get("output", "")))
except Exception:
    print("")
PY
        )
        SLURM_ERR_PATH=$(python3 - <<'PY'
import json
import os

cfg_path = os.environ.get("SLURM_CONFIG_PATH", "")
job_id = os.environ.get("SLURM_JOB_ID", "")
task_id = os.environ.get("TASK_ID", "")
job_name = os.environ.get("SLURM_JOB_NAME", "")
submit_dir = os.environ.get("SLURM_SUBMIT_DIR", "")

def expand(p):
    if not p:
        return ""
    out = p
    out = out.replace("%A", job_id)
    out = out.replace("%a", task_id)
    out = out.replace("%j", job_id)
    out = out.replace("%x", job_name)
    if not out.startswith("/"):
        out = os.path.join(submit_dir or os.getcwd(), out)
    return out

try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(expand(cfg.get("error", "")))
except Exception:
    print("")
PY
        )
    fi

    if [ -z "$SLURM_OUT_PATH" ]; then
        SLURM_OUT_PATH="/n/netscratch/yu_lab/Lab/chou/logs/${SLURM_JOB_ID}_${TASK_ID}.out"
    fi
    if [ -z "$SLURM_ERR_PATH" ]; then
        SLURM_ERR_PATH="/n/netscratch/yu_lab/Lab/chou/logs/${SLURM_JOB_ID}_${TASK_ID}.err"
    fi

    if [ -f "$SLURM_OUT_PATH" ]; then
        rsync -avz "$SLURM_OUT_PATH" "$HOME_LOGS/" || true
    fi
    if [ -f "$SLURM_ERR_PATH" ]; then
        rsync -avz "$SLURM_ERR_PATH" "$HOME_LOGS/" || true
    fi

    RAY_LOG_COPY_OK=0
    if [ -n "${RAY_TMPDIR:-}" ] && [ -d "$RAY_TMPDIR" ]; then
        RAY_LOG_DEST="${HOME_LOGS}/ray_${SLURM_JOB_ID}_${TASK_ID}"
        mkdir -p "$RAY_LOG_DEST"
        shopt -s nullglob
        for SESSION_DIR in "$RAY_TMPDIR"/session_*; do
            if [ -d "$SESSION_DIR/logs" ]; then
                if rsync -avz "$SESSION_DIR/logs/" "$RAY_LOG_DEST/"; then
                    RAY_LOG_COPY_OK=1
                fi
            fi
        done
        shopt -u nullglob
        if [ "$RAY_LOG_COPY_OK" -eq 1 ]; then
            rm -rf "$RAY_TMPDIR"
        fi
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

# 6. Launch the profiling run
TRAIN_SCRIPT="run_verl_train_nonval.sh"
case "${USE_VALIDATION:-}" in
    1|true|TRUE|yes|YES)
        TRAIN_SCRIPT="run_verl_train_val.sh"
        ;;
esac
export TRAIN_SCRIPT

echo "Launching Research Suite..."
bash "${SCRIPT_DIR}/run_with_phase_monitoring.sh"
