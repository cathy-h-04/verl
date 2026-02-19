#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
cd /home/cathxhou/projects/verl_research

# Activate venv
source /home/cathxhou/projects/verl_research/verl-env/bin/activate

# Common env
EPOCHS=1
POLL_INTERVAL=1
GRANULARITY=operation
DATASET_NAME=gsm8k

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_job() {
  local script_path="$1"
  local use_validation="$2"

  local script_name
  script_name="$(basename "$script_path")"

  if [[ ! -f "$script_path" ]]; then
    log "ERROR: Script not found: $script_path"
    exit 1
  fi

  log "=== START: ${script_name} (USE_VALIDATION=${use_validation}) ==="
  EPOCHS="${EPOCHS}" \
  POLL_INTERVAL="${POLL_INTERVAL}" \
  GRANULARITY="${GRANULARITY}" \
  DATASET_NAME="${DATASET_NAME}" \
  USE_VALIDATION="${use_validation}" \
  bash "$script_path"
  log "=== DONE:  ${script_name} (USE_VALIDATION=${use_validation}) ==="
}

log "Queue starting."

# 1) small validation
run_job "/home/cathxhou/projects/verl_research/profiling_scripts/script_small_validation.sh" 1

# 2) large normal (non-validation)
run_job "/home/cathxhou/projects/verl_research/profiling_scripts/script_large_normal.sh" 0

# 3) large validation
run_job "/home/cathxhou/projects/verl_research/profiling_scripts/script_large_validation.sh" 1

log "Queue finished successfully."
