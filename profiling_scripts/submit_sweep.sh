#!/usr/bin/env bash
set -euo pipefail

PARAM_FILE="${PARAM_FILE:-/n/home08/chou/verl_research/profiling_scripts/sweep_params.txt}"
if [ ! -f "$PARAM_FILE" ]; then
  echo "ERROR: PARAM_FILE not found: $PARAM_FILE"
  exit 1
fi

NUM_LINES=$(tail -n +2 "$PARAM_FILE" | awk 'NF{c++} END{print c+0}')
if [ "$NUM_LINES" -lt 1 ]; then
  echo "ERROR: No data rows found in $PARAM_FILE"
  exit 1
fi

MAIL_ARGS=(--mail-user="cathyhou@college.harvard.edu" --mail-type=FAIL,TIME_LIMIT,END)

sbatch "${MAIL_ARGS[@]}" \
  --array=1-"$NUM_LINES" \
  --export=ALL,PARAM_FILE="$PARAM_FILE" \
  /n/home08/chou/verl_research/profiling_scripts/ray_on_slurm.slurm
