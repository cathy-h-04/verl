# Profiling Suite (Slurm + Ray)

## Where Slurm logs go
Slurm stdout/stderr for array jobs are written to:
`/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.out`
`/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.err`

Where:
- `%A` = Slurm job ID
- `%a` = array task ID

## Primary entry points
- `ray_on_slurm.slurm`
  - Submit via `profiling_scripts/submit_sweep.sh` (auto-sets array size)
  - You can override `PARAM_FILE` and `MAIL_USER` in the environment.
  - Starts Ray (head + workers if nnodes>1)
  - Reads one line from `sweep_params.txt` based on `SLURM_ARRAY_TASK_ID`
  - Dispatches a single experiment per array task

- `sweep_params.txt`
  - One experiment per line (CSV with header)
  - Format:
    `NAME,MODEL,EPOCHS,POLL,GRANULARITY,POLICY,NODES,GPUS,DATASET,VAL,TOTAL_STEPS,SAVE_FREQ,RESUME_PATH`
  - You can override the file with `PARAM_FILE=...` when submitting.

- `submit_sweep.sh`
  - Computes `--array` from the number of data rows in `PARAM_FILE`.
  - Optional email alerts via `MAIL_USER=you@harvard.edu`.

- `sweep_params_stage1.txt`
  - Stage 1 runs (e.g., stop at step 200) using `TOTAL_STEPS=200` and `SAVE_FREQ=200`.

- `sweep_params_stage2.txt`
  - Stage 2 resumption runs. `SAVE_FREQ=-1` disables checkpointing and `RESUME_PATH` should point to stage 1 checkpoints.

## Internal scripts (called automatically)
- `policy_size_model_nnodes.sh`
  - Single-run entry point, invoked by `ray_on_slurm.slurm`.

- `run_with_phase_monitoring.sh`
  - Orchestrator: starts GPU monitor, runs training, handles cleanup.

- `run_verl_train_nonval.sh`
  - Training without validation (fast profiling).

- `run_verl_train_val.sh`
  - Training with validation + accuracy summary CSV.

- `monitor_nvidia_smi_phased.sh`
  - GPU monitoring + phase annotations.

- `verl_subphase_profiler.py`
  - Phase IPC + optional sub-phase timing logs (phases: idle, rollout, rl_policy, training, validation, other).

## Data/outputs (cluster scratch)
All artifacts are under:
`/n/netscratch/yu_lab/Lab/chou`

- Logs: `logs/`
  - Training log: `logs/${EXPERIMENT_NAME}.log` (non-critical)
- Monitoring outputs (primary artifacts):
  - Non-val: `monitoring/${EXPERIMENT_NAME}/...`
  - Val: `monitoring_val/${EXPERIMENT_NAME}/...`
- Checkpoints: `checkpoints/${EXPERIMENT_NAME}`
- Data: `data/gsm8k/*`

## Results (persistent home)
Monitoring outputs are synced to:
`/n/home08/chou/verl_research/results/monitoring/${EXPERIMENT_NAME}/`
`/n/home08/chou/verl_research/results/monitoring_val/${EXPERIMENT_NAME}/`
