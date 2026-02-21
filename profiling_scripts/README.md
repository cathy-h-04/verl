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
  - Submit via `sbatch profiling_scripts/ray_on_slurm.slurm`
  - Starts Ray (head + workers if nnodes>1)
  - Reads one line from `sweep_params.txt` based on `SLURM_ARRAY_TASK_ID`
  - Dispatches a single experiment per array task

- `sweep_params.csv`
  - One experiment per line (CSV with header)
  - Format:
    `BASE_EXPERIMENT_NAME,MODEL_NAME,EPOCHS,POLL_INTERVAL,GRANULARITY,POLICY,NNODES,N_GPUS_PER_NODE,DATASET_NAME,USE_VALIDATION`

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

- `verl_phase_profiler.py`, `verl_subphase_profiler.py`
  - Phase IPC + optional sub-phase timing logs.

## Data/outputs (cluster scratch)
All artifacts are under:
`/n/netscratch/yu_lab/Lab/chou`

- Logs: `logs/`
  - Training log: `logs/${EXPERIMENT_NAME}.log`
  - Phase CSV: `logs/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}_phased_<timestamp>.csv`
  - Validation CSV: `logs/${EXPERIMENT_NAME}/val_accuracy.csv` (validation runs)
- Checkpoints: `checkpoints/${EXPERIMENT_NAME}`
- Data: `data/gsm8k/*`
