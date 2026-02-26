# Profiling Suite (Slurm + Ray)

## Where Slurm logs go
Slurm stdout/stderr for array jobs are written to:
`/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.out`
`/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.err`

Where:
- `%A` = Slurm job ID
- `%a` = array task ID

## Primary entry points
- `submit_sweep.sh`
  - Submits a Slurm array from a JSON runs file.
  - Reads Slurm resources from a separate `slurm.json`.
  - Computes `--array` from run count.
- `run_interactive_from_params.sh`
  - Runs a single JSON run inside an interactive allocation (no sbatch).

## JSON config layout
Place experiment configs in a subfolder, e.g.:
- `experiments/stage1_llama8b/runs.json`

### `runs.json` format (high-level)
- `runs`: list of per-run overrides (put frequently changed fields here).
- `defaults.run`: shared run-level defaults.
- `defaults.train`: shared training defaults (rarely changed).

Example:
```json
{
  "runs": [
    {
      "name": "gsm8k_val_profile",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "total_steps": null,
      "save_freq": null
    }
  ],
  "defaults": {
    "run": {
      "total_epochs": 1,
      "poll_interval": 1,
      "granularity": "phase",
      "policy": "ppo",
      "nnodes": 1,
      "gpus_per_node": 4,
      "dataset": "gsm8k",
      "use_validation": true,
      "val_freq": 20,
      "rollout_n": 4
    },
    "train": {
      "train_batch_size": 128,
      "ppo_mini_batch_size": 32,
      "micro_batch_size_per_gpu": 4,
      "gpu_memory_util": 0.5
    }
  }
}
```

### `slurm.json` format (resources)
Key fields:
- `job_name`, `partition`, `time`, `nodes`, `ntasks_per_node`
- `gpus_per_node`, `cpus_per_task`, `mem`
- `output`, `error`, `mail_user`, `mail_type`

## Typical commands
Batch sweep:
```bash
./submit_sweep.sh --runs experiments/stage1_llama8b/runs.json
```

Interactive (single run):
```bash
./run_interactive_from_params.sh --config experiments/stage1_llama8b/runs.json --line 1
```

## Config persistence (reproducibility)
Resolved configs are written alongside monitoring outputs:
- `run_config.json` (resolved run + defaults)
- `slurm_config.json` (if provided)

## Internal scripts (called automatically)
- `run_with_phase_monitoring.sh`
  - Orchestrator: starts GPU monitor, runs training, handles cleanup.
- `run_verl_train_nonval.sh`
  - Training without validation (fast profiling).
- `run_verl_train_val.sh`
  - Training with validation.
- `monitor_nvidia_smi_phased.sh`
  - GPU monitoring + phase annotations.
- `verl_subphase_profiler.py`
  - Phase IPC + optional sub-phase timing logs.
