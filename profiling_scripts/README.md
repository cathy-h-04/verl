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
- Optional run-level `power_cap_w`:
  - `null` / omitted / `"default"` / `0`: no cap (use default GPU power limit)
  - positive integer (for example `560`): applies `nvidia-smi -pl <watts>` at run start
    and restores original limits during cleanup.
- Optional train-level `rollout_quantization`:
  - omitted / `null` / `"default"`: no rollout quantization override (backward compatible)
  - `"fp8"`: sets `actor_rollout_ref.rollout.quantization=fp8` for vLLM rollout.
- Optional train-level reward-model overrides (backward compatible):
  - `reward_model_enable`: `true`/`false` to force-enable or disable model RM.
  - `reward_model_name`: HF model id/path used when model RM is enabled.
  - `reward_model_micro_batch_size_per_gpu`: RM scoring micro-batch size per GPU.

Example:
```json
{
  "runs": [
    {
      "name": "gsm8k_val_profile",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "total_steps": null,
      "save_freq": null,
      "power_cap_w": null,
      "rollout_quantization": null
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
  - Orchestrator: runs training + phase telemetry, handles cleanup.
- `run_verl_train_nonval.sh`
  - Training without validation (fast profiling).
- `run_verl_train_val.sh`
  - Training with validation.
- `run_verl_train_sft_nonval.sh`
  - SFT training without validation.
- `run_verl_train_sft_val.sh`
  - SFT training with validation.
- `monitor_nvidia_smi_phased.sh`
  - Legacy GPU monitoring + phase annotations (not used by default launcher path).
- `verl_subphase_profiler.py`
  - Phase IPC + optional sub-phase timing logs.
  - Also emits NVML/RAPL boundary + periodic JSONL telemetry in the monitoring directory:
    - `nvml_boundary.jsonl`, `nvml_periodic.jsonl`
    - `rapl_boundary.jsonl`, `rapl_periodic.jsonl`
    - `tokens_and_steps.jsonl`
  - Sampling cadence knobs:
    - `VERL_TELEMETRY_SAMPLE_INTERVAL_S` (seconds)
    - `VERL_TELEMETRY_SAMPLE_HZ` (frequency; overrides interval if set)
  - In `granularity=operation`, `phase_timings_<experiment>.jsonl` emits one record per subphase metric with explicit tags:
    - `phase_name`, `subphase_name`, `value`, `metric_unit`
  - Wide trainer JSONL (`<experiment>.jsonl`) deduplicates timing breakdown keys in `operation` mode:
    - drops `generation_timing/*`, `timing_dist_s/*`, and most `timing_s/*`
    - keeps `timing_s/step` and `timing_s/gen` as top-level convenience KPIs
    - drops `perf/time_per_step` (duplicate of `timing_s/step`)

- `postprocess_energy_metrics.py`
  - Computes phase-level derived metrics (energy deltas, avg power, throttle fraction, correlations, J/token).
  - Example:
    - `python3 profiling_scripts/postprocess_energy_metrics.py --monitor-dir /path/to/monitoring/<experiment>`
