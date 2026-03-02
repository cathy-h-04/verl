# Experiment Dataset Builder

This directory contains a standalone dataset ingestion script for experiment artifacts under `results/`.

## Script

- `build_run_datasets.py`

## What It Produces

The script writes Parquet tables to:

- default: `/n/home08/chou/verl_research/DATASETS`

Output files:

1. `runs.parquet`
2. `run_lineage.parquet`
3. `step_metrics_long.parquet`
4. `step_metrics_wide_curated.parquet`
5. `phase_timings_long.parquet`
6. `tokens_and_steps.parquet`
7. `hardware_boundary.parquet`
8. `hardware_periodic.parquet`
9. `phase_summary.parquet`
10. `ingestion_report.parquet`

## Ingestion Rules

- Run discovery scans recursively under `--results-root` for folders containing `experiment_name.txt`.
- Required run files are validated, including dynamic files:
  - `[EXP_NAME].jsonl`
  - `[EXP_NAME]_config.json`
  - `phase_timings_[EXP_NAME].jsonl`
- Incomplete runs are excluded (e.g., zero-line critical JSONL files).
- Runs with JSON parse errors in critical logs are excluded.
- Validation phases/steps are retained.
- Warmup idle rows are removed from periodic hardware logs:
  - `phase_name == "idle"` and `iteration/global_step == 0`

## CLI

```bash
python analysis_scripts/build_run_datasets.py \
  --results-root /n/home08/chou/verl_research/results \
  --output-root /n/home08/chou/verl_research/DATASETS \
  --workers 1 \
  --overwrite
```

Arguments:

- `--results-root`: input root (default `/n/home08/chou/verl_research/results`)
- `--output-root`: output directory (default `/n/home08/chou/verl_research/DATASETS`)
- `--workers`: reserved for future parallel parsing (currently single-process)
- `--overwrite`: remove existing output directory before writing

## Notes on Keys

- Canonical run key: `run_id` (from `experiment_name.txt`)
- Canonical step key: `global_step` (from `step`/`iteration`)
- Canonical phase key: `run_id + global_step + phase_name + phase_id`
- Device identity:
  - NVML: `device_id = gpu_uuid`
  - RAPL: `device_id = rapl_domain`

## Resume Run Handling

Resumed runs remain separate physical runs (`run_id` per folder), with lineage fields in:

- `run_lineage.parquet`
- `runs.parquet`

Fields include:

- `is_resumed_run`
- `resume_path`
- `resume_parent_run_name`
- `resume_from_global_step`
