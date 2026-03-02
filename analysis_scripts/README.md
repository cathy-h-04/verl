# Experiment Dataset Builder

This directory contains a standalone dataset ingestion script for experiment artifacts under `results/`.

## Script

- `build_run_datasets.py`
- `utils.py` (shared ingestion/data transformation utility functions)
- `ingestion_checks.py` (all mismatch/integrity checks and hard-fail rules)
- `periodic_aggregations.py` (shared phase-window + time-weighted periodic utilities)

## What It Produces

The script writes Parquet tables to:

- default: `/n/home08/chou/verl_research/DATASETS`

Output files:

1. `runs.parquet`
2. `run_lineage.parquet`
3. `step_index_map.parquet`
4. `ingestion_checks.parquet`
5. `phase_instances.parquet`
6. `boundary_pair_integrity.parquet`
7. `step_metrics_long.parquet`
8. `step_metrics_wide_curated.parquet`
9. `phase_timings_long.parquet`
10. `tokens_and_steps.parquet`
11. `hardware_boundary.parquet`
12. `hardware_periodic.parquet`
13. `phase_summary.parquet`
14. `phase_fact.parquet`
15. `ingestion_report.parquet`
16. `phase_fact_view.parquet`
17. `step_fact_view.parquet`
18. `run_summary_view.parquet`
19. `comparison_view.parquet`
20. `device_timeseries_view.parquet`
21. `integrity_view.parquet`

## Ingestion Rules

- Run discovery scans recursively under `--results-root` for folders containing `experiment_name.txt`.
- Required run files are validated, including dynamic files:
  - `[EXP_NAME].jsonl`
  - `[EXP_NAME]_config.json`
  - `phase_timings_[EXP_NAME].jsonl`
- Incomplete runs are excluded (e.g., zero-line critical JSONL files).
- Runs with JSON parse errors in critical logs are excluded.
- Canonical step contract:
  - `global_step_canonical` is stored across step-scoped tables.
  - Raw fields are persisted:
    - `global_step_raw_stepfield`
    - `global_step_raw_iterationfield`
  - `step_index_map.parquet` is generated with:
    - `run_id, raw_step, raw_iteration, canonical_step, mismatch_flag, observation_count`
  - Ingestion hard-fails on any non-zero mismatch count.
- Join coverage check:
  - `ingestion_checks.parquet` reports per-run join coverage between step-field and iteration-field domains.
- Phase instance and boundary integrity contract:
  - `phase_instance_id = hash(run_id, global_step_canonical, phase_id, phase_name)`
  - `boundary_pair_key = hash(phase_instance_id, source, device_id)` for boundary rows
  - `phase_instances.parquet` stores:
    - `phase_start_ts_monotonic_ns`
    - `phase_end_ts_monotonic_ns`
  - `boundary_pair_integrity.parquet` stores START/END counts per boundary pair.
  - Ingestion hard-fails if any run has `boundary_pair_integrity < 1.0`.
- Canonical phase fact:
  - `phase_fact.parquet` is one row per `phase_instance_id` with strict schema.
  - It merges:
    - canonical wall-time (`phase_start/end_ts_monotonic_ns`, `phase_duration_s_canonical`)
    - boundary energies (`gpu_energy_j`, `cpu_energy_j`, `dram_energy_j`, `total_energy_j`)
    - phase token denominators (from `tokens_and_steps.jsonl` only)
    - periodic shape metrics as time-weighted averages
  - Analysis mask columns (boolean):
    - `is_warmup_idle`
    - `is_validation_step`
    - `is_incomplete_phase`
    - `is_outlier_sample`
  - Periodic aggregation implementation:
    - phase-window filtering uses monotonic timestamps
    - time-weighted means are computed from adjacent `ts_monotonic_ns` deltas
    - sample means are also stored for drift quantification against time-weighted values
- Validation phases/steps are retained.
- Warmup idle rows are removed from periodic hardware logs:
  - `phase_name == "idle"` and `iteration/global_step == 0`
- Ingestion report includes per-run mask counts:
  - `analysis_mask_is_warmup_idle_count`
  - `analysis_mask_is_validation_step_count`
  - `analysis_mask_is_incomplete_phase_count`
  - `analysis_mask_is_outlier_sample_count`
- Analysis-ready views:
  - `phase_fact_view.parquet`: one row per phase instance with identity, canonical timing/energy, token denominators, derived J/token and share metrics, periodic shape metrics, and integrity booleans.
  - `step_fact_view.parquet`: one row per run-step with phase-summed totals, throughput/efficiency metrics, and derived step J/token + EDP.
  - `run_summary_view.parquet`: one row per run with config snapshot fields, sample counts, mean/median energy efficiency aggregates, and validation outcome summaries.
  - `comparison_view.parquet`: grouped aggregates for plotting/paper comparisons (policy/model/variant/checkpoint continuation).
  - `device_timeseries_view.parquet`: periodic per-device time series for debug plots.
  - `integrity_view.parquet`: per-run integrity status and exclusion diagnostics for notebook filtering.

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
