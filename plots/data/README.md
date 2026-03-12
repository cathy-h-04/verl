# Data Loader

`plots/data/loader.py` provides a tiny API for loading analysis-ready views as pandas DataFrames.

## Expected View Locations

By default, `load_view(...)` reads from `DATASETS/` at repo root:

- `DATASETS/phase_fact_view.parquet`
- `DATASETS/step_fact_view.parquet`
- `DATASETS/run_summary_view.parquet`
- `DATASETS/comparison_view.parquet`

You can pass a different root with `dataset_root=...`.

## API

```python
from plots.data.loader import load_view

df, meta = load_view("phase_fact_view")
```

Optional explicit filtering/column selection (no hidden filters in loader):

```python
df, meta = load_view(
    "comparison_view",
    columns=["policy", "model", "total_energy_j_mean"],
    row_filter={"policy": ["ppo", "grpo"]},
)
```

## Metadata Returned

`load_view(...)` returns `(df, metadata)` where metadata includes:

- `dataset_version` (explicit from metadata/version files if available, otherwise computed fingerprint)
- `schema_version` (explicit from parquet metadata if available, otherwise computed schema hash)
- `view_name`, `view_path`, `row_count`, `column_count`, `columns`

## Shared Filtering Behavior

Filtering is not applied inside `load_view(...)`; plots/selectors call `plots.plotting.filters.apply_analysis_ok(...)`.

For row-level views (`phase_fact_view`, `step_fact_view`, `device_timeseries_view`), shared analysis filtering also excludes the first 5 canonical iterations for full-epoch, non-checkpointed runs (`is_resumed_run == False`, `configured_total_steps is null`, `total_epochs is set`).

## Acceptance Script

Run:

```bash
./verl-env/bin/python -m plots.data.check_loader_views
```

This loads each supported view and prints shape + columns.
