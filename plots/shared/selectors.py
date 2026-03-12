"""Run selectors and shared analysis_ok rule."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Iterable, Sequence

import pandas as pd

from plots.data.manifest import RunManifest, build_run_manifest
from plots.shared.loaders import RunRecord


def select_runs_by_id(runs: Iterable[RunRecord], run_ids: Sequence[str]) -> list[RunRecord]:
    """Select runs by explicit IDs, preserving user-specified order."""
    run_index = {run.run_id: run for run in runs}
    missing_run_ids = [run_id for run_id in run_ids if run_id not in run_index]
    if missing_run_ids:
        missing = ", ".join(missing_run_ids)
        raise ValueError(f"Missing run IDs: {missing}")
    return [run_index[run_id] for run_id in run_ids]


def analysis_ok(run: RunRecord) -> bool:
    """Shared rule every plot must apply before plotting."""
    return (
        run.run_config_path is not None
        and run.tokens_and_steps_path is not None
        and run.primary_metrics_path is not None
    )


def select_run_dataframe_with_manifest(
    runs: Iterable[RunRecord],
    *,
    selected_run_ids: Sequence[str],
    plot_name: str,
    grouping_label: str,
    data_sources: dict[str, Any] | None = None,
    row_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    explain_filter: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]] | None = None,
    return_debug_info: bool = False,
) -> tuple[pd.DataFrame, RunManifest] | tuple[pd.DataFrame, RunManifest, dict[str, Any]]:
    """Build run dataframe + RunManifest without exposing schema details to plots."""
    runs_by_id = {run.run_id: run for run in runs}
    ordered_runs = [runs_by_id[run_id] for run_id in selected_run_ids if run_id in runs_by_id]
    rows: list[dict[str, Any]] = []
    for run in ordered_runs:
        row = run.to_manifest_dict()
        row["analysis_ok"] = analysis_ok(run)
        rows.append(row)

    df_before = pd.DataFrame(rows)
    df = row_filter(df_before) if row_filter is not None else df_before.copy()
    filtered_rows = df.to_dict(orient="records")
    included_ids = [str(run_id) for run_id in df.get("run_id", pd.Series(dtype=str)).tolist()]
    dropped_ids = [run_id for run_id in selected_run_ids if run_id not in set(included_ids)]
    manifest = build_run_manifest(
        plot_name=plot_name,
        run_rows=filtered_rows,
        data_sources={
            "selector_group": grouping_label,
            "views": [],
            **(data_sources or {}),
        },
    )

    if not return_debug_info:
        return df, manifest

    debug_info = {
        "selected_run_ids": list(selected_run_ids),
        "included_run_ids": included_ids,
        "dropped_run_ids": dropped_ids,
        "missing_count": len(dropped_ids),
        "filtering": explain_filter(df_before, df) if explain_filter is not None else None,
    }
    return df, manifest, debug_info
