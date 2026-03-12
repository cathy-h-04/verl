"""Shared default analysis filtering for analysis-ready view dataframes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# If analysis_ok is not precomputed, derive default predicate from these columns when present.
INTEGRITY_TRUE_COLUMNS = ("boundary_integrity_ok", "join_integrity_ok")
MASK_FALSE_COLUMNS = (
    "is_warmup_idle",
    "is_validation_step",
    "is_outlier_sample",
    "is_incomplete_phase",
)
ROW_LEVEL_STARTUP_EXCLUSION_VIEWS = {"phase_fact_view", "step_fact_view", "device_timeseries_view"}
STARTUP_ITERATIONS_TO_EXCLUDE = 5

# dataset_root (resolved str) -> set(run_id)
_STARTUP_ELIGIBLE_RUN_CACHE: dict[str, set[str]] = {}


def apply_analysis_ok(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared analysis predicate and return filtered rows.

    Rule:
    - If `analysis_ok` exists: keep rows where `analysis_ok == True`.
    - Otherwise: derive predicate from integrity + mask columns that are present.
    """
    if df.empty:
        return df.copy()

    predicate = _analysis_predicate(df)
    startup_mask = _startup_exclusion_mask(df)
    if startup_mask is not None:
        predicate &= ~startup_mask
    return df.loc[predicate].copy()


def explain_filtering(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict[str, Any]:
    """Return auditable filtering summary with reason breakdown when columns exist."""
    rows_before = int(len(df_before))
    rows_after = int(len(df_after))
    removed_count = rows_before - rows_after
    startup_info = _startup_rule_info(df_before)

    if rows_before == 0:
        return {
            "rows_before": 0,
            "rows_after": 0,
            "rows_removed": 0,
            "predicate_mode": "empty",
            "reasons": {"startup_iterations_excluded": 0},
            "startup_rule_enabled": startup_info["enabled"],
            "startup_iterations_n": STARTUP_ITERATIONS_TO_EXCLUDE,
            "eligible_run_count": startup_info["eligible_run_count"],
            "startup_rule_reason": startup_info["reason"],
        }

    reason_masks = _reason_masks(df_before)
    removed_mask = _removed_mask(df_before=df_before, df_after=df_after)
    reasons = {reason: int((mask & removed_mask).sum()) for reason, mask in reason_masks.items()}
    startup_mask = _startup_exclusion_mask(df_before)
    reasons["startup_iterations_excluded"] = int((startup_mask & removed_mask).sum()) if startup_mask is not None else 0

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": removed_count,
        "predicate_mode": "analysis_ok_column" if "analysis_ok" in df_before.columns else "derived_from_integrity_masks",
        "reasons": reasons,
        "columns_used": sorted({c for c in ("analysis_ok", *INTEGRITY_TRUE_COLUMNS, *MASK_FALSE_COLUMNS) if c in df_before.columns}),
        "startup_rule_enabled": startup_info["enabled"],
        "startup_iterations_n": STARTUP_ITERATIONS_TO_EXCLUDE,
        "eligible_run_count": startup_info["eligible_run_count"],
        "startup_rule_reason": startup_info["reason"],
    }


def _analysis_predicate(df: pd.DataFrame) -> pd.Series:
    if "analysis_ok" in df.columns:
        return _as_bool_series(df["analysis_ok"], default=False)

    predicate = pd.Series(True, index=df.index, dtype=bool)

    for column in INTEGRITY_TRUE_COLUMNS:
        if column in df.columns:
            predicate &= _as_bool_series(df[column], default=False)

    for column in MASK_FALSE_COLUMNS:
        if column in df.columns:
            predicate &= ~_as_bool_series(df[column], default=False)

    return predicate


def _reason_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}

    if "analysis_ok" in df.columns:
        masks["analysis_ok_false_or_missing"] = ~_as_bool_series(df["analysis_ok"], default=False)
        return masks

    for column in INTEGRITY_TRUE_COLUMNS:
        if column in df.columns:
            masks[f"{column}_failed"] = ~_as_bool_series(df[column], default=False)

    for column in MASK_FALSE_COLUMNS:
        if column in df.columns:
            masks[f"{column}_excluded"] = _as_bool_series(df[column], default=False)

    return masks


def _startup_exclusion_mask(df: pd.DataFrame) -> pd.Series | None:
    info = _startup_rule_info(df)
    if not info["enabled"]:
        return None

    step_col = info["step_col"]
    eligible_runs: set[str] = info["eligible_run_ids"]
    if step_col is None or not eligible_runs:
        return None

    run_ids = df["run_id"].astype(str)
    steps = pd.to_numeric(df[step_col], errors="coerce")
    mask = pd.Series(False, index=df.index, dtype=bool)

    for run_id, grp in df.assign(_run_id=run_ids, _step=steps).groupby("_run_id", dropna=False):
        if run_id not in eligible_runs:
            continue
        step_values = sorted(v for v in grp["_step"].dropna().unique())
        if not step_values:
            continue
        startup_steps = set(step_values[:STARTUP_ITERATIONS_TO_EXCLUDE])
        grp_mask = grp["_step"].isin(startup_steps)
        mask.loc[grp.index] = grp_mask.values

    return mask


def _startup_rule_info(df: pd.DataFrame) -> dict[str, Any]:
    if "run_id" not in df.columns:
        return {
            "enabled": False,
            "reason": "run_id_missing",
            "step_col": None,
            "eligible_run_ids": set(),
            "eligible_run_count": 0,
        }

    views = _extract_views(df)
    if not views:
        return {
            "enabled": False,
            "reason": "view_metadata_missing",
            "step_col": None,
            "eligible_run_ids": set(),
            "eligible_run_count": 0,
        }
    if not any(view in ROW_LEVEL_STARTUP_EXCLUSION_VIEWS for view in views):
        return {
            "enabled": False,
            "reason": f"view_not_row_level:{views}",
            "step_col": None,
            "eligible_run_ids": set(),
            "eligible_run_count": 0,
        }

    step_col = "global_step_canonical" if "global_step_canonical" in df.columns else "global_step" if "global_step" in df.columns else None
    if step_col is None:
        return {
            "enabled": False,
            "reason": "step_column_missing",
            "step_col": None,
            "eligible_run_ids": set(),
            "eligible_run_count": 0,
        }

    dataset_root = _extract_dataset_root(df)
    if dataset_root is None:
        return {
            "enabled": False,
            "reason": "dataset_root_missing",
            "step_col": step_col,
            "eligible_run_ids": set(),
            "eligible_run_count": 0,
        }

    eligible_run_ids = _load_startup_eligible_run_ids(dataset_root)
    return {
        "enabled": True,
        "reason": "ok",
        "step_col": step_col,
        "eligible_run_ids": eligible_run_ids,
        "eligible_run_count": len(eligible_run_ids),
    }


def _extract_views(df: pd.DataFrame) -> list[str]:
    source = df.attrs.get("data_source", {})
    views = source.get("views") if isinstance(source, dict) else None
    if isinstance(views, list):
        return [str(v) for v in views]
    if isinstance(views, str):
        return [views]
    return []


def _extract_dataset_root(df: pd.DataFrame) -> str | None:
    source = df.attrs.get("data_source", {})
    root = source.get("dataset_root") if isinstance(source, dict) else None
    if root is None:
        return None
    return str(Path(root).expanduser().resolve())


def _load_startup_eligible_run_ids(dataset_root: str) -> set[str]:
    if dataset_root in _STARTUP_ELIGIBLE_RUN_CACHE:
        return _STARTUP_ELIGIBLE_RUN_CACHE[dataset_root]

    runs_path = Path(dataset_root) / "runs.parquet"
    if not runs_path.exists():
        _STARTUP_ELIGIBLE_RUN_CACHE[dataset_root] = set()
        return set()

    runs = pd.read_parquet(runs_path, columns=["run_id", "is_resumed_run", "configured_total_steps", "total_epochs"])
    resumed = _as_bool_series(runs["is_resumed_run"], default=False) if "is_resumed_run" in runs.columns else pd.Series(False, index=runs.index)
    configured_steps_null = runs["configured_total_steps"].isna() if "configured_total_steps" in runs.columns else pd.Series(False, index=runs.index)
    total_epochs_set = runs["total_epochs"].notna() if "total_epochs" in runs.columns else pd.Series(False, index=runs.index)
    eligible_mask = (~resumed) & configured_steps_null & total_epochs_set

    eligible_ids = set(runs.loc[eligible_mask, "run_id"].dropna().astype(str).tolist())
    _STARTUP_ELIGIBLE_RUN_CACHE[dataset_root] = eligible_ids
    return eligible_ids


def _removed_mask(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.Series:
    if set(df_after.index).issubset(set(df_before.index)):
        kept_idx = set(df_after.index)
        return pd.Series([idx not in kept_idx for idx in df_before.index], index=df_before.index, dtype=bool)

    # Fallback for index-destructive transforms: infer only count-level removal.
    removed = max(int(len(df_before) - len(df_after)), 0)
    mask = pd.Series(False, index=df_before.index, dtype=bool)
    if removed > 0:
        mask.iloc[:removed] = True
    return mask


def _as_bool_series(series: pd.Series, *, default: bool) -> pd.Series:
    normalized = series.copy()

    if pd.api.types.is_bool_dtype(normalized):
        return normalized.fillna(default)

    if pd.api.types.is_numeric_dtype(normalized):
        return normalized.fillna(1 if default else 0).astype(float).astype(bool)

    lowered = normalized.astype(str).str.lower()
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", "none", "nan", ""}

    out = pd.Series(default, index=series.index, dtype=bool)
    out[lowered.isin(true_values)] = True
    out[lowered.isin(false_values)] = False
    return out
