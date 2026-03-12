"""Auditable run selectors that return explicit run sets + RunManifest."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from plots.data.manifest import RunManifest, build_run_manifest, summarize_manifest


def select_runs_by_ids(
    run_ids: Sequence[str],
    df: pd.DataFrame,
    *,
    model: str | None = None,
    policy: str | None = None,
    date_tag: str | None = None,
) -> tuple[pd.DataFrame, RunManifest]:
    """Select an explicit list of run IDs and return `(df_subset, manifest)`."""
    if not run_ids:
        raise ValueError("select_runs_by_ids requires a non-empty run_ids list.")
    _require_columns(df, ["run_id"], "select_runs_by_ids")

    unique_ids = list(dict.fromkeys(str(run_id) for run_id in run_ids))
    subset = df[df["run_id"].astype(str).isin(unique_ids)].copy()
    subset = _apply_optional_constraints(subset, model=model, policy=policy, date_tag=date_tag, selector_name="select_runs_by_ids")
    subset = _reorder_by_run_ids(subset, unique_ids)

    return _finalize_selection(
        selector_name="select_runs_by_ids",
        grouping_label="explicit_run_ids",
        source_df=df,
        subset=subset,
        selected_run_ids=unique_ids,
        constraints={"model": model, "policy": policy, "date_tag": date_tag},
    )


def select_baseline(
    df_runs: pd.DataFrame,
    model: str | None = None,
    policy: str | None = None,
    date_tag: str | None = None,
) -> tuple[pd.DataFrame, RunManifest]:
    """Select baseline runs and return `(df_subset, manifest)`.

    Baseline default:
    - if `is_checkpoint_continuation` exists, keep rows where it is False.
    """
    _require_columns(df_runs, ["run_id"], "select_baseline")
    subset = df_runs.copy()
    criteria: list[str] = []

    if "is_checkpoint_continuation" in subset.columns:
        continuation = _as_bool_series(subset["is_checkpoint_continuation"], default=False)
        subset = subset[~continuation].copy()
        criteria.append("is_checkpoint_continuation == False")

    subset = _apply_optional_constraints(subset, model=model, policy=policy, date_tag=date_tag, selector_name="select_baseline")
    if not criteria and model is None and policy is None and date_tag is None:
        raise ValueError(
            "select_baseline refused to include all runs implicitly. "
            "Pass at least one constraint (model/policy/date_tag), or provide `is_checkpoint_continuation` in df_runs."
        )

    return _finalize_selection(
        selector_name="select_baseline",
        grouping_label="baseline",
        source_df=df_runs,
        subset=subset,
        selected_run_ids=subset["run_id"].astype(str).tolist(),
        constraints={"model": model, "policy": policy, "date_tag": date_tag, "criteria": criteria},
    )


def select_comparison_group(
    df_runs: pd.DataFrame,
    group_label: str,
    *,
    model: str | None = None,
    policy: str | None = None,
    date_tag: str | None = None,
) -> tuple[pd.DataFrame, RunManifest]:
    """Select a named comparison group and return `(df_subset, manifest)`."""
    _require_columns(df_runs, ["run_id"], "select_comparison_group")
    label = group_label.strip()
    if not label:
        raise ValueError("select_comparison_group requires a non-empty group_label.")

    subset, used_column = _filter_group_label(df_runs, label=label)
    subset = _apply_optional_constraints(subset, model=model, policy=policy, date_tag=date_tag, selector_name="select_comparison_group")

    return _finalize_selection(
        selector_name="select_comparison_group",
        grouping_label=label,
        source_df=df_runs,
        subset=subset,
        selected_run_ids=subset["run_id"].astype(str).tolist(),
        constraints={"group_label": label, "group_column": used_column, "model": model, "policy": policy, "date_tag": date_tag},
    )


def _filter_group_label(df_runs: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, str]:
    candidate_columns = ["logical_run_group", "experiment_variant", "variant_tags", "group_label"]
    available_columns = [col for col in candidate_columns if col in df_runs.columns]
    if not available_columns:
        raise ValueError(
            "select_comparison_group could not find any grouping columns. "
            f"Expected one of {candidate_columns}, got columns: {list(df_runs.columns)}"
        )

    exact_matches: list[tuple[str, pd.Series]] = []
    contains_matches: list[tuple[str, pd.Series]] = []
    for col in available_columns:
        values = df_runs[col].astype(str)
        exact = values.str.lower() == label.lower()
        contains = values.str.contains(label, case=False, na=False)
        exact_matches.append((col, exact))
        contains_matches.append((col, contains))

    for col, mask in exact_matches:
        if mask.any():
            return df_runs[mask].copy(), col

    for col, mask in contains_matches:
        if mask.any():
            return df_runs[mask].copy(), col

    # Return empty with first available column name for diagnostic context.
    return df_runs.iloc[0:0].copy(), available_columns[0]


def _apply_optional_constraints(
    df: pd.DataFrame,
    *,
    model: str | None,
    policy: str | None,
    date_tag: str | None,
    selector_name: str,
) -> pd.DataFrame:
    subset = df

    if model is not None:
        _require_columns(subset, ["model"], selector_name)
        subset = subset[subset["model"].astype(str) == str(model)].copy()

    if policy is not None:
        _require_columns(subset, ["policy"], selector_name)
        subset = subset[subset["policy"].astype(str) == str(policy)].copy()

    if date_tag is not None:
        _require_columns(subset, ["run_id"], selector_name)
        subset = subset[subset["run_id"].astype(str).str.contains(str(date_tag), na=False)].copy()

    return subset


def _finalize_selection(
    *,
    selector_name: str,
    grouping_label: str,
    source_df: pd.DataFrame,
    subset: pd.DataFrame,
    selected_run_ids: Sequence[str],
    constraints: dict[str, Any],
) -> tuple[pd.DataFrame, RunManifest]:
    if subset.empty:
        raise ValueError(_build_empty_selection_message(selector_name, grouping_label, source_df, selected_run_ids, constraints))

    data_sources = _infer_data_sources(source_df)
    data_sources["selector_group"] = grouping_label
    data_sources["selector_name"] = selector_name

    manifest = build_run_manifest(
        plot_name=f"selector:{selector_name}",
        run_rows=subset.to_dict(orient="records"),
        data_sources=data_sources,
    )
    summarize_manifest(manifest)
    return subset, manifest


def _build_empty_selection_message(
    selector_name: str,
    grouping_label: str,
    source_df: pd.DataFrame,
    selected_run_ids: Sequence[str],
    constraints: dict[str, Any],
) -> str:
    available_models = sorted(source_df["model"].dropna().astype(str).unique().tolist())[:10] if "model" in source_df.columns else []
    available_policies = sorted(source_df["policy"].dropna().astype(str).unique().tolist())[:10] if "policy" in source_df.columns else []
    available_groups: list[str] = []
    for col in ("logical_run_group", "experiment_variant", "variant_tags", "group_label"):
        if col in source_df.columns:
            available_groups = sorted(source_df[col].dropna().astype(str).unique().tolist())[:10]
            if available_groups:
                break

    return (
        f"{selector_name} produced 0 rows for grouping_label='{grouping_label}'. "
        f"selected_run_ids={list(selected_run_ids)} constraints={constraints}. "
        f"input_rows={len(source_df)}. "
        f"Hint: verify run_id spellings and constraint values. "
        f"available_models(sample)={available_models}; "
        f"available_policies(sample)={available_policies}; "
        f"available_groups(sample)={available_groups}."
    )


def _require_columns(df: pd.DataFrame, required: Sequence[str], selector_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{selector_name} missing required columns {missing}. available_columns={list(df.columns)}")


def _reorder_by_run_ids(df: pd.DataFrame, ordered_run_ids: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    rank = {run_id: idx for idx, run_id in enumerate(ordered_run_ids)}
    out = df.copy()
    out["__run_id_rank__"] = out["run_id"].astype(str).map(rank).fillna(len(rank))
    out = out.sort_values("__run_id_rank__").drop(columns="__run_id_rank__")
    return out


def _as_bool_series(series: pd.Series, *, default: bool) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(1 if default else 0).astype(float).astype(bool)
    values = series.astype(str).str.lower()
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", "none", "nan", ""}
    out = pd.Series(default, index=series.index, dtype=bool)
    out[values.isin(true_values)] = True
    out[values.isin(false_values)] = False
    return out


def _infer_data_sources(df: pd.DataFrame) -> dict[str, Any]:
    source = df.attrs.get("data_source", {})
    if isinstance(source, dict):
        return dict(source)
    return {}
