"""Targeted smoke checks for startup-iteration exclusion in shared filtering."""

from __future__ import annotations

import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import STARTUP_ITERATIONS_TO_EXCLUDE, apply_analysis_ok, explain_filtering


def _run_level_removed_steps(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict[str, list[int]]:
    step_col = "global_step_canonical" if "global_step_canonical" in df_before.columns else "global_step"
    before = df_before[["run_id", step_col]].copy()
    after = df_after[["run_id", step_col]].copy()

    before["step"] = pd.to_numeric(before[step_col], errors="coerce")
    after["step"] = pd.to_numeric(after[step_col], errors="coerce")
    before = before.dropna(subset=["run_id", "step"])
    after = after.dropna(subset=["run_id", "step"])

    removed: dict[str, list[int]] = {}
    for run_id, grp in before.groupby("run_id", dropna=False):
        before_steps = set(int(v) for v in grp["step"].tolist())
        after_steps = set(int(v) for v in after.loc[after["run_id"] == run_id, "step"].tolist())
        missing = sorted(before_steps - after_steps)
        if missing:
            removed[str(run_id)] = missing
    return removed


def _assert_startup_behavior(df_before: pd.DataFrame, df_after: pd.DataFrame, *, view_name: str) -> None:
    removed = _run_level_removed_steps(df_before, df_after)

    runs_df = pd.read_parquet("DATASETS/runs.parquet", columns=["run_id", "is_resumed_run", "configured_total_steps", "total_epochs"])
    resumed = runs_df["is_resumed_run"].fillna(False).astype(bool)
    eligible = runs_df[(~resumed) & runs_df["configured_total_steps"].isna() & runs_df["total_epochs"].notna()]["run_id"].astype(str)
    ineligible = set(runs_df["run_id"].astype(str)) - set(eligible.tolist())

    for run_id in eligible:
        run_id = str(run_id)
        if run_id not in removed:
            continue
        first_removed = removed[run_id][:STARTUP_ITERATIONS_TO_EXCLUDE]
        expected = list(range(1, STARTUP_ITERATIONS_TO_EXCLUDE + 1))
        if first_removed != expected:
            raise AssertionError(
                f"{view_name}: eligible run {run_id} did not remove expected startup steps {expected}; got {first_removed}"
            )

    for run_id in ineligible:
        if run_id in removed and any(step <= STARTUP_ITERATIONS_TO_EXCLUDE for step in removed[run_id]):
            raise AssertionError(
                f"{view_name}: ineligible run {run_id} had startup exclusion applied unexpectedly: {removed[run_id]}"
            )


def _check_view(view_name: str) -> None:
    df_before, _ = load_view(view_name)
    df_after = apply_analysis_ok(df_before)
    info = explain_filtering(df_before, df_after)
    removed = _run_level_removed_steps(df_before, df_after)

    print(f"{view_name}: rows_before={len(df_before)} rows_after={len(df_after)}")
    print(f"  startup_rule={info.get('startup_rule_enabled')} reason={info.get('startup_rule_reason')}")
    print(f"  startup_removed_rows={info.get('reasons', {}).get('startup_iterations_excluded', 0)}")
    sample = {k: v[:8] for k, v in list(removed.items())[:4]}
    print(f"  removed_steps_sample={sample}")

    _assert_startup_behavior(df_before, df_after, view_name=view_name)


def main() -> None:
    _check_view("phase_fact_view")
    _check_view("step_fact_view")

    # Non-row-level view should not apply startup rule.
    df_comp, _ = load_view("comparison_view")
    df_comp_after = apply_analysis_ok(df_comp)
    comp_info = explain_filtering(df_comp, df_comp_after)
    print(
        "comparison_view:"
        f" rows_before={len(df_comp)} rows_after={len(df_comp_after)}"
        f" startup_rule={comp_info.get('startup_rule_enabled')}"
        f" reason={comp_info.get('startup_rule_reason')}"
    )
    if comp_info.get("startup_rule_enabled"):
        raise AssertionError("comparison_view unexpectedly had startup rule enabled.")


if __name__ == "__main__":
    main()

