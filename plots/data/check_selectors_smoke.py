"""Smoke test for auditable run selectors."""

from __future__ import annotations

from plots.data.loader import load_view
from plots.data.selectors import select_baseline, select_comparison_group, select_runs_by_ids


def main() -> None:
    df_runs, _ = load_view("run_summary_view")
    print(f"loaded run_summary_view: rows={len(df_runs)} columns={len(df_runs.columns)}")

    run_ids = df_runs["run_id"].astype(str).head(2).tolist()
    by_ids_df, _ = select_runs_by_ids(run_ids, df_runs)
    print(f"select_runs_by_ids rows={len(by_ids_df)} run_ids={run_ids}")

    baseline_model = None
    if "model" in df_runs.columns:
        non_null_models = [m for m in df_runs["model"].astype(str).tolist() if m and m.lower() != "nan"]
        baseline_model = non_null_models[0] if non_null_models else None
    baseline_df, _ = select_baseline(df_runs, model=baseline_model)
    print(f"select_baseline rows={len(baseline_df)} model={baseline_model}")

    group_label = None
    for col in ("logical_run_group", "experiment_variant", "variant_tags", "group_label"):
        if col not in df_runs.columns:
            continue
        non_null_groups = [g for g in df_runs[col].astype(str).tolist() if g and g.lower() != "nan"]
        if non_null_groups:
            group_label = non_null_groups[0]
            break
    if group_label is None:
        raise RuntimeError("No group label candidate found in run_summary_view for comparison selector smoke test.")

    comp_df, _ = select_comparison_group(df_runs, group_label=group_label)
    print(f"select_comparison_group rows={len(comp_df)} group_label={group_label}")


if __name__ == "__main__":
    main()

