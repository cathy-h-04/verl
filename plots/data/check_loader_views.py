"""Acceptance script for plots.data.loader."""

from __future__ import annotations

from plots.data.loader import KNOWN_VIEWS, load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


def main() -> None:
    for view_name in sorted(KNOWN_VIEWS):
        df, meta = load_view(view_name)
        df_filtered = apply_analysis_ok(df)
        filtering = explain_filtering(df, df_filtered)
        print(f"{view_name}: shape={df.shape}")
        print(f"  columns={list(df.columns)}")
        print(f"  dataset_version={meta['dataset_version']} schema_version={meta['schema_version']}")
        print(
            "  filtering:"
            f" rows_before={filtering['rows_before']}"
            f" rows_after={filtering['rows_after']}"
            f" rows_removed={filtering['rows_removed']}"
        )
        print(
            "  startup_rule:"
            f" enabled={filtering.get('startup_rule_enabled')}"
            f" n={filtering.get('startup_iterations_n')}"
            f" eligible_run_count={filtering.get('eligible_run_count')}"
            f" reason={filtering.get('startup_rule_reason')}"
        )
        if filtering["reasons"]:
            print(f"  drop_reasons={filtering['reasons']}")


if __name__ == "__main__":
    main()
