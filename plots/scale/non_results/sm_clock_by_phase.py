"""SM clock by configuration for llama scaling runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper, scatter_paper


OUTPATH = Path("plots/out/scale/sm_clock_by_phase.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
CONFIG_COLORS = {
    "2xA100": "#E76F51",
    "2xH200": "#2A9D8F",
    "4xA100": "#E9C46A",
    "4xH200": "#457B9D",
}

FIGURE_TITLE_SIZE = 18
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12


def _config_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_a100" in rid:
        return "2xA100"
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    if "4gpu_h200" in rid:
        return "4xH200"
    return "Unknown"


def _select_clean_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")

    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(
        summary_df[
            [
                "run_id",
                "join_coverage_rate",
                "phase_boundary_integrity_rate",
                "is_checkpoint_continuation",
            ]
        ],
        on="run_id",
        how="inner",
        validate="one_to_one",
    )
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[
        selected["config"].isin(CONFIG_ORDER)
        & (pd.to_numeric(selected["join_coverage_rate"], errors="coerce") == 1.0)
        & (pd.to_numeric(selected["phase_boundary_integrity_rate"], errors="coerce") == 1.0)
        & (~selected["is_checkpoint_continuation"].fillna(False).astype(bool))
    ].copy()
    if selected.empty:
        raise ValueError("No clean llama_scaling runs selected.")
    return selected[["run_id", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_clean_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    ddf, _ = load_view("device_timeseries_view")
    needed = [
        "run_id",
        "global_step_canonical",
        "sm_clock_mhz",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    df = ddf[ddf["run_id"].astype(str).isin(run_ids)][[c for c in needed if c in ddf.columns]].copy()
    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["sm_clock_mhz"] = pd.to_numeric(df["sm_clock_mhz"], errors="coerce")
    df = df.dropna(subset=["global_step_canonical", "sm_clock_mhz"]).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    step_means = (
        df.groupby(["run_id", "config", "global_step_canonical"], dropna=False)["sm_clock_mhz"]
        .mean()
        .reset_index()
    )
    run_means = (
        step_means.groupby(["run_id", "config"], dropna=False)["sm_clock_mhz"]
        .mean()
        .reset_index()
        .rename(columns={"sm_clock_mhz": "run_mean_sm_clock_mhz"})
    )

    print("run-level SM clock means:")
    print(run_means.sort_values(["config", "run_id"]).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    x = np.arange(len(CONFIG_ORDER), dtype=float)

    for idx, config in enumerate(CONFIG_ORDER):
        sub_step = step_means[step_means["config"] == config]
        if not sub_step.empty:
            ax.scatter(
                np.full(len(sub_step), x[idx], dtype=float),
                sub_step["sm_clock_mhz"],
                s=26,
                color=CONFIG_COLORS[config],
                alpha=0.18,
                linewidths=0,
                zorder=2,
            )

        sub_run = run_means[run_means["config"] == config]
        if not sub_run.empty:
            ax.scatter(
                np.full(len(sub_run), x[idx], dtype=float),
                sub_run["run_mean_sm_clock_mhz"],
                s=180,
                color=CONFIG_COLORS[config],
                edgecolors="black",
                linewidths=1.0,
                zorder=3,
            )

    scatter_paper(ax)
    ax.set_title("SM Clock by Configuration", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", pad=12)
    ax.set_xlabel("Configuration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("SM clock (MHz)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x, [CONFIG_DISPLAY[c] for c in CONFIG_ORDER], rotation=0)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="sm_clock_by_phase",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["runs", "run_summary_view", "device_timeseries_view"],
            "selection": "clean scaling runs",
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
