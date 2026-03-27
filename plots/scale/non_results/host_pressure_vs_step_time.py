"""Host-pressure test: CPU load/utilization versus step time for scaling runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/host_pressure_vs_step_time.png")
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
X_SPECS = (
    ("load1_per_cpu", "load1 / cpus_per_task", True),
    ("load5_per_cpu", "load5 / cpus_per_task", True),
    ("load15_per_cpu", "load15 / cpus_per_task", True),
    ("cpu_util_pct_total", "cpu_util_pct_total", False),
)

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 13
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 11


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


def _select_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id", "slurm_cpus_per_task"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected["slurm_cpus_per_task"] = pd.to_numeric(selected["slurm_cpus_per_task"], errors="coerce")
    selected = selected[selected["config"].isin(CONFIG_ORDER)].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    if selected.empty:
        raise ValueError("No llama_scaling runs selected.")
    return selected[["run_id", "policy_norm", "config", "slurm_cpus_per_task"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    filter_cols = [
        "run_id",
        "global_step_canonical",
        "step_time_s",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    filter_cols = [c for c in filter_cols if c in step_fact.columns]
    sf = step_fact[step_fact["run_id"].astype(str).isin(run_ids)][filter_cols].copy()
    before = sf.copy()
    sf = apply_analysis_ok(sf)
    print(f"step_filtering={explain_filtering(before, sf)}")
    sf["global_step_canonical"] = pd.to_numeric(sf["global_step_canonical"], errors="coerce")
    sf["step_time_s"] = pd.to_numeric(sf["step_time_s"], errors="coerce")
    sf = sf.dropna(subset=["global_step_canonical", "step_time_s"]).copy()
    sf["global_step_canonical"] = sf["global_step_canonical"].astype(int)

    hp, _ = load_view("hardware_periodic")
    cols = ["run_id", "global_step_canonical", "ts_monotonic_ns", "load1", "load5", "load15", "cpu_util_pct_total"]
    hp = hp[hp["run_id"].astype(str).isin(run_ids)][cols].copy()
    hp["global_step_canonical"] = pd.to_numeric(hp["global_step_canonical"], errors="coerce")
    hp["ts_monotonic_ns"] = pd.to_numeric(hp["ts_monotonic_ns"], errors="coerce")
    for c in ["load1", "load5", "load15", "cpu_util_pct_total"]:
        hp[c] = pd.to_numeric(hp[c], errors="coerce")
    hp = hp.dropna(subset=["global_step_canonical", "ts_monotonic_ns"]).copy()
    hp["global_step_canonical"] = hp["global_step_canonical"].astype(int)

    # Deduplicate the per-domain RAPL rows by timestamp within each step.
    hp = (
        hp.groupby(["run_id", "global_step_canonical", "ts_monotonic_ns"], dropna=False)[["load1", "load5", "load15", "cpu_util_pct_total"]]
        .mean()
        .reset_index()
    )
    hp_step = (
        hp.groupby(["run_id", "global_step_canonical"], dropna=False)[["load1", "load5", "load15", "cpu_util_pct_total"]]
        .mean()
        .reset_index()
    )

    df = sf.merge(hp_step, on=["run_id", "global_step_canonical"], how="left")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    df["cpus_per_task"] = df["slurm_cpus_per_task"].replace(0, np.nan)
    df["load1_per_cpu"] = df["load1"] / df["cpus_per_task"]
    df["load5_per_cpu"] = df["load5"] / df["cpus_per_task"]
    df["load15_per_cpu"] = df["load15"] / df["cpus_per_task"]

    run_means = (
        df.groupby(["run_id", "config"], dropna=False)[["step_time_s", "load1_per_cpu", "load5_per_cpu", "load15_per_cpu", "cpu_util_pct_total"]]
        .mean()
        .reset_index()
        .sort_values(["config", "run_id"])
    )
    print("run-level host pressure summary:")
    print(run_means.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.0), sharey=True)
    axes_flat = axes.flatten()
    for ax, (x_col, x_label, show_threshold) in zip(axes_flat, X_SPECS):
        for config in CONFIG_ORDER:
            sub = df[df["config"] == config].dropna(subset=[x_col, "step_time_s"]).copy()
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub["step_time_s"],
                s=16,
                alpha=0.16,
                color=CONFIG_COLORS[config],
                edgecolors="none",
                zorder=1,
            )
            means = run_means[run_means["config"] == config].dropna(subset=[x_col, "step_time_s"]).copy()
            if not means.empty:
                ax.scatter(
                    means[x_col],
                    means["step_time_s"],
                    s=92,
                    marker="D",
                    color=CONFIG_COLORS[config],
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
        if show_threshold:
            ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(x_label, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
        ax.grid(axis="both", alpha=0.24, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    axes[0, 0].set_ylabel("timing_s/step", fontsize=AXIS_LABEL_SIZE)
    axes[1, 0].set_ylabel("timing_s/step", fontsize=AXIS_LABEL_SIZE)

    config_handles = [
        Line2D([0], [0], marker="D", linestyle="None", markersize=7, markerfacecolor=CONFIG_COLORS[cfg], markeredgecolor="black", label=CONFIG_DISPLAY[cfg])
        for cfg in CONFIG_ORDER
    ]
    point_handle = Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor="#888888", markeredgecolor="none", alpha=0.35, label="Step-level points")
    threshold_handle = Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2, label="load/core = 1")
    fig.suptitle("Host Pressure vs Step Time", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=config_handles + [point_handle, threshold_handle], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="host_pressure_vs_step_time",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["step_fact_view", "hardware_periodic", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
