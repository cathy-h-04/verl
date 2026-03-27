"""Reserved VRAM comparison across scaling configurations, grouped by policy."""

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


OUTPATH = Path("plots/out/scale/non_results/max_memory_reserved_by_configuration.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
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
SUBPLOT_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 12


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
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[selected["policy_norm"].isin(POLICY_ORDER) & selected["config"].isin(CONFIG_ORDER)].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    if selected.empty:
        raise ValueError("No llama_scaling runs selected.")
    return selected[["run_id", "policy_norm", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    wide, _ = load_view("step_metrics_wide_curated")
    step_fact, _ = load_view("step_fact_view")

    filter_df = step_fact[step_fact["run_id"].astype(str).isin(run_ids)][
        [c for c in ["run_id", "global_step_canonical", "boundary_integrity_ok", "join_integrity_ok", "is_warmup_idle", "is_validation_step", "is_incomplete_phase", "is_outlier_sample"] if c in step_fact.columns]
    ].copy()
    before = filter_df.copy()
    filter_df = apply_analysis_ok(filter_df)
    print(f"filtering={explain_filtering(before, filter_df)}")
    keys = filter_df[["run_id", "global_step_canonical"]].drop_duplicates()

    df = wide[wide["run_id"].astype(str).isin(run_ids)][["run_id", "global_step_canonical", "metric_perf_max_memory_reserved_gb"]].copy()
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["metric_perf_max_memory_reserved_gb"] = pd.to_numeric(df["metric_perf_max_memory_reserved_gb"], errors="coerce")
    df = df.dropna(subset=["global_step_canonical", "metric_perf_max_memory_reserved_gb"]).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)
    df = df.merge(keys, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    summary = (
        df.groupby(["policy_norm", "config", "run_id"], dropna=False)["metric_perf_max_memory_reserved_gb"]
        .mean()
        .reset_index(name="run_mean_reserved_gb")
        .sort_values(["policy_norm", "config", "run_id"])
    )
    print("run-level reserved memory summary:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(14.6, 5.4), sharey=True)
    rng = np.random.default_rng(17)
    x_map = {cfg: i for i, cfg in enumerate(CONFIG_ORDER)}
    y_max = float(df["metric_perf_max_memory_reserved_gb"].max()) if not df.empty else 1.0

    for ax, policy in zip(axes, POLICY_ORDER):
        panel = df[df["policy_norm"] == policy].copy()
        if panel.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        box_data = []
        box_pos = []
        for config in CONFIG_ORDER:
            sub = panel[panel["config"] == config]
            if sub.empty:
                continue
            x0 = x_map[config]
            jitter = rng.uniform(-0.15, 0.15, size=len(sub))
            ax.scatter(
                x0 + jitter,
                sub["metric_perf_max_memory_reserved_gb"],
                s=12,
                alpha=0.18,
                color=CONFIG_COLORS[config],
                edgecolors="none",
                zorder=1,
            )
            box_data.append(sub["metric_perf_max_memory_reserved_gb"].to_numpy())
            box_pos.append(x0)
        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=box_pos,
                widths=0.34,
                showfliers=False,
                patch_artist=True,
                boxprops={"edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
            )
            for patch, pos in zip(bp["boxes"], box_pos):
                patch.set_facecolor(CONFIG_COLORS[CONFIG_ORDER[pos]])
                patch.set_alpha(0.35)

        run_means = summary[summary["policy_norm"] == policy]
        for config in CONFIG_ORDER:
            sub = run_means[run_means["config"] == config]
            if sub.empty:
                continue
            x0 = x_map[config]
            ax.scatter(
                [x0],
                sub["run_mean_reserved_gb"],
                s=88,
                marker="D",
                color=CONFIG_COLORS[config],
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )

        ax.set_title(POLICY_DISPLAY[policy], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xticks(range(len(CONFIG_ORDER)))
        ax.set_xticklabels([CONFIG_DISPLAY[cfg] for cfg in CONFIG_ORDER], rotation=0)
        ax.grid(axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.set_ylim(0, y_max * 1.15)

    axes[0].set_ylabel("perf/max_memory_reserved_gb", fontsize=AXIS_LABEL_SIZE)

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=8, markerfacecolor=CONFIG_COLORS[cfg], markeredgecolor="black", label=CONFIG_DISPLAY[cfg])
        for cfg in CONFIG_ORDER
    ]
    mean_handle = Line2D([0], [0], marker="D", linestyle="None", markersize=7, markerfacecolor="#888888", markeredgecolor="black", label="Run mean")
    fig.suptitle("Reserved VRAM by Hardware Configuration and Policy", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=legend_handles + [mean_handle], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="max_memory_reserved_by_configuration",
        run_ids=run_ids,
        data_sources={"root": "results/monitoring_val/llama_scaling", "views": ["step_metrics_wide_curated", "step_fact_view", "runs", "run_summary_view"], "filter": "apply_analysis_ok"},
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
