"""Decoding efficiency versus response length for llama scaling runs."""

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


OUTPATH = Path("plots/out/scale/decode_efficiency_vs_response_length.png")
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
CONFIG_MARKERS = {
    "2xA100": "*",
    "2xH200": "o",
    "4xA100": "s",
    "4xH200": "^",
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

    step_fact, _ = load_view("step_fact_view")
    filters = step_fact[step_fact["run_id"].astype(str).isin(run_ids)][
        [c for c in ["run_id", "global_step_canonical", "boundary_integrity_ok", "join_integrity_ok", "is_warmup_idle", "is_validation_step", "is_incomplete_phase", "is_outlier_sample"] if c in step_fact.columns]
    ].copy()
    before = filters.copy()
    filters = apply_analysis_ok(filters)
    print(f"filtering={explain_filtering(before, filters)}")
    keys = filters[["run_id", "global_step_canonical"]].drop_duplicates()

    long_df, _ = load_view("step_metrics_long")
    metric_df = long_df[long_df["run_id"].astype(str).isin(run_ids)][["run_id", "global_step_canonical", "metric_key", "metric_value_float"]].copy()
    metric_df["global_step_canonical"] = pd.to_numeric(metric_df["global_step_canonical"], errors="coerce")
    metric_df["metric_value_float"] = pd.to_numeric(metric_df["metric_value_float"], errors="coerce")
    metric_df = metric_df.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    metric_df["global_step_canonical"] = metric_df["global_step_canonical"].astype(int)
    metric_df = metric_df[metric_df["metric_key"].isin(["timing_per_token_ms/gen", "response_length/mean"])].copy()
    metric_df = metric_df.merge(keys, on=["run_id", "global_step_canonical"], how="inner")
    wide = (
        metric_df.pivot_table(
            index=["run_id", "global_step_canonical"],
            columns="metric_key",
            values="metric_value_float",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide = wide.rename(columns={"timing_per_token_ms/gen": "timing_per_token_ms_gen", "response_length/mean": "response_length_mean"})
    wide = wide.dropna(subset=["timing_per_token_ms_gen", "response_length_mean"]).copy()
    wide = wide.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    run_means = (
        wide.groupby(["run_id", "policy_norm", "config"], dropna=False)[["timing_per_token_ms_gen", "response_length_mean"]]
        .mean()
        .reset_index()
        .sort_values(["policy_norm", "config", "run_id"])
    )
    print("run-level decode efficiency summary:")
    print(run_means.to_string(index=False))

    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(14.8, 5.4), sharey=True, sharex=True)
    for ax, policy in zip(axes, POLICY_ORDER):
        panel = wide[wide["policy_norm"] == policy].copy()
        if panel.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        for config in CONFIG_ORDER:
            sub = panel[panel["config"] == config].copy()
            if not sub.empty:
                ax.scatter(
                    sub["response_length_mean"],
                    sub["timing_per_token_ms_gen"],
                    s=24,
                    alpha=0.32,
                    color=CONFIG_COLORS[config],
                    edgecolors="none",
                    zorder=1,
                )
                if len(sub) >= 2:
                    x = sub["response_length_mean"].to_numpy(dtype=float)
                    y = sub["timing_per_token_ms_gen"].to_numpy(dtype=float)
                    order = np.argsort(x)
                    coef = np.polyfit(x, y, deg=1)
                    x_line = np.linspace(float(x.min()), float(x.max()), 100)
                    y_line = coef[0] * x_line + coef[1]
                    ax.plot(
                        x_line,
                        y_line,
                        color=CONFIG_COLORS[config],
                        linewidth=2.0,
                        alpha=0.95,
                        zorder=2,
                    )
        ax.set_title(POLICY_DISPLAY[policy], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("response_length/mean", fontsize=AXIS_LABEL_SIZE)
        ax.grid(axis="both", alpha=0.24, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    axes[0].set_ylabel("timing_per_token_ms/gen", fontsize=AXIS_LABEL_SIZE)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor=CONFIG_COLORS[cfg], markeredgecolor="none", label=CONFIG_DISPLAY[cfg])
        for cfg in CONFIG_ORDER
    ]
    fig.suptitle("Decoding Efficiency vs Response Length Across Scaling Configurations", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="decode_efficiency_vs_response_length",
        run_ids=run_ids,
        data_sources={"root": "results/monitoring_val/llama_scaling", "views": ["step_metrics_long", "step_fact_view", "runs", "run_summary_view"], "filter": "apply_analysis_ok"},
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
