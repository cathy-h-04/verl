"""Temperature and power distributions for SW power-cap throttled GPU samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/throttle_power_temp_diagnostics.png")

CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
PHASE_COLORS = {
    "rollout": "#1D4E89",
    "rl_policy": "#7A7A7A",
    "training": "#C73E1D",
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
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    if "is_checkpoint_continuation" in runs_df.columns:
        flags = runs_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    selected = selected[selected["config"].isin(CONFIG_ORDER)].drop_duplicates()
    if selected.empty:
        raise ValueError("No llama_scaling runs selected.")
    return selected


def _style_boxplot(bp: dict, facecolor: str) -> None:
    for box in bp["boxes"]:
        box.set(facecolor=facecolor, edgecolor="black", linewidth=0.85, alpha=0.88)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=0.8)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=0.8)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.1)
    for flier in bp["fliers"]:
        flier.set(marker="o", markersize=1.8, markerfacecolor=facecolor, markeredgecolor="none", alpha=0.18)


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    view, _ = load_view("device_timeseries_view")
    cols = ["run_id", "device_kind", "phase_name", "power_w", "temp_gpu_c", "thr_sw_power_cap"]
    df = view[cols].copy()
    df = df[df["run_id"].astype(str).isin(run_ids)].copy()
    df = df[df["device_kind"].astype(str).str.lower() == "gpu"].copy()
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(PHASE_ORDER)].copy()
    df["thr_sw_power_cap"] = df["thr_sw_power_cap"].fillna(False).astype(bool)
    df["power_w"] = pd.to_numeric(df["power_w"], errors="coerce")
    df["temp_gpu_c"] = pd.to_numeric(df["temp_gpu_c"], errors="coerce")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    df = df[df["thr_sw_power_cap"]].dropna(subset=["power_w", "temp_gpu_c"]).copy()

    summary = (
        df.groupby(["config", "phase_name"], as_index=False)
        .agg(
            n=("thr_sw_power_cap", "size"),
            temp_mean=("temp_gpu_c", "mean"),
            temp_p95=("temp_gpu_c", lambda s: float(s.quantile(0.95))),
            temp_max=("temp_gpu_c", "max"),
            power_mean=("power_w", "mean"),
            power_p95=("power_w", lambda s: float(s.quantile(0.95))),
            power_max=("power_w", "max"),
        )
        .sort_values(["config", "phase_name"])
    )
    print("throttled-sample power/temperature summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), sharex=True)
    x = np.arange(len(CONFIG_ORDER), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(PHASE_ORDER))
    metrics = [
        ("temp_gpu_c", "GPU temperature (C)", "Temperature During Power-Cap Throttle"),
        ("power_w", "GPU power (W)", "Power During Power-Cap Throttle"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for idx, phase_name in enumerate(PHASE_ORDER):
            positions = x + offsets[idx]
            data = []
            plot_positions = []
            for pos, config in zip(positions, CONFIG_ORDER):
                vals = df[(df["config"] == config) & (df["phase_name"] == phase_name)][metric].dropna().to_numpy()
                if vals.size == 0:
                    continue
                data.append(vals)
                plot_positions.append(pos)
            if not data:
                continue
            bp = ax.boxplot(
                data,
                positions=plot_positions,
                widths=0.18,
                patch_artist=True,
                manage_ticks=False,
                showfliers=True,
            )
            _style_boxplot(bp, PHASE_COLORS[phase_name])

        ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.set_xticks(x, [CONFIG_DISPLAY[cfg] for cfg in CONFIG_ORDER])
        ax.grid(axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    axes[0].axhline(80.0, color="black", linestyle="--", linewidth=1.0, alpha=0.85)
    axes[0].text(
        0.98,
        0.96,
        "80C thermal caution line",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    handles = [Patch(facecolor=PHASE_COLORS[phase], edgecolor="black", label=PHASE_DISPLAY[phase]) for phase in PHASE_ORDER]
    fig.suptitle("Power-Cap Throttle Diagnostics", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
