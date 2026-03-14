"""Rollout memory-bandwidth utilization by policy and dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/task/rollout_mem_bandwidth.png")
TARGET_DATASETS = ("gsm8k", "rlhf-ff")
POLICY_ORDER = ("ppo", "remax", "grpo")
DATASET_DISPLAY = {
    "gsm8k": "gsm8k",
    "rlhf-ff": "full-hh-rlhf",
}
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
DATASET_COLORS = {
    "gsm8k": "#295894",
    "rlhf-ff": "#D04A1C",
}
JITTER_ALPHA = 0.35
JITTER_SIZE = 3.5
BAR_WIDTH = 0.32


def _load_plot_df() -> pd.DataFrame:
    runs, _ = load_view("run_summary_view")
    periodic, _ = load_view("hardware_periodic")

    required_runs = ["run_id", "dataset", "policy"]
    missing_runs = [col for col in required_runs if col not in runs.columns]
    if missing_runs:
        raise ValueError(f"run_summary_view missing required columns: {missing_runs}")

    required_periodic = ["run_id", "phase_name", "sm_util_pct", "mem_util_pct", "record_type", "source"]
    missing_periodic = [col for col in required_periodic if col not in periodic.columns]
    if missing_periodic:
        raise ValueError(f"hardware_periodic missing required columns: {missing_periodic}")

    runs = runs[required_runs].drop_duplicates().copy()
    runs["dataset"] = runs["dataset"].astype(str).str.lower()
    runs["policy"] = runs["policy"].astype(str).str.lower()
    runs = runs[runs["dataset"].isin(TARGET_DATASETS) & runs["policy"].isin(POLICY_ORDER)].copy()
    if runs.empty:
        raise ValueError("No gsm8k / rlhf-ff PPO/ReMax/GRPO runs found in run_summary_view.")

    df = periodic[periodic["run_id"].astype(str).isin(runs["run_id"].astype(str))][required_periodic].copy()
    df = df.merge(runs, on="run_id", how="inner", validate="many_to_one")
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[
        (df["record_type"].astype(str).str.upper() == "PERIODIC")
        & (df["source"].astype(str).str.lower() == "nvml")
        & (df["phase_name"] == "rollout")
    ].copy()
    df["sm_util_pct"] = pd.to_numeric(df["sm_util_pct"], errors="coerce")
    df["mem_util_pct"] = pd.to_numeric(df["mem_util_pct"], errors="coerce")
    df = df.dropna(subset=["sm_util_pct", "mem_util_pct"]).copy()
    df = df[(df["sm_util_pct"] >= 0.0) & (df["mem_util_pct"] >= 0.0)].copy()
    if df.empty:
        raise ValueError("No rollout-phase periodic SM/memory utilization samples available for plotting.")
    return df


def _draw_box_row(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    positions = np.arange(len(POLICY_ORDER), dtype=float)
    offset = BAR_WIDTH / 2

    for dataset_i, dataset in enumerate(TARGET_DATASETS):
        pos = positions + (-offset if dataset_i == 0 else offset)
        data = []
        valid_pos = []
        valid_policies = []
        for i, policy in enumerate(POLICY_ORDER):
            vals = df[(df["policy"] == policy) & (df["dataset"] == dataset)][metric].to_list()
            if vals:
                data.append(vals)
                valid_pos.append(pos[i])
                valid_policies.append(policy)
        if not data:
            continue
        bp = ax.boxplot(
            data,
            positions=valid_pos,
            widths=BAR_WIDTH,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            boxprops={"edgecolor": "black", "linewidth": 0.7},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(DATASET_COLORS[dataset])
            patch.set_alpha(1.0)

        rng = np.random.default_rng(42 + dataset_i)
        for xpos, policy in zip(valid_pos, valid_policies):
            pts = df[(df["policy"] == policy) & (df["dataset"] == dataset)][metric].to_numpy()
            if pts.size == 0:
                continue
            jx = xpos + rng.uniform(-BAR_WIDTH * 0.4, BAR_WIDTH * 0.4, size=len(pts))
            ax.scatter(jx, pts, s=JITTER_SIZE, color=DATASET_COLORS[dataset], alpha=JITTER_ALPHA, zorder=4, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels([POLICY_DISPLAY[p] for p in POLICY_ORDER])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(POLICY_ORDER) - 0.4)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)


def main() -> None:
    df = _load_plot_df()

    summary = (
        df.groupby(["policy", "dataset"], dropna=False)
        .agg(
            n_samples=("run_id", "size"),
            sm_util_median=("sm_util_pct", "median"),
            sm_util_q25=("sm_util_pct", lambda s: s.quantile(0.25)),
            sm_util_q75=("sm_util_pct", lambda s: s.quantile(0.75)),
            mem_util_median=("mem_util_pct", "median"),
            mem_util_q25=("mem_util_pct", lambda s: s.quantile(0.25)),
            mem_util_q75=("mem_util_pct", lambda s: s.quantile(0.75)),
        )
        .reset_index()
        .sort_values(["policy", "dataset"])
    )
    print("rollout periodic sample summary by policy and dataset:")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(1, 1, figsize=(13.5, 5.0))
    _draw_box_row(ax, df, "mem_util_pct", "Memory Bandwidth Utilization")
    ax.set_xlabel("Policy")
    ax.set_ylim(-3.0, 103.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    for i, policy in enumerate(POLICY_ORDER):
        gsm8k_vals = df[(df["policy"] == policy) & (df["dataset"] == "gsm8k")]["mem_util_pct"]
        rlhf_vals = df[(df["policy"] == policy) & (df["dataset"] == "rlhf-ff")]["mem_util_pct"]
        if gsm8k_vals.empty or rlhf_vals.empty:
            continue
        gsm8k_median = float(gsm8k_vals.median())
        rlhf_median = float(rlhf_vals.median())
        if gsm8k_median <= 0:
            if rlhf_median <= 0:
                continue
            label = "n/a"
        else:
            multiple = rlhf_median / gsm8k_median
            label = f"{multiple:.1f}x"
        rlhf_q75 = float(rlhf_vals.quantile(0.75))
        rlhf_q25 = float(rlhf_vals.quantile(0.25))
        y_pos = rlhf_q75 - max(4.0, 0.18 * max(rlhf_q75 - rlhf_q25, 1.0))
        txt = ax.text(
            float(i) + (BAR_WIDTH / 2),
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=5,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.6, foreground="black")])

    legend_handles = [
        Patch(facecolor=DATASET_COLORS[dataset], edgecolor="black", label=DATASET_DISPLAY[dataset])
        for dataset in TARGET_DATASETS
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=9,
    )
    fig.suptitle(
        "Rollout Memory Bandwidth Utilization by Policy and Dataset",
        y=1.02,
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
