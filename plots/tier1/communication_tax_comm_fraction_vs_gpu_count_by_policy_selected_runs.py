"""Plot 1.1 (segmented): Communication tax by policy for selected runs.

Metric:
- comm_fraction/step = comm_s_all_ops / step_s
  where comm_s_all_ops is computed from step_metrics_long as:
    - sum of all comm_s/* keys
    - excluding aggregate keys comm_s/step and comm_s/total

Segmentation:
- One panel per policy (PPO/ReMax/GRPO)
- Within each panel: platform comparison (2xH200 vs 4xA100)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier1/communication_tax_comm_fraction_vs_gpu_count_by_policy_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PLATFORM_ORDER = ["2xH200", "4xA100"]
PLATFORM_COLORS = {"2xH200": "#4c78a8", "4xA100": "#f58518"}
POLICY_ORDER = ["ppo", "remax", "grpo"]


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def _norm_policy(x: str) -> str:
    return str(x).strip().lower().replace("remx", "remax")


def main() -> None:
    step_fact, _ = load_view("step_fact_view")
    wide, _ = load_view("step_metrics_wide_curated")
    step_long, _ = load_view("step_metrics_long")

    required_wide = ["run_id", "global_step_canonical", "metric_timing_s_step"]
    missing_wide = [c for c in required_wide if c not in wide.columns]
    if missing_wide:
        raise ValueError(f"step_metrics_wide_curated missing required columns: {missing_wide}")
    required_long = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    missing_long = [c for c in required_long if c not in step_long.columns]
    if missing_long:
        raise ValueError(f"step_metrics_long missing required columns: {missing_long}")

    sf = step_fact[step_fact["run_id"].astype(str).isin(RUN_IDS)].copy()
    if sf.empty:
        raise ValueError("No step_fact_view rows for selected RUN_IDS.")

    sf_before = sf.copy()
    sf = apply_analysis_ok(sf)
    print(f"filtering={explain_filtering(sf_before, sf)}")
    if sf.empty:
        raise ValueError("No rows remain after apply_analysis_ok.")

    keys = sf[["run_id", "global_step_canonical", "policy", "step_time_s"]].drop_duplicates().copy()
    keys["step_time_s"] = pd.to_numeric(keys["step_time_s"], errors="coerce")
    keys["policy_norm"] = keys["policy"].map(_norm_policy)

    w = wide[wide["run_id"].astype(str).isin(RUN_IDS)][required_wide].copy()
    w["metric_timing_s_step"] = pd.to_numeric(w["metric_timing_s_step"], errors="coerce")
    w = w.dropna(subset=["global_step_canonical"]).copy()

    m = step_long[step_long["run_id"].astype(str).isin(RUN_IDS)][required_long].copy()
    m = m[m["metric_key"].astype(str).str.startswith("comm_s/")].copy()
    m = m[~m["metric_key"].isin(["comm_s/step", "comm_s/total"])].copy()
    m["metric_value_float"] = pd.to_numeric(m["metric_value_float"], errors="coerce")
    m = m.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    comm_by_step = (
        m.groupby(["run_id", "global_step_canonical"], dropna=False)["metric_value_float"]
        .sum(min_count=1)
        .reset_index(name="comm_s_step_all_ops")
    )

    df = keys.merge(w, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(comm_by_step, on=["run_id", "global_step_canonical"], how="inner")
    df["step_s_for_fraction"] = df["metric_timing_s_step"].where(df["metric_timing_s_step"] > 0, df["step_time_s"])
    df = df[df["step_s_for_fraction"] > 0].copy()
    df["comm_fraction_step"] = df["comm_s_step_all_ops"] / df["step_s_for_fraction"]
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df = df[df["platform"].isin(PLATFORM_ORDER)].copy()
    df = df[df["policy_norm"].isin(POLICY_ORDER)].copy()
    df = df.dropna(subset=["comm_fraction_step"]).copy()

    if df.empty:
        raise ValueError("No valid comm_fraction rows after joins/filters.")

    summary = (
        df.groupby(["policy_norm", "platform"], dropna=False)["comm_fraction_step"]
        .agg(n_steps="size", mean="mean", median="median", p90=lambda s: s.quantile(0.90))
        .reset_index()
        .sort_values(["policy_norm", "platform"])
    )
    run_means = (
        df.groupby(["policy_norm", "platform", "run_id"], dropna=False)["comm_fraction_step"]
        .mean()
        .reset_index(name="run_mean_comm_fraction")
        .sort_values(["policy_norm", "platform", "run_id"])
    )

    print("policy x platform summary:")
    print(summary.to_string(index=False))
    print("policy x platform x run means:")
    print(run_means.to_string(index=False))

    rng = np.random.default_rng(17)
    global_top = float(df["comm_fraction_step"].quantile(0.995))
    y_max = global_top * 1.18 if global_top > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.4), sharey=True)
    x_map = {p: i for i, p in enumerate(PLATFORM_ORDER)}

    for ax, policy in zip(axes, POLICY_ORDER):
        sub = df[df["policy_norm"] == policy].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        box_data = []
        box_pos = []
        for platform in PLATFORM_ORDER:
            s = sub[sub["platform"] == platform].copy()
            if s.empty:
                continue
            x0 = x_map[platform]
            jitter = rng.uniform(-0.16, 0.16, size=len(s))
            ax.scatter(
                x0 + jitter,
                s["comm_fraction_step"],
                s=12,
                alpha=0.18,
                color=PLATFORM_COLORS[platform],
                edgecolors="none",
                zorder=1,
            )
            box_data.append(s["comm_fraction_step"].to_numpy())
            box_pos.append(x0)

        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=box_pos,
                widths=0.35,
                showfliers=False,
                patch_artist=True,
                boxprops={"edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8},
            )
            for patch, pos in zip(bp["boxes"], box_pos):
                patch.set_facecolor(PLATFORM_COLORS[PLATFORM_ORDER[pos]])
                patch.set_alpha(0.35)

        rm = run_means[run_means["policy_norm"] == policy].copy()
        for platform in PLATFORM_ORDER:
            s = rm[rm["platform"] == platform].copy()
            if s.empty:
                continue
            x0 = x_map[platform]
            jitter = np.linspace(-0.08, 0.08, num=len(s)) if len(s) > 1 else np.array([0.0])
            ax.scatter(
                x0 + jitter,
                s["run_mean_comm_fraction"],
                s=70,
                marker="D",
                color=PLATFORM_COLORS[platform],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                zorder=3,
            )

        stat = summary[summary["policy_norm"] == policy].set_index("platform")
        for platform in PLATFORM_ORDER:
            if platform in stat.index:
                ax.text(
                    x_map[platform],
                    y_max * 0.92,
                    f"mean={float(stat.loc[platform, 'mean']):.4f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
                )

        if all(p in stat.index for p in PLATFORM_ORDER):
            mean_2 = float(stat.loc["2xH200", "mean"])
            mean_4 = float(stat.loc["4xA100", "mean"])
            if mean_2 > 0:
                tax_pct = 100.0 * (mean_4 - mean_2) / mean_2
                ax.text(
                    0.5,
                    0.985,
                    f"tax={tax_pct:+.1f}%",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    bbox={"facecolor": "#f7f7f7", "edgecolor": "#666666", "boxstyle": "round,pad=0.2", "alpha": 0.9},
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(PLATFORM_ORDER)
        ax.set_title(policy.upper(), pad=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Platform")

    axes[0].set_ylabel("comm_fraction/step (all comm_s/* / step_s)")
    fig.suptitle("Plot 1.1: Communication Tax by Policy (All Communication Ops)", y=0.985)

    legend_platform = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=8, markerfacecolor=PLATFORM_COLORS[p], markeredgecolor="black", label=p)
        for p in PLATFORM_ORDER
    ]
    legend_points = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor="#888888", markeredgecolor="none", alpha=0.35, label="Step-level points"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=6, markerfacecolor="#888888", markeredgecolor="black", label="Run mean"),
    ]
    leg1 = fig.legend(handles=legend_platform, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.33, 0.93), title="Platform")
    fig.add_artist(leg1)
    fig.legend(handles=legend_points, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.73, 0.93))

    fig.tight_layout(rect=(0, 0, 1, 0.87))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
