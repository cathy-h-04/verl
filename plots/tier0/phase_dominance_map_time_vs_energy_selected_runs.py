"""Plot 0.2: Phase Dominance Map (time share vs energy share) for selected runs.

Purpose:
- Diagnose phase-level power density by comparing energy_share vs time_share.

Data grain:
- Step-level phase rows from phase_fact_view.

Axes:
- X: time_share
- Y: energy_share

Grouping:
- Phase name (color) x Platform (subplot facet).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/phase_dominance_map_time_vs_energy_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PHASE_ORDER = ["rollout", "training", "rl_policy"]
PHASE_COLORS = {
    "rollout": "#4c78a8",
    "training": "#f58518",
    "rl_policy": "#54a24b",
}

PLATFORM_ORDER = ["2xH200", "4xA100"]


def _phase_bucket(phase_name: str) -> str:
    s = str(phase_name).strip().lower()
    if "rollout" in s:
        return "rollout"
    if "train" in s:
        return "training"
    if "rl_policy" in s or "policy" in s:
        return "rl_policy"
    return "other"


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "unknown"


def main() -> None:
    phase, _ = load_view("phase_fact_view")
    required = ["run_id", "phase_name", "time_share", "energy_share"]
    missing = [c for c in required if c not in phase.columns]
    if missing:
        raise ValueError(f"phase_fact_view missing required columns: {missing}")

    available = set(phase["run_id"].astype(str).unique().tolist())
    missing_runs = sorted(run_id for run_id in RUN_IDS if run_id not in available)
    if missing_runs:
        raise ValueError(f"Missing run IDs in phase_fact_view: {missing_runs}")

    df = phase[phase["run_id"].astype(str).isin(RUN_IDS)].copy()
    if df.empty:
        raise ValueError("No rows found for selected RUN_IDS before filtering.")

    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")
    if df.empty:
        raise ValueError("No rows remain after apply_analysis_ok.")

    df["phase_bucket"] = df["phase_name"].map(_phase_bucket)
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df["time_share"] = pd.to_numeric(df["time_share"], errors="coerce")
    df["energy_share"] = pd.to_numeric(df["energy_share"], errors="coerce")

    df = df[df["phase_bucket"].isin(PHASE_ORDER)].copy()
    df = df[df["platform"].isin(PLATFORM_ORDER)].copy()
    df = df.dropna(subset=["time_share", "energy_share"]).copy()
    df = df[(df["time_share"] >= 0) & (df["energy_share"] >= 0)].copy()

    if df.empty:
        raise ValueError("No valid rows to plot after phase/platform and numeric filtering.")

    summary = (
        df.groupby(["platform", "phase_bucket"], dropna=False)
        .agg(
            n_points=("run_id", "size"),
            mean_time_share=("time_share", "mean"),
            mean_energy_share=("energy_share", "mean"),
        )
        .reset_index()
        .sort_values(["platform", "phase_bucket"])
    )
    summary["mean_power_density_index"] = summary["mean_energy_share"] / summary["mean_time_share"]
    print("platform x phase summary:")
    print(summary.to_string(index=False))

    run_counts = (
        df.groupby("platform", dropna=False)["run_id"].nunique().rename("n_runs").reset_index().sort_values("platform")
    )
    print("runs included by platform:")
    print(run_counts.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, platform in zip(axes, PLATFORM_ORDER):
        sub = df[df["platform"] == platform].copy()
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.0, alpha=0.8, zorder=1)

        for phase_name in PHASE_ORDER:
            s = sub[sub["phase_bucket"] == phase_name]
            if s.empty:
                continue
            ax.scatter(
                s["time_share"],
                s["energy_share"],
                s=24,
                color=PHASE_COLORS[phase_name],
                alpha=0.22,
                edgecolors="none",
                zorder=2,
            )
            mean_x = float(s["time_share"].mean())
            mean_y = float(s["energy_share"].mean())
            ax.scatter(
                [mean_x],
                [mean_y],
                s=170,
                marker="o",
                color=PHASE_COLORS[phase_name],
                edgecolors="black",
                linewidths=0.9,
                zorder=4,
            )
            ax.annotate(
                phase_name,
                (mean_x, mean_y),
                textcoords="offset points",
                xytext=(5, 6),
                fontsize=8,
            )

        n_runs = int(sub["run_id"].nunique())
        n_points = int(len(sub))
        ax.set_title(f"{platform} (n_runs={n_runs}, n_points={n_points})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
        ax.set_xlabel("time_share")

    axes[0].set_ylabel("energy_share")
    fig.suptitle("Plot 0.2: Phase Dominance Map (Time Share vs Energy Share)", y=0.99)

    legend_phase = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor=PHASE_COLORS[p],
            markeredgecolor="black",
            label=p,
        )
        for p in PHASE_ORDER
    ]
    legend_ref = [
        Line2D([0], [0], linestyle="--", color="#666666", label="y = x (equal density)"),
    ]
    leg1 = fig.legend(handles=legend_phase, title="Phase", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.add_artist(leg1)
    fig.legend(handles=legend_ref, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.90))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
