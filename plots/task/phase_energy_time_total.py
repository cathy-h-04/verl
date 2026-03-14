"""Absolute phase energy/time totals in bar-chart form for task comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


INCLUDE_VALIDATION = False
OUTPATH = Path("plots/out/task/phase_energy_time_total.png")
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
TARGET_DATASETS = ("gsm8k", "rlhf-ff")
DATASET_DISPLAY = {
    "gsm8k": "gsm8k",
    "rlhf-ff": "full-hh-rlhf",
}
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "rollout",
    "rl_policy": "preparation",
    "training": "training",
}
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
PHASE_COLORS = {
    "rollout": "#4C78A8",
    "training": "#F58518",
    "rl_policy": "#54A24B",
}
ENERGY_SCALE = 1000.0
TIME_SCALE = 60.0


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = ["run_id", "phase_name", "phase_time_s", "total_energy_j", "policy"]
    optional_cols = [
        "global_step_canonical",
        "global_step",
        "analysis_ok",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    df, _ = load_view("phase_fact_view")
    needed = [col for col in required_cols + optional_cols if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"phase_fact_view is missing required columns {missing_required}")
    return df[needed].copy()


def _select_runs() -> pd.DataFrame:
    runs_df, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "dataset"]
    missing = [col for col in required if col not in runs_df.columns]
    if missing:
        raise ValueError(f"run_summary_view is missing required selection columns {missing}")
    runs_df = runs_df.copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["dataset_group"] = runs_df["dataset"].astype(str).str.lower()
    selected_runs = runs_df[
        runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["dataset_group"].isin(TARGET_DATASETS)
    ].copy()
    checkpoint_mask = (
        ~selected_runs["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in selected_runs.columns
        else True
    )
    selected_runs = selected_runs[checkpoint_mask].copy()
    if selected_runs.empty:
        raise ValueError("No task-comparison runs selected.")
    return selected_runs[["run_id", "policy_norm", "dataset_group"]].drop_duplicates()


def _format_energy_kj(value_kj: float) -> str:
    return f"{value_kj:.1f}"


def _format_time_min(value_min: float) -> str:
    return f"{value_min:.1f}"


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    selected_runs = _select_runs()

    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    plot_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    if plot_df.empty:
        raise ValueError(f"Selected run_ids produced no rows in phase_fact_view: {selected_run_ids}")

    plot_df_before_filter = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    filtering = explain_filtering(plot_df_before_filter, plot_df)
    print(f"filtering={filtering}")

    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    plot_df = plot_df.merge(selected_runs, on="run_id", how="left", validate="many_to_one")
    plot_df["policy_norm"] = plot_df["policy"].astype(str).str.lower()
    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[~plot_df["phase_bucket"].isin(["other", "validation"])].copy()

    run_counts = (
        selected_runs.groupby(["dataset_group", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["dataset_group", "policy_norm"])
    )
    print("runs included by (dataset, policy):")
    print(run_counts.to_string(index=False))

    totals = (
        plot_df.groupby(["dataset_group", "policy_norm", "phase_bucket"], dropna=False)[["total_energy_j", "phase_time_s"]]
        .sum()
        .reset_index()
    )
    print("phase totals by (dataset, policy, phase):")
    print(totals.sort_values(["dataset_group", "policy_norm", "phase_bucket"]).to_string(index=False))

    fig, axes = plt.subplots(len(TARGET_DATASETS), len(POLICY_ORDER), figsize=(16, 9))
    x = np.arange(len(PHASE_ORDER), dtype=float)
    width = 0.28
    global_energy_max_kj = max(float(totals["total_energy_j"].max() / ENERGY_SCALE), 1.0)
    global_time_max_min = max(float(totals["phase_time_s"].max() / TIME_SCALE), 1.0)

    for row_idx, dataset in enumerate(TARGET_DATASETS):
        facet_df = totals[totals["dataset_group"] == dataset]
        for col_idx, policy in enumerate(POLICY_ORDER):
            ax = axes[row_idx][col_idx]
            ax_time = ax.twinx()
            combo_df = (
                facet_df[facet_df["policy_norm"] == policy]
                .set_index("phase_bucket")
                .reindex(PHASE_ORDER)
                .fillna(0.0)
            )
            energy_vals_kj = combo_df["total_energy_j"].to_numpy(dtype=float) / ENERGY_SCALE
            time_vals_min = combo_df["phase_time_s"].to_numpy(dtype=float) / TIME_SCALE
            colors = [PHASE_COLORS[phase] for phase in PHASE_ORDER]

            energy_bars = ax.bar(x - width / 2, energy_vals_kj, width=width, color=colors, edgecolor="black", linewidth=0.8)
            time_bars = ax_time.bar(x + width / 2, time_vals_min, width=width, color=colors, edgecolor="black", linewidth=0.8, alpha=0.45)

            ax.set_title(f"{DATASET_DISPLAY[dataset]} | {POLICY_DISPLAY[policy]}", fontsize=10, fontweight="bold")
            ax.set_xticks(x, [PHASE_DISPLAY[phase] for phase in PHASE_ORDER], rotation=0)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_ylabel(f"{DATASET_DISPLAY[dataset]}\nenergy (kJ)")
            if col_idx == len(POLICY_ORDER) - 1:
                ax_time.set_ylabel("time (min)")
            ax.set_ylim(0, global_energy_max_kj * 1.12)
            ax_time.set_ylim(0, global_time_max_min * 1.12)

            for bar, val in zip(energy_bars, energy_vals_kj):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    _format_energy_kj(float(val)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
            for bar, val in zip(time_bars, time_vals_min):
                ax_time.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    _format_time_min(float(val)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

    phase_handles = [Patch(facecolor=PHASE_COLORS[phase], edgecolor="black", label=PHASE_DISPLAY[phase]) for phase in PHASE_ORDER]
    metric_handles = [
        Patch(facecolor="#888888", edgecolor="black", label="energy", alpha=1.0),
        Patch(facecolor="#888888", edgecolor="black", label="time", alpha=0.45),
    ]
    fig.suptitle("Total Phase Energy and Time by Policy and Dataset", y=0.995, fontweight="bold")
    fig.legend(handles=phase_handles, title="phase colors", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.36, 0.92))
    fig.legend(handles=metric_handles, title="metric type", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.79, 0.92))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
