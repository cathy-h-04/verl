"""Phase average GPU power by policy for task comparison.

Creates 3 policy panels. Within each panel, each phase has side-by-side
dataset bars (gsm8k vs full-hh-rlhf) with light/dark styling.

avg_power_w = total_energy_j / phase_time_s
for each (dataset, policy, phase) cell after analysis filtering.
"""

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
OUTPATH = Path("plots/out/task/phase_avg_power_total.png")
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
DATASET_COLORS = {
    "gsm8k": "#295894",
    "rlhf-ff": "#D04A1C",
}


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
    plot_df["total_energy_j"] = pd.to_numeric(plot_df["total_energy_j"], errors="coerce")
    plot_df["phase_time_s"] = pd.to_numeric(plot_df["phase_time_s"], errors="coerce")
    plot_df = plot_df.dropna(subset=["total_energy_j", "phase_time_s"]).copy()
    plot_df = plot_df[plot_df["phase_time_s"] > 0].copy()

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
    totals["avg_power_w"] = totals["total_energy_j"] / totals["phase_time_s"]
    print("phase avg power by (dataset, policy, phase):")
    print(totals.sort_values(["dataset_group", "policy_norm", "phase_bucket"]).to_string(index=False))

    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(14.5, 5.6), sharey=True)
    x = np.arange(len(PHASE_ORDER), dtype=float)
    bar_width = 0.34

    global_power_max = max(float(totals["avg_power_w"].max()), 1.0)

    for col_idx, policy in enumerate(POLICY_ORDER):
        ax = axes[col_idx]
        combo_df = (
            totals[totals["policy_norm"] == policy]
            .set_index(["dataset_group", "phase_bucket"])
            .sort_index()
        )

        gsm_vals = []
        rlhf_vals = []
        for phase in PHASE_ORDER:
            gsm_vals.append(float(combo_df.loc[("gsm8k", phase), "avg_power_w"]) if ("gsm8k", phase) in combo_df.index else 0.0)
            rlhf_vals.append(float(combo_df.loc[("rlhf-ff", phase), "avg_power_w"]) if ("rlhf-ff", phase) in combo_df.index else 0.0)

        bars_gsm = ax.bar(
            x - bar_width / 2,
            gsm_vals,
            width=bar_width,
            color=DATASET_COLORS["gsm8k"],
            edgecolor="black",
            linewidth=0.8,
            label=DATASET_DISPLAY["gsm8k"],
        )
        bars_rlhf = ax.bar(
            x + bar_width / 2,
            rlhf_vals,
            width=bar_width,
            color=DATASET_COLORS["rlhf-ff"],
            edgecolor="black",
            linewidth=0.8,
            label=DATASET_DISPLAY["rlhf-ff"],
        )

        ax.set_title(POLICY_DISPLAY[policy], fontsize=11, fontweight="bold")
        ax.set_xticks(x, [PHASE_DISPLAY[phase] for phase in PHASE_ORDER], rotation=0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, global_power_max * 1.12)

        if col_idx == 0:
            ax.set_ylabel("average power (W)")

        for bars in (bars_gsm, bars_rlhf):
            for bar in bars:
                val = float(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + global_power_max * 0.012,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                    fontweight="bold",
                )

    dataset_handles = [
        Patch(facecolor=DATASET_COLORS["gsm8k"], edgecolor="black", label=DATASET_DISPLAY["gsm8k"]),
        Patch(facecolor=DATASET_COLORS["rlhf-ff"], edgecolor="black", label=DATASET_DISPLAY["rlhf-ff"]),
    ]
    fig.suptitle("Phase Average GPU Power by Policy", y=0.995, fontweight="bold")
    fig.legend(handles=dataset_handles, title="dataset", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
