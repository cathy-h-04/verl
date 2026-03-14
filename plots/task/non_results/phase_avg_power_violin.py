"""Task-comparison average GPU power by phase, faceted by dataset and colored by policy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/task/non_results/phase_avg_power_violin.png")
INCLUDE_VALIDATION = False

TARGET_SLURM_TO_DATASET = {
    "llama_rm_gsm8k": "gsm8k",
    "llama_rm_rlhf": "full-hh-rlhf",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
DATASET_ORDER = ("gsm8k", "full-hh-rlhf")
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
DATASET_DISPLAY = {
    "gsm8k": "GSM8K",
    "full-hh-rlhf": "full-hh-rlhf",
}
POLICY_COLORS = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = ["run_id", "phase_name", "avg_power_w", "policy", "model"]
    filter_cols_optional = [
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
    needed = [col for col in required_cols + filter_cols_optional if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            "phase_fact_view is missing required columns "
            f"{missing_required}. Available columns: {list(df.columns)}"
        )
    return df[needed].copy()


def _load_runs_with_slurm_metadata() -> pd.DataFrame:
    df_runs, _ = load_view("runs")
    required = ["run_id", "slurm_job_name", "policy"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(
            "runs is missing required columns "
            f"{missing}. Available columns: {list(df_runs.columns)}"
        )
    return df_runs[required].copy()


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    runs_meta_df = _load_runs_with_slurm_metadata()

    runs_meta_df["policy_norm"] = runs_meta_df["policy"].astype(str).str.lower()
    runs_meta_df["slurm_norm"] = runs_meta_df["slurm_job_name"].astype(str).str.lower()
    runs_meta_df["dataset"] = runs_meta_df["slurm_norm"].map(TARGET_SLURM_TO_DATASET)

    selected_runs = runs_meta_df[
        runs_meta_df["dataset"].notna() & runs_meta_df["policy_norm"].isin(TARGET_POLICIES)
    ].copy()
    if selected_runs.empty:
        raise ValueError("No task runs selected — check slurm_job_name values.")

    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    plot_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    if plot_df.empty:
        raise ValueError(f"No phase_fact_view rows for selected runs: {selected_run_ids}")

    plot_df_before_filter = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    filtering = explain_filtering(plot_df_before_filter, plot_df)
    print(f"filtering={filtering}")

    # attach dataset + policy_norm from run metadata
    plot_df = plot_df.merge(
        selected_runs[["run_id", "dataset", "policy_norm"]],
        on="run_id",
        how="left",
    )

    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[plot_df["phase_bucket"].isin(PHASE_ORDER)].copy()
    plot_df["avg_power_w"] = pd.to_numeric(plot_df["avg_power_w"], errors="coerce")
    plot_df = plot_df.dropna(subset=["avg_power_w"]).copy()

    run_counts = (
        selected_runs.groupby(["dataset", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["dataset", "policy_norm"])
    )
    print("runs included by (dataset, policy):")
    print(run_counts.to_string(index=False))

    point_counts = (
        plot_df.groupby(["dataset", "phase_bucket", "policy_norm"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["dataset", "phase_bucket", "policy_norm"])
    )
    print("points plotted by (dataset, phase_bucket, policy):")
    print(point_counts.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.8), sharey=True)
    facet_axes = dict(zip(DATASET_ORDER, axes))

    for dataset in DATASET_ORDER:
        ax = facet_axes[dataset]
        facet_df = plot_df[plot_df["dataset"] == dataset].copy()

        data = []
        positions = []
        color_by_violin = []
        for phase_i, phase in enumerate(PHASE_ORDER, start=1):
            for policy_i, policy in enumerate(POLICY_ORDER):
                vals = facet_df.loc[
                    (facet_df["phase_bucket"] == phase) & (facet_df["policy_norm"] == policy),
                    "avg_power_w",
                ].tolist()
                if not vals:
                    continue
                offset = (policy_i - 1) * 0.24
                data.append(vals)
                positions.append(phase_i + offset)
                color_by_violin.append(POLICY_COLORS[policy])

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.22,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(vp["bodies"], color_by_violin):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_linewidth(0.7)
            body.set_alpha(0.45)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.085,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.2},
            flierprops={"marker": ".", "markersize": 2.0, "alpha": 0.25, "markerfacecolor": "black", "markeredgecolor": "black"},
        )

        ax.set_xticks(range(1, len(PHASE_ORDER) + 1))
        ax.set_xticklabels([PHASE_DISPLAY[phase] for phase in PHASE_ORDER])
        ax.set_xlabel("Phase", fontweight="bold")
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Average GPU power (W)", fontweight="bold")
    fig.suptitle(
        "GPU Power by Phase — GSM8K vs full-hh-rlhf (Llama-3.1-8B, fixed RM)",
        y=0.99,
        fontweight="bold",
    )
    policy_handles = [
        Patch(facecolor=POLICY_COLORS[p], edgecolor="black", alpha=0.45, label=POLICY_DISPLAY[p]) for p in POLICY_ORDER
    ]
    fig.legend(handles=policy_handles, title="Policy", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
