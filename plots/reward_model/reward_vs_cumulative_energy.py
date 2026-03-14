"""Validation reward versus cumulative energy for reward-mechanism runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/reward_model/reward_vs_cumulative_energy.png")
REWARD_KEY = "val-core/openai/gsm8k/reward/mean@1"
ENERGY_TO_GJ = 1e9

TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
POLICY_COLORS = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}
TARGET_EXPERIMENT_FACETS = ("Llama Reward Function", "Llama Reward Model")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama Reward Function": "llama_new_baseline",
    "Llama Reward Model": "llama_rm_gsm8k",
}
LOGICAL_GROUP_PREFIXES_BY_FACET = {
    "Llama Reward Function": ("stage1_llama8b_",),
    "Llama Reward Model": ("llama8b_",),
}
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
BEST_POINT_COLOR = "#D62728"


def _experiment_facet(slurm_job_name: str, logical_run_group: str) -> str:
    slurm_text = str(slurm_job_name).strip().lower()
    logical_text = str(logical_run_group).strip().lower()
    for facet in TARGET_EXPERIMENT_FACETS:
        expected_slurm = TARGET_SLURM_JOB_NAME_BY_FACET[facet]
        logical_prefixes = LOGICAL_GROUP_PREFIXES_BY_FACET[facet]
        if slurm_text == expected_slurm and logical_text.startswith(logical_prefixes):
            return facet
    return "Other"


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")
    required = ["run_id", "policy", "logical_run_group"]
    missing = [col for col in required if col not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    ).copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()
    runs_df["experiment_facet"] = [
        _experiment_facet(slurm_job_name=slurm_job_name, logical_run_group=logical_run_group_value)
        for slurm_job_name, logical_run_group_value in zip(runs_df["slurm_job_name"], runs_df["logical_run_group"])
    ]

    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["experiment_facet"].isin(TARGET_EXPERIMENT_FACETS)
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )
    integrity_mask = (
        (pd.to_numeric(runs_df["join_coverage_rate"], errors="coerce") == 1.0)
        & (pd.to_numeric(runs_df["phase_boundary_integrity_rate"], errors="coerce") == 1.0)
        if {"join_coverage_rate", "phase_boundary_integrity_rate"}.issubset(runs_df.columns)
        else True
    )

    selected = runs_df[non_rollout_knob_mask & target_mask & checkpoint_mask & integrity_mask][
        ["run_id", "policy_norm", "experiment_facet"]
    ].drop_duplicates()
    if selected.empty:
        raise ValueError("No reward-mechanism runs selected.")
    return selected


def main() -> None:
    selected_runs = _select_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    required_step = ["run_id", "global_step_canonical", "step_total_energy_j"]
    missing_step = [c for c in required_step if c not in step_fact.columns]
    if missing_step:
        raise ValueError(f"step_fact_view missing required columns: {missing_step}")

    steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)][required_step].copy()
    steps["global_step_canonical"] = pd.to_numeric(steps["global_step_canonical"], errors="coerce")
    steps["step_total_energy_j"] = pd.to_numeric(steps["step_total_energy_j"], errors="coerce")
    steps = steps.dropna(subset=["global_step_canonical", "step_total_energy_j"]).copy()
    steps["global_step_canonical"] = steps["global_step_canonical"].astype(int)
    steps = steps.sort_values(["run_id", "global_step_canonical"])
    steps["cumulative_energy_gj"] = steps.groupby("run_id")["step_total_energy_j"].cumsum() / ENERGY_TO_GJ

    step_metrics_long, _ = load_view("step_metrics_long")
    required_long = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    missing_long = [c for c in required_long if c not in step_metrics_long.columns]
    if missing_long:
        raise ValueError(f"step_metrics_long missing required columns: {missing_long}")

    reward = step_metrics_long[step_metrics_long["metric_key"].astype(str) == REWARD_KEY][required_long].copy()
    reward = reward[reward["run_id"].astype(str).isin(selected_run_ids)].copy()
    reward["global_step_canonical"] = pd.to_numeric(reward["global_step_canonical"], errors="coerce")
    reward["metric_value_float"] = pd.to_numeric(reward["metric_value_float"], errors="coerce")
    reward = reward.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    reward["global_step_canonical"] = reward["global_step_canonical"].astype(int)

    plot_df = reward.merge(
        steps[["run_id", "global_step_canonical", "cumulative_energy_gj"]],
        on=["run_id", "global_step_canonical"],
        how="inner",
        validate="one_to_one",
    ).merge(
        selected_runs,
        on="run_id",
        how="inner",
        validate="many_to_one",
    )
    if plot_df.empty:
        raise ValueError("No reward trajectory rows after joining validation reward to cumulative energy.")

    plot_df = plot_df.sort_values(["experiment_facet", "policy_norm", "global_step_canonical"])

    print("rows used for plotting:")
    print(
        plot_df[
            ["run_id", "experiment_facet", "policy_norm", "global_step_canonical", "cumulative_energy_gj", "metric_value_float"]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), sharex=True, sharey=True)
    for ax, facet in zip(axes, TARGET_EXPERIMENT_FACETS):
        sub = plot_df[plot_df["experiment_facet"] == facet].copy()
        for policy in TARGET_POLICIES:
            psub = sub[sub["policy_norm"] == policy].copy()
            if psub.empty:
                continue
            psub = psub.sort_values("global_step_canonical")
            ax.plot(
                psub["cumulative_energy_gj"],
                psub["metric_value_float"],
                marker="o",
                markersize=5.5,
                linewidth=2.1,
                color=POLICY_COLORS[policy],
                label=POLICY_DISPLAY[policy],
            )

            best_idx = psub["metric_value_float"].idxmax()
            best_row = psub.loc[best_idx]
            ax.scatter(
                [float(best_row["cumulative_energy_gj"])],
                [float(best_row["metric_value_float"])],
                s=90,
                facecolors="none",
                edgecolors=BEST_POINT_COLOR,
                linewidths=1.8,
                zorder=5,
            )

        ax.set_title(EXPERIMENT_DISPLAY[facet], fontweight="bold")
        ax.set_xlabel("Cumulative Energy (GJ)")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Validation Reward (gsm8k reward@1)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=BEST_POINT_COLOR,
            markeredgewidth=1.8,
        )
    )
    labels.append("best validation reward")
    fig.suptitle("Validation Reward vs Cumulative Energy by Reward Mechanism", y=0.99, fontweight="bold")
    fig.legend(handles, labels, title="Policy", frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
