"""Actor MFU versus cumulative runtime, split by policy and reward mechanism."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/reward_model/non_results/mfu_actor_critic_vs_time.png")
SECONDS_TO_HOURS = 3600.0
METRIC_KEY = "perf/mfu/actor"

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
EXPERIMENT_LINESTYLE = {
    "Llama Reward Function": "-",
    "Llama Reward Model": "--",
}
EXPERIMENT_ALPHA = {
    "Llama Reward Function": 0.45,
    "Llama Reward Model": 0.98,
}
EXPERIMENT_LINEWIDTH = {
    "Llama Reward Function": 1.9,
    "Llama Reward Model": 2.5,
}


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


def _load_actor_mfu(selected_runs: pd.DataFrame) -> pd.DataFrame:
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    required_steps = ["run_id", "global_step_canonical", "step_time_s"]
    missing_steps = [c for c in required_steps if c not in step_fact.columns]
    if missing_steps:
        raise ValueError(f"step_fact_view missing required columns: {missing_steps}")
    steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)][required_steps].copy()
    steps = apply_analysis_ok(steps)
    steps["global_step_canonical"] = pd.to_numeric(steps["global_step_canonical"], errors="coerce")
    steps["step_time_s"] = pd.to_numeric(steps["step_time_s"], errors="coerce")
    steps = steps.dropna(subset=["global_step_canonical", "step_time_s"]).copy()
    steps["global_step_canonical"] = steps["global_step_canonical"].astype(int)
    steps = steps.sort_values(["run_id", "global_step_canonical"])
    steps["cumulative_time_h"] = steps.groupby("run_id")["step_time_s"].cumsum() / SECONDS_TO_HOURS

    step_metrics_long, _ = load_view("step_metrics_long")
    required_metrics = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    missing_metrics = [c for c in required_metrics if c not in step_metrics_long.columns]
    if missing_metrics:
        raise ValueError(f"step_metrics_long missing required columns: {missing_metrics}")

    metrics = step_metrics_long[
        step_metrics_long["run_id"].astype(str).isin(selected_run_ids)
        & (step_metrics_long["metric_key"].astype(str) == METRIC_KEY)
    ][required_metrics].copy()
    metrics["global_step_canonical"] = pd.to_numeric(metrics["global_step_canonical"], errors="coerce")
    metrics["metric_value_float"] = pd.to_numeric(metrics["metric_value_float"], errors="coerce")
    metrics = metrics.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    metrics["global_step_canonical"] = metrics["global_step_canonical"].astype(int)

    plot_df = metrics.merge(
        steps[["run_id", "global_step_canonical", "cumulative_time_h"]],
        on=["run_id", "global_step_canonical"],
        how="inner",
        validate="many_to_one",
    ).merge(
        selected_runs,
        on="run_id",
        how="inner",
        validate="many_to_one",
    )
    if plot_df.empty:
        raise ValueError("No actor MFU rows after joining metrics to retained steps.")
    return plot_df.sort_values(["policy_norm", "experiment_facet", "run_id", "global_step_canonical"])


def main() -> None:
    selected_runs = _select_runs()
    plot_df = _load_actor_mfu(selected_runs)

    summary = (
        plot_df.groupby(["experiment_facet", "policy_norm"], dropna=False)["metric_value_float"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
        .sort_values(["policy_norm", "experiment_facet"])
    )
    print("actor MFU summary:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(TARGET_POLICIES), figsize=(13.0, 4.9), sharex=True, sharey=True)
    axes = list(axes)

    y_min = float(plot_df["metric_value_float"].min())
    y_max = float(plot_df["metric_value_float"].max())
    y_pad = max(0.0015, 0.08 * (y_max - y_min))

    for ax, policy in zip(axes, TARGET_POLICIES):
        sub = plot_df[plot_df["policy_norm"] == policy].copy()
        for facet in TARGET_EXPERIMENT_FACETS:
            psub = sub[sub["experiment_facet"] == facet].copy()
            if psub.empty:
                continue
            psub = (
                psub.groupby(["run_id", "global_step_canonical", "cumulative_time_h"], as_index=False)["metric_value_float"]
                .mean()
                .sort_values("global_step_canonical")
            )
            ax.plot(
                psub["cumulative_time_h"],
                psub["metric_value_float"],
                color=POLICY_COLORS[policy],
                linestyle=EXPERIMENT_LINESTYLE[facet],
                linewidth=EXPERIMENT_LINEWIDTH[facet],
                alpha=EXPERIMENT_ALPHA[facet],
            )

        ax.set_title(POLICY_DISPLAY[policy], fontweight="bold")
        ax.set_xlabel("Cumulative Runtime (h)")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    axes[0].set_ylabel("Actor MFU")

    policy_handles = [
        Line2D([0], [0], color=POLICY_COLORS[policy], linewidth=2.4, label=POLICY_DISPLAY[policy])
        for policy in TARGET_POLICIES
    ]
    experiment_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=EXPERIMENT_LINESTYLE[facet],
            linewidth=EXPERIMENT_LINEWIDTH[facet],
            alpha=EXPERIMENT_ALPHA[facet],
            label=EXPERIMENT_DISPLAY[facet],
        )
        for facet in TARGET_EXPERIMENT_FACETS
    ]

    fig.suptitle("Actor MFU vs Cumulative Runtime by Policy and Reward Mechanism", y=0.99, fontweight="bold")
    fig.legend(
        policy_handles,
        [h.get_label() for h in policy_handles],
        title="Policy",
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.32, 0.92),
    )
    fig.legend(
        experiment_handles,
        [h.get_label() for h in experiment_handles],
        title="Experiment",
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.76, 0.92),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.1)

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"\nwrote {saved}")


if __name__ == "__main__":
    main()
