"""Cumulative energy vs validation performance trajectories for baseline runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/baselines/phase_validation_energy_total.png")
ENERGY_TO_GJ = 1e9
GSM8K_METRIC_KEY = "val-core/openai/gsm8k/reward/mean@1"

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
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
BEST_POINT_COLOR = "#D62728"
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _select_baseline_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")
    required = ["run_id", "policy", "model", "logical_run_group"]
    missing = [c for c in required if c not in run_summary.columns]
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
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    expected_slurm = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm
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

    selected = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & slurm_job_mask & checkpoint_mask & integrity_mask
    ][["run_id", "model_facet", "policy_norm"]].drop_duplicates()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def main() -> None:
    selected_runs = _select_baseline_runs()
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

    acc = step_metrics_long[step_metrics_long["metric_key"] == GSM8K_METRIC_KEY][required_long].copy()
    acc = acc[acc["run_id"].astype(str).isin(selected_run_ids)].copy()
    acc["global_step_canonical"] = pd.to_numeric(acc["global_step_canonical"], errors="coerce")
    acc["metric_value_float"] = pd.to_numeric(acc["metric_value_float"], errors="coerce")
    acc = acc.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    acc["global_step_canonical"] = acc["global_step_canonical"].astype(int)

    plot_df = acc.merge(
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
        raise ValueError("No validation trajectory rows after joining validation metrics to cumulative energy.")

    plot_df = plot_df.sort_values(["model_facet", "policy_norm", "global_step_canonical"])

    print("rows used for plotting:")
    print(
        plot_df[
            ["run_id", "model_facet", "policy_norm", "global_step_canonical", "cumulative_energy_gj", "metric_value_float"]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), sharex=True, sharey=True)
    for ax, model in zip(axes, TARGET_MODEL_FACETS):
        sub = plot_df[plot_df["model_facet"] == model].copy()
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
            best_x = float(best_row["metric_value_float"])
            best_y = float(best_row["cumulative_energy_gj"])

            ax.scatter(
                [best_y],
                [best_x],
                s=90,
                facecolors="none",
                edgecolors=BEST_POINT_COLOR,
                linewidths=1.8,
                zorder=5,
            )
        ax.set_title(MODEL_DISPLAY[model], fontweight="bold")
        ax.set_xlabel("Cumulative Energy (GJ)")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Validation Score (gsm8k accuracy)")
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
    labels.append("best validation score")
    fig.suptitle("Validation Performance vs Cumulative Energy by Model and Phase", y=0.99, fontweight="bold")
    fig.legend(handles, labels, title="Policy", frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
