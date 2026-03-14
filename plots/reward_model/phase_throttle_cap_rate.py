"""SW power-cap throttle rate by phase and reward mechanism."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/reward_model/phase_throttle_cap_rate.png")
INCLUDE_VALIDATION = False

TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
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
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {"rollout": "Rollout", "rl_policy": "Preparation", "training": "Training"}

EXPERIMENT_COLORS = {"Llama Reward Function": "#D04A1C", "Llama Reward Model": "#295894"}
JITTER_ALPHA = 0.35
JITTER_SIZE = 3.5
BAR_WIDTH = 0.32


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

    selected = runs_df[non_rollout_knob_mask & target_mask & checkpoint_mask & integrity_mask].copy()
    if selected.empty:
        raise ValueError("No reward-mechanism runs selected.")
    return selected


def main() -> None:
    selected_runs = _select_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    selected_meta = selected_runs[["run_id", "experiment_facet", "policy_norm"]].drop_duplicates()

    phase_fact, _ = load_view("phase_fact_view")
    df_before = phase_fact[phase_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    df = apply_analysis_ok(df_before)
    filter_info = explain_filtering(df_before, df)
    print("[filtering]", filter_info)

    if not INCLUDE_VALIDATION and "is_validation_step" in df.columns:
        df = df[~df["is_validation_step"].fillna(False)].copy()

    df["phase_n"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_n"].isin(PHASE_ORDER)].copy()
    df["throttle_rate"] = pd.to_numeric(df["throttle_sw_power_cap_rate"], errors="coerce")
    df = df.dropna(subset=["throttle_rate"]).copy()
    df = df.merge(selected_meta, on="run_id", how="inner")

    step_df = df.groupby(
        ["run_id", "global_step_canonical", "phase_n", "experiment_facet", "policy_norm"],
        as_index=False,
    )["throttle_rate"].mean()

    summary = step_df.groupby(["phase_n", "experiment_facet", "policy_norm"], as_index=False).agg(
        mean=("throttle_rate", "mean"),
        std=("throttle_rate", "std"),
        n=("throttle_rate", "count"),
    )

    print("\nplot summary (mean throttle cap rate):")
    print(summary.sort_values(["policy_norm", "phase_n", "experiment_facet"]).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)
    x = np.arange(len(POLICY_ORDER), dtype=float)

    for ax, phase in zip(axes, PHASE_ORDER):
        phase_sum = summary[summary["phase_n"] == phase].copy()
        phase_steps = step_df[step_df["phase_n"] == phase].copy()

        for i, experiment_facet in enumerate(TARGET_EXPERIMENT_FACETS):
            offset = (i - 0.5) * BAR_WIDTH
            xpos = x + offset
            color = EXPERIMENT_COLORS[experiment_facet]

            means = []
            stds = []
            for policy in POLICY_ORDER:
                row = phase_sum[
                    (phase_sum["experiment_facet"] == experiment_facet) & (phase_sum["policy_norm"] == policy)
                ]
                means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
                stds.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

            ax.bar(
                xpos,
                means,
                width=BAR_WIDTH,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                label=EXPERIMENT_DISPLAY[experiment_facet] if phase == PHASE_ORDER[0] else None,
                zorder=2,
            )
            ax.errorbar(
                xpos,
                means,
                yerr=stds,
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1.0,
                zorder=3,
            )

            for xp, m, s in zip(xpos, means, stds):
                if np.isnan(m):
                    continue
                top = m + s
                ax.text(
                    xp,
                    top + 0.055,
                    f"\u00b1{s:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                    zorder=5,
                )

            rng = np.random.default_rng(42 + i)
            for j, policy in enumerate(POLICY_ORDER):
                pts = phase_steps[
                    (phase_steps["experiment_facet"] == experiment_facet) & (phase_steps["policy_norm"] == policy)
                ]["throttle_rate"].to_numpy()
                if pts.size == 0:
                    continue
                jx = xpos[j] + rng.uniform(-BAR_WIDTH * 0.4, BAR_WIDTH * 0.4, size=len(pts))
                ax.scatter(jx, pts, s=JITTER_SIZE, color=color, alpha=JITTER_ALPHA, zorder=4, linewidths=0)

        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_DISPLAY[p] for p in POLICY_ORDER])
        ax.set_xlabel("Policy")
        ax.set_title(PHASE_DISPLAY[phase], fontweight="bold")
        ax.set_ylim(-0.04, 1.28)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.6)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(10)
        ax.title.set_size(11)

    axes[0].set_ylabel("Fraction of samples\nwith SW power-cap throttle", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.97),
        fontsize=9,
    )
    fig.suptitle(
        "Phase SW Power-Cap Throttle Rate by Policy and Reward Mechanism",
        y=1.02,
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"\nwrote {saved}")


if __name__ == "__main__":
    main()
