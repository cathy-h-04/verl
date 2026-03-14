"""Average GPU power by phase for reward-function vs reward-model runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


INCLUDE_VALIDATION = False
OUTPATH = Path("plots/out/reward_model/phase_avg_power_violin.png")
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
TARGET_EXPERIMENT_FACETS = ("Llama Reward Function", "Llama Reward Model")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama Reward Function": "llama_new_baseline",
    "Llama Reward Model": "llama_rm_gsm8k",
}
LOGICAL_GROUP_PREFIXES_BY_FACET = {
    "Llama Reward Function": ("stage1_llama8b_",),
    "Llama Reward Model": ("llama8b_",),
}
TARGET_PHASE = "rl_policy"
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
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
MECHANISM_ORDER = ("Llama Reward Function", "Llama Reward Model")
MECHANISM_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
MECHANISM_COLORS = {
    "Llama Reward Function": "#D04A1C",
    "Llama Reward Model": "#295894",
}


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _experiment_facet(slurm_job_name: str, logical_run_group: str) -> str:
    slurm_text = str(slurm_job_name).strip().lower()
    logical_text = str(logical_run_group).strip().lower()
    for facet in TARGET_EXPERIMENT_FACETS:
        expected_slurm = TARGET_SLURM_JOB_NAME_BY_FACET[facet]
        logical_prefixes = LOGICAL_GROUP_PREFIXES_BY_FACET[facet]
        if slurm_text == expected_slurm and logical_text.startswith(logical_prefixes):
            return facet
    return "Other"


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


def _load_run_summary_for_selection() -> pd.DataFrame:
    df_runs, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "model", "logical_run_group"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(
            "run_summary_view is missing required selection columns "
            f"{missing}. Available columns: {list(df_runs.columns)}"
        )
    return df_runs.copy()


def _load_runs_with_slurm_metadata() -> pd.DataFrame:
    df_runs, _ = load_view("runs")
    required = ["run_id", "slurm_job_name"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(
            "runs is missing required slurm metadata columns "
            f"{missing}. Available columns: {list(df_runs.columns)}"
        )
    return df_runs[required].copy()


def _select_runs() -> pd.DataFrame:
    runs_df = _load_run_summary_for_selection()
    runs_meta_df = _load_runs_with_slurm_metadata()
    runs_df = runs_df.merge(runs_meta_df, on="run_id", how="left", validate="one_to_one")

    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()
    runs_df["experiment_facet"] = [
        _experiment_facet(slurm_job_name=slurm_job_name, logical_run_group=logical_run_group)
        for slurm_job_name, logical_run_group in zip(runs_df["slurm_job_name"], runs_df["logical_run_group"])
    ]

    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["experiment_facet"].isin(
        TARGET_EXPERIMENT_FACETS
    )
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected_runs = runs_df[non_rollout_knob_mask & target_pair_mask & checkpoint_mask].copy()
    if selected_runs.empty:
        raise ValueError("No reward-model runs selected.")
    return selected_runs


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

    plot_df = plot_df.merge(selected_runs[["run_id", "experiment_facet"]], on="run_id", how="left", validate="many_to_one")
    plot_df["policy_norm"] = plot_df["policy"].astype(str).str.lower()
    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[plot_df["phase_bucket"] == TARGET_PHASE].copy()
    plot_df["avg_power_w"] = pd.to_numeric(plot_df["avg_power_w"], errors="coerce")
    plot_df = plot_df.dropna(subset=["avg_power_w"]).copy()

    run_counts = (
        selected_runs.groupby(["experiment_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["experiment_facet", "policy_norm"])
    )
    print("runs included by (experiment, policy):")
    print(run_counts.to_string(index=False))

    point_counts = (
        plot_df.groupby(["experiment_facet", "policy_norm"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["experiment_facet", "policy_norm"])
    )
    print("preparation points plotted by (experiment, policy):")
    print(point_counts.to_string(index=False))

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.2))

    data = []
    positions = []
    mechanism_by_violin = []
    max_power = float(plot_df["avg_power_w"].max()) if not plot_df.empty else 0.0
    medians_by_policy: dict[str, dict[str, float]] = {policy: {} for policy in POLICY_ORDER}

    for policy_i, policy in enumerate(POLICY_ORDER, start=1):
        center = float(policy_i)
        for mechanism_i, mechanism in enumerate(MECHANISM_ORDER):
            vals = plot_df.loc[
                (plot_df["policy_norm"] == policy) & (plot_df["experiment_facet"] == mechanism),
                "avg_power_w",
            ].tolist()
            if not vals:
                continue
            offset = -0.16 if mechanism_i == 0 else 0.16
            data.append(vals)
            positions.append(center + offset)
            mechanism_by_violin.append(mechanism)
            medians_by_policy[policy][mechanism] = float(np.median(vals))

    if not data:
        raise ValueError("No preparation-phase data available for plotting.")

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.26,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, mechanism in zip(vp["bodies"], mechanism_by_violin):
        body.set_facecolor(MECHANISM_COLORS[mechanism])
        body.set_edgecolor("black")
        body.set_linewidth(0.7)
        body.set_alpha(0.45)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.09,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 2.0, "alpha": 0.25, "markerfacecolor": "black", "markeredgecolor": "black"},
    )

    y_top = max_power * 1.08 if max_power > 0 else 1.0
    for policy_i, policy in enumerate(POLICY_ORDER, start=1):
        medians = medians_by_policy.get(policy, {})
        if len(medians) < 2:
            continue
        baseline = medians.get("Llama Reward Function")
        reward_model = medians.get("Llama Reward Model")
        if baseline is None or reward_model is None or baseline <= 0:
            continue
        pct_increase = ((reward_model - baseline) / baseline) * 100.0
        ax.text(
            float(policy_i),
            y_top,
            f"{pct_increase:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(range(1, len(POLICY_ORDER) + 1))
    ax.set_xticklabels([POLICY_DISPLAY[p] for p in POLICY_ORDER])
    ax.set_xlabel("Policy", fontweight="bold")
    ax.set_ylabel("Preparation Average GPU Power (W)", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_ylim(top=y_top * 1.06)

    mechanism_handles = [
        Patch(facecolor=MECHANISM_COLORS[m], edgecolor="black", alpha=0.45, label=MECHANISM_DISPLAY[m])
        for m in MECHANISM_ORDER
    ]
    fig.legend(
        handles=mechanism_handles,
        title="Reward Mechanism",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )
    fig.suptitle("Average Power by Policy and Reward Mechanism", y=0.985, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
