"""Preparation-only energy/time totals for reward-function vs reward-model runs."""

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
OUTPATH = Path("plots/out/reward_model/phase_energy_time_total.png")
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
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "rollout",
    "rl_policy": "preparation",
    "training": "training",
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
ENERGY_SCALE = 1000.0
TIME_SCALE = 60.0
TARGET_PHASE = "rl_policy"


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
    required_cols = ["run_id", "phase_name", "phase_time_s", "total_energy_j", "policy", "model"]
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

    plot_df = plot_df.merge(selected_runs[["run_id", "experiment_facet"]], on="run_id", how="left", validate="many_to_one")
    plot_df["policy_norm"] = plot_df["policy"].astype(str).str.lower()
    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[~plot_df["phase_bucket"].isin(["other", "validation"])].copy()

    run_counts = (
        selected_runs.groupby(["experiment_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["experiment_facet", "policy_norm"])
    )
    print("runs included by (experiment, policy):")
    print(run_counts.to_string(index=False))

    totals = (
        plot_df.groupby(["experiment_facet", "policy_norm", "phase_bucket"], dropna=False)[["total_energy_j", "phase_time_s"]]
        .sum()
        .reset_index()
    )
    print("phase totals by (experiment, policy, phase):")
    print(totals.sort_values(["experiment_facet", "policy_norm", "phase_bucket"]).to_string(index=False))

    prep_totals = (
        totals[totals["phase_bucket"] == TARGET_PHASE]
        .copy()
        .set_index(["experiment_facet", "policy_norm"])
        .sort_index()
    )
    print("preparation totals by (experiment, policy):")
    print(prep_totals.reset_index().to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)
    x = np.arange(len(POLICY_ORDER), dtype=float)
    width = 0.34
    global_energy_max_kj = max(float(prep_totals["total_energy_j"].max() / ENERGY_SCALE), 1.0)
    global_time_max_min = max(float(prep_totals["phase_time_s"].max() / TIME_SCALE), 1.0)

    for panel_idx, facet in enumerate(TARGET_EXPERIMENT_FACETS):
        ax = axes[panel_idx]
        ax_time = ax.twinx()
        facet_df = prep_totals.loc[facet] if facet in prep_totals.index.get_level_values(0) else pd.DataFrame()
        if isinstance(facet_df, pd.Series):
            facet_df = facet_df.to_frame().T
        facet_df = facet_df.reindex(POLICY_ORDER).fillna(0.0)

        energy_vals_kj = facet_df["total_energy_j"].to_numpy(dtype=float) / ENERGY_SCALE
        time_vals_min = facet_df["phase_time_s"].to_numpy(dtype=float) / TIME_SCALE
        colors = [POLICY_COLORS[policy] for policy in POLICY_ORDER]

        energy_bars = ax.bar(
            x - width / 2,
            energy_vals_kj,
            width=width,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )
        time_bars = ax_time.bar(
            x + width / 2,
            time_vals_min,
            width=width,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.35,
        )

        ax.set_title(EXPERIMENT_DISPLAY[facet], fontsize=11, fontweight="bold")
        ax.set_xticks(x, [POLICY_DISPLAY[policy] for policy in POLICY_ORDER], rotation=0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, global_energy_max_kj * 1.14)
        ax_time.set_ylim(0, global_time_max_min * 1.14)

        if panel_idx == 0:
            ax.set_ylabel("Preparation Energy (kJ)")
        if panel_idx == len(TARGET_EXPERIMENT_FACETS) - 1:
            ax_time.set_ylabel("Preparation Time (min)")

        for bar, val in zip(energy_bars, energy_vals_kj):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + global_energy_max_kj * 0.025,
                _format_energy_kj(float(val)),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                fontweight="bold",
            )
        for bar, val in zip(time_bars, time_vals_min):
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + global_time_max_min * 0.025,
                _format_time_min(float(val)),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    metric_handles = [
        Patch(facecolor="#888888", edgecolor="black", label="energy", alpha=1.0),
        Patch(facecolor="#888888", edgecolor="black", label="time", alpha=0.35),
    ]
    fig.suptitle("Preparation Energy and Time by Policy and Reward Mechanism", y=0.99, fontweight="bold")
    fig.legend(
        handles=metric_handles,
        title="metric type",
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
