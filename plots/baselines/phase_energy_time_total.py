"""Baseline absolute phase energy/time totals in bar-chart form."""

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
OUTPATH = Path("plots/out/baselines/phase_energy_time_total.png")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
TARGET_MODEL_FACETS = ("Llama", "Qwen")
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "rollout",
    "rl_policy": "preparation",
    "training": "training",
}
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
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


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
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
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    expected_slurm_job_name = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm_job_name
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected_runs = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & checkpoint_mask & slurm_job_mask
    ].copy()
    if selected_runs.empty:
        raise ValueError("No baseline runs selected.")
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

    plot_df["policy_norm"] = plot_df["policy"].astype(str).str.lower()
    plot_df["model_facet"] = plot_df["model"].map(_model_facet)
    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[~plot_df["phase_bucket"].isin(["other", "validation"])].copy()

    run_counts = (
        selected_runs.groupby(["model_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["model_facet", "policy_norm"])
    )
    print("runs included by (model, policy):")
    print(run_counts.to_string(index=False))

    totals = (
        plot_df.groupby(["model_facet", "policy_norm", "phase_bucket"], dropna=False)[["total_energy_j", "phase_time_s"]]
        .sum()
        .reset_index()
    )
    print("phase totals by (model, policy, phase):")
    print(totals.sort_values(["model_facet", "policy_norm", "phase_bucket"]).to_string(index=False))

    fig, axes = plt.subplots(len(TARGET_MODEL_FACETS), len(POLICY_ORDER), figsize=(16, 9))
    x = np.arange(len(PHASE_ORDER), dtype=float)
    width = 0.28
    global_energy_max_kj = max(float(totals["total_energy_j"].max() / ENERGY_SCALE), 1.0)
    global_time_max_min = max(float(totals["phase_time_s"].max() / TIME_SCALE), 1.0)

    for row_idx, facet in enumerate(TARGET_MODEL_FACETS):
        facet_df = totals[totals["model_facet"] == facet]
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
                alpha=0.45,
            )

            ax.set_title(f"{MODEL_DISPLAY[facet]} | {POLICY_DISPLAY[policy]}", fontsize=10, fontweight="bold")
            ax.set_xticks(x, [PHASE_DISPLAY[phase] for phase in PHASE_ORDER], rotation=0)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)

            if col_idx == 0:
                ax.set_ylabel(f"{facet}\nenergy (kJ)")
            if col_idx == len(POLICY_ORDER) - 1:
                ax_time.set_ylabel("time (min)")

            ax.set_ylim(0, global_energy_max_kj * 1.12)
            ax_time.set_ylim(0, global_time_max_min * 1.12)

            for bar, val in zip(energy_bars, energy_vals_kj):
                label_y = bar.get_height() * 0.5
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    _format_energy_kj(float(val)),
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
            for bar, val in zip(time_bars, time_vals_min):
                label_y = bar.get_height() * 0.5
                ax_time.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    _format_time_min(float(val)),
                    ha="center",
                    va="center",
                    rotation=0,
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[phase], edgecolor="black", label=PHASE_DISPLAY[phase]) for phase in PHASE_ORDER
    ]
    metric_handles = [
        Patch(facecolor="#888888", edgecolor="black", label="energy", alpha=1.0),
        Patch(facecolor="#888888", edgecolor="black", label="time", alpha=0.45),
    ]
    fig.suptitle(
        "Total Phase Energy and Time by Model and Policy",
        y=0.995,
        fontweight="bold",
    )
    fig.legend(handles=phase_handles, title="phase colors", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.36, 0.92))
    fig.legend(handles=metric_handles, title="metric type", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.79, 0.92))
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
