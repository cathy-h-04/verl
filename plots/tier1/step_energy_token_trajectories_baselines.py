"""Baseline trajectories with two lines per panel:
1) step_total_energy_j
2) step_rollout_output_tokens

x-axis: absolute_global_step
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier1/step_energy_token_trajectories_baselines.png")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_MODEL_FACETS = ("Llama", "Qwen")
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
    required = [
        "run_id",
        "policy",
        "model",
        "logical_run_group",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
    ]
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
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    runs_df["logical_run_group"] = runs_df["logical_run_group"].astype(str).str.lower()
    runs_df["slurm_job_name"] = runs_df["slurm_job_name"].astype(str).str.lower()
    runs_df["is_checkpoint_continuation"] = runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
    runs_df["join_coverage_rate"] = pd.to_numeric(runs_df["join_coverage_rate"], errors="coerce")
    runs_df["phase_boundary_integrity_rate"] = pd.to_numeric(runs_df["phase_boundary_integrity_rate"], errors="coerce")

    baseline_label_mask = runs_df["logical_run_group"].str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(("Llama", "Qwen"))
    expected_slurm = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"] == expected_slurm
    integrity_mask = (runs_df["join_coverage_rate"] == 1.0) & (runs_df["phase_boundary_integrity_rate"] == 1.0)
    baseline_mask = (~runs_df["is_checkpoint_continuation"]) & baseline_label_mask & target_pair_mask & slurm_job_mask & integrity_mask

    selected = runs_df.loc[baseline_mask, ["run_id", "policy_norm", "model_facet"]].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def main() -> None:
    selected_runs = _select_baseline_runs()
    print("selected runs:")
    print(selected_runs.sort_values(["policy_norm", "model_facet"]).to_string(index=False))

    step, _ = load_view("step_fact_view")
    required_step = [
        "run_id",
        "absolute_global_step",
        "step_total_energy_j",
        "step_rollout_output_tokens",
        "is_warmup_idle",
        "is_incomplete_phase",
        "is_outlier_sample",
        "boundary_integrity_ok",
        "join_integrity_ok",
    ]
    missing_step = [c for c in required_step if c not in step.columns]
    if missing_step:
        raise ValueError(f"step_fact_view missing required columns: {missing_step}")

    df = step[step["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))].copy()
    df = df.merge(selected_runs, on="run_id", how="inner")

    df["boundary_integrity_ok"] = df["boundary_integrity_ok"].fillna(True).astype(bool)
    df["join_integrity_ok"] = df["join_integrity_ok"].fillna(True).astype(bool)
    df["is_warmup_idle"] = df["is_warmup_idle"].fillna(False).astype(bool)
    df["is_incomplete_phase"] = df["is_incomplete_phase"].fillna(False).astype(bool)
    df["is_outlier_sample"] = df["is_outlier_sample"].fillna(False).astype(bool)
    df = df[
        df["boundary_integrity_ok"]
        & df["join_integrity_ok"]
        & (~df["is_warmup_idle"])
        & (~df["is_incomplete_phase"])
        & (~df["is_outlier_sample"])
    ].copy()

    df["absolute_global_step"] = pd.to_numeric(df["absolute_global_step"], errors="coerce")
    df["step_total_energy_j"] = pd.to_numeric(df["step_total_energy_j"], errors="coerce")
    df["step_rollout_output_tokens"] = pd.to_numeric(df["step_rollout_output_tokens"], errors="coerce")
    df = df.dropna(subset=["absolute_global_step", "step_total_energy_j", "step_rollout_output_tokens"]).copy()
    df["absolute_global_step"] = df["absolute_global_step"].astype(int)

    curve = (
        df.groupby(["policy_norm", "model_facet", "absolute_global_step"], dropna=False)[
            ["step_total_energy_j", "step_rollout_output_tokens"]
        ]
        .mean()
        .reset_index()
        .sort_values(["policy_norm", "model_facet", "absolute_global_step"])
    )
    print("trajectory coverage by (policy, model):")
    print(
        curve.groupby(["policy_norm", "model_facet"], dropna=False)["absolute_global_step"]
        .agg(min_step="min", max_step="max", n_steps="nunique")
        .reset_index()
        .sort_values(["policy_norm", "model_facet"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(
        len(TARGET_MODEL_FACETS),
        len(TARGET_POLICIES),
        figsize=(16, 8),
        sharex=True,
    )
    if len(TARGET_MODEL_FACETS) == 1:
        axes = [axes]

    for r, model in enumerate(TARGET_MODEL_FACETS):
        for c, policy in enumerate(TARGET_POLICIES):
            ax1 = axes[r][c]
            ax2 = ax1.twinx()
            sub = curve[(curve["policy_norm"] == policy) & (curve["model_facet"] == model)].copy()
            if sub.empty:
                ax1.set_title(f"{policy.upper()} | {model} (no data)")
                continue
            sub = sub.sort_values("absolute_global_step")
            ax1.plot(
                sub["absolute_global_step"],
                sub["step_total_energy_j"],
                color="#1f77b4",
                linewidth=1.2,
                alpha=0.5,
            )
            ax1.scatter(
                sub["absolute_global_step"],
                sub["step_total_energy_j"],
                color="#1f77b4",
                s=18,
                alpha=0.9,
            )
            ax2.plot(
                sub["absolute_global_step"],
                sub["step_rollout_output_tokens"],
                color="#ff7f0e",
                linewidth=1.2,
                alpha=0.5,
            )
            ax2.scatter(
                sub["absolute_global_step"],
                sub["step_rollout_output_tokens"],
                color="#ff7f0e",
                s=18,
                alpha=0.9,
            )
            ax1.grid(alpha=0.2)
            ax1.set_title(f"{policy.upper()} | {model}")
            if c == 0:
                ax1.set_ylabel("step_total_energy_j", color="#1f77b4")
            if c == len(TARGET_POLICIES) - 1:
                ax2.set_ylabel("step_rollout_output_tokens", color="#ff7f0e")
            if r == len(TARGET_MODEL_FACETS) - 1:
                ax1.set_xlabel("absolute_global_step")

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.0, label="step_total_energy_j"),
        Line2D([0], [0], color="#ff7f0e", lw=2.0, label="step_rollout_output_tokens"),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Baseline Trajectories by Policy and Model: Step Energy (J) and Output Tokens", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
