"""Cumulative energy required to first reach validation thresholds."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/baselines/validation_threshold_energy.png")
ENERGY_TO_GJ = 1e9
GSM8K_METRIC_KEY = "val-core/openai/gsm8k/reward/mean@1"
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90)

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
    "ppo": "#1f77b4",
    "remax": "#ff7f0e",
    "grpo": "#2ca02c",
}
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
    target_pair_mask = (
        runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    )
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


def _build_validation_energy_df(selected_runs: pd.DataFrame) -> pd.DataFrame:
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)][
        ["run_id", "global_step_canonical", "step_total_energy_j"]
    ].copy()
    steps["global_step_canonical"] = pd.to_numeric(steps["global_step_canonical"], errors="coerce")
    steps["step_total_energy_j"] = pd.to_numeric(steps["step_total_energy_j"], errors="coerce")
    steps = steps.dropna(subset=["global_step_canonical", "step_total_energy_j"]).copy()
    steps["global_step_canonical"] = steps["global_step_canonical"].astype(int)
    steps = steps.sort_values(["run_id", "global_step_canonical"])
    steps["cumulative_energy_gj"] = steps.groupby("run_id")["step_total_energy_j"].cumsum() / ENERGY_TO_GJ

    step_metrics_long, _ = load_view("step_metrics_long")
    acc = step_metrics_long[step_metrics_long["metric_key"] == GSM8K_METRIC_KEY][
        ["run_id", "global_step_canonical", "metric_value_float"]
    ].copy()
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
        raise ValueError("No validation rows after joining validation metrics to cumulative energy.")
    return plot_df.sort_values(["model_facet", "policy_norm", "global_step_canonical"])


def _first_crossing_rows(plot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_facet, policy_norm, run_id), g in plot_df.groupby(["model_facet", "policy_norm", "run_id"], dropna=False):
        g = g.sort_values("global_step_canonical")
        for threshold in THRESHOLDS:
            hit = g[g["metric_value_float"] >= threshold]
            if hit.empty:
                rows.append(
                    {
                        "model_facet": model_facet,
                        "policy_norm": policy_norm,
                        "run_id": run_id,
                        "threshold": threshold,
                        "cumulative_energy_gj": np.nan,
                    }
                )
            else:
                first = hit.iloc[0]
                rows.append(
                    {
                        "model_facet": model_facet,
                        "policy_norm": policy_norm,
                        "run_id": run_id,
                        "threshold": threshold,
                        "cumulative_energy_gj": float(first["cumulative_energy_gj"]),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No threshold-crossing rows available.")
    return out


def main() -> None:
    selected_runs = _select_baseline_runs()
    plot_df = _build_validation_energy_df(selected_runs)
    threshold_df = _first_crossing_rows(plot_df)

    print("threshold crossing rows:")
    print(
        threshold_df.sort_values(["model_facet", "policy_norm", "threshold"]).to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharex=True, sharey=True)
    for ax, model in zip(axes, TARGET_MODEL_FACETS):
        sub = threshold_df[threshold_df["model_facet"] == model].copy()
        for policy in TARGET_POLICIES:
            psub = sub[sub["policy_norm"] == policy].sort_values("threshold")
            if psub.empty:
                continue
            ax.plot(
                psub["threshold"],
                psub["cumulative_energy_gj"],
                marker="o",
                markersize=5.5,
                linewidth=2.1,
                color=POLICY_COLORS[policy],
                label=POLICY_DISPLAY[policy],
            )
        ax.set_title(MODEL_DISPLAY[model], fontweight="bold")
        ax.set_xlabel("Target Validation Threshold")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_xticks(THRESHOLDS)
        ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS])

    axes[0].set_ylabel("Cumulative Energy to First Reach Threshold (GJ)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cumulative Energy to Validation Threshold by Model and Policy", y=0.99, fontweight="bold")
    fig.legend(handles, labels, title="Policy", frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
