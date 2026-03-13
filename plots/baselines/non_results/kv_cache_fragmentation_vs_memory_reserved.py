"""Sequence length versus reserved memory for baseline runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/baselines/non_results/kv_cache_fragmentation_vs_memory_reserved.png")
SEQ_KEY = "global_seqlen/mean"
MEM_KEY = "perf/max_memory_reserved_gb"

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {"Llama": "Llama-3.1-8B-Inst", "Qwen": "Qwen2.5-3B-Inst"}
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
POLICY_COLORS = {"ppo": "#5B2A86", "remax": "#FF5C7A", "grpo": "#0097A7"}
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


def _eligible_steps(selected_run_ids: list[str]) -> pd.DataFrame:
    phase_fact, _ = load_view("phase_fact_view")
    pf = phase_fact[phase_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    pf = apply_analysis_ok(pf)
    eligible = pf[["run_id", "global_step_canonical"]].drop_duplicates()
    eligible["global_step_canonical"] = pd.to_numeric(eligible["global_step_canonical"], errors="coerce").astype(int)
    return eligible


def main() -> None:
    selected_runs = _select_baseline_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    eligible = _eligible_steps(selected_run_ids)

    step_metrics_long, _ = load_view("step_metrics_long")
    sml = step_metrics_long[step_metrics_long["run_id"].astype(str).isin(selected_run_ids)].copy()
    sml["global_step_canonical"] = pd.to_numeric(sml["global_step_canonical"], errors="coerce")
    sml["metric_value_float"] = pd.to_numeric(sml["metric_value_float"], errors="coerce")
    sml = sml.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    sml["global_step_canonical"] = sml["global_step_canonical"].astype(int)
    sml = sml.merge(eligible, on=["run_id", "global_step_canonical"], how="inner")

    seq_df = sml[sml["metric_key"] == SEQ_KEY][["run_id", "global_step_canonical", "metric_value_float"]].copy()
    seq_df = seq_df.rename(columns={"metric_value_float": "global_seqlen_mean"})
    mem_df = sml[sml["metric_key"] == MEM_KEY][["run_id", "global_step_canonical", "metric_value_float"]].copy()
    mem_df = mem_df.rename(columns={"metric_value_float": "max_memory_reserved_gb"})

    plot_df = seq_df.merge(
        mem_df,
        on=["run_id", "global_step_canonical"],
        how="inner",
        validate="one_to_one",
    ).merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    if plot_df.empty:
        raise ValueError("No rows after joining sequence length and reserved memory.")

    print("rows used:")
    print(plot_df.sort_values(["model_facet", "policy_norm", "global_step_canonical"]).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharex=True, sharey=True)
    for ax, model in zip(axes, TARGET_MODEL_FACETS):
        sub = plot_df[plot_df["model_facet"] == model].copy()
        for policy in TARGET_POLICIES:
            psub = sub[sub["policy_norm"] == policy].copy()
            if psub.empty:
                continue
            ax.scatter(
                psub["global_seqlen_mean"],
                psub["max_memory_reserved_gb"],
                s=40,
                alpha=0.8,
                color=POLICY_COLORS[policy],
                label=POLICY_DISPLAY[policy],
            )
        ax.set_title(MODEL_DISPLAY[model], fontweight="bold")
        ax.set_xlabel("global_seqlen/mean")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("perf/max_memory_reserved_gb")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("KV Cache Fragmentation vs Memory Reserved", y=0.99, fontweight="bold")
    fig.legend(handles, labels, title="Policy", frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
