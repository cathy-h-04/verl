"""Validation score versus cumulative generated tokens.

Uses the same ReMax proxy correction as the generated-token energy plot:
- PPO/GRPO denominator = logged rollout total tokens
- ReMax denominator = rollout total + extra prompt pass + estimated baseline output tokens
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/baselines/validation_vs_cumulative_tokens.png")
GSM8K_METRIC_KEY = "val-core/openai/gsm8k/reward/mean@1"
TOKENS_TO_M = 1e6

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


def _load_remax_gen_ratio(run_ids: list[str]) -> pd.DataFrame:
    pt, _ = load_view("phase_timings_long")
    df = pt[
        pt["run_id"].astype(str).isin(run_ids)
        & pt["subphase_name"].isin(["gen", "gen_max"])
        & pt["metric_unit"].astype(str).eq("s")
    ][["run_id", "global_step_canonical", "subphase_name", "value"]].copy()
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["global_step_canonical", "value"]).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)

    piv = (
        df.pivot_table(index=["run_id", "global_step_canonical"], columns="subphase_name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    piv["genmax_over_gen"] = piv["gen_max"] / piv["gen"]
    piv.loc[~np.isfinite(piv["genmax_over_gen"]), "genmax_over_gen"] = np.nan
    return piv[["run_id", "global_step_canonical", "genmax_over_gen"]]


def main() -> None:
    selected_runs = _select_baseline_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    token_cols = ["run_id", "global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens"]
    tokens = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)][token_cols].copy()
    for col in token_cols[1:]:
        tokens[col] = pd.to_numeric(tokens[col], errors="coerce")
    tokens = tokens.dropna(subset=token_cols[1:]).copy()
    tokens["global_step_canonical"] = tokens["global_step_canonical"].astype(int)
    tokens["step_rollout_prompt_tokens"] = tokens["step_rollout_total_tokens"] - tokens["step_rollout_output_tokens"]
    tokens = tokens.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    remax_ratio = _load_remax_gen_ratio(selected_runs[selected_runs["policy_norm"] == "remax"]["run_id"].astype(str).tolist())
    tokens = tokens.merge(remax_ratio, on=["run_id", "global_step_canonical"], how="left")
    tokens["genmax_over_gen"] = pd.to_numeric(tokens["genmax_over_gen"], errors="coerce")
    run_medians = tokens.groupby("run_id")["genmax_over_gen"].median().to_dict()
    tokens["genmax_over_gen"] = tokens.apply(
        lambda row: run_medians.get(row["run_id"], np.nan) if pd.isna(row["genmax_over_gen"]) else row["genmax_over_gen"],
        axis=1,
    )

    tokens["corrected_step_tokens"] = tokens["step_rollout_total_tokens"]
    remax_mask = tokens["policy_norm"] == "remax"
    tokens.loc[remax_mask, "corrected_step_tokens"] = (
        tokens.loc[remax_mask, "step_rollout_total_tokens"]
        + tokens.loc[remax_mask, "step_rollout_prompt_tokens"]
        + tokens.loc[remax_mask, "step_rollout_output_tokens"] * tokens.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )
    tokens = tokens[tokens["corrected_step_tokens"] > 0].copy()
    tokens = tokens.sort_values(["run_id", "global_step_canonical"]).drop_duplicates(
        ["run_id", "global_step_canonical"], keep="last"
    )
    tokens["cumulative_tokens_m"] = tokens.groupby("run_id")["corrected_step_tokens"].cumsum() / TOKENS_TO_M

    step_metrics_long, _ = load_view("step_metrics_long")
    sml = step_metrics_long[step_metrics_long["run_id"].astype(str).isin(selected_run_ids)].copy()
    sml["global_step_canonical"] = pd.to_numeric(sml["global_step_canonical"], errors="coerce")
    sml["metric_value_float"] = pd.to_numeric(sml["metric_value_float"], errors="coerce")
    sml = sml.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    sml["global_step_canonical"] = sml["global_step_canonical"].astype(int)

    val = sml[sml["metric_key"] == GSM8K_METRIC_KEY][["run_id", "global_step_canonical", "metric_value_float"]].copy()
    val = val.rename(columns={"metric_value_float": "validation_score"})
    val = val.sort_values(["run_id", "global_step_canonical"]).drop_duplicates(
        ["run_id", "global_step_canonical"], keep="last"
    )

    plot_df = val.merge(
        tokens[["run_id", "global_step_canonical", "cumulative_tokens_m"]],
        on=["run_id", "global_step_canonical"],
        how="inner",
        validate="one_to_one",
    ).merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    if plot_df.empty:
        raise ValueError("No validation rows after joining cumulative tokens.")

    print("rows used:")
    print(plot_df.sort_values(["model_facet", "policy_norm", "global_step_canonical"]).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), sharex=True, sharey=True)
    for ax, model in zip(axes, TARGET_MODEL_FACETS):
        sub = plot_df[plot_df["model_facet"] == model].copy()
        for policy in TARGET_POLICIES:
            psub = sub[sub["policy_norm"] == policy].sort_values("global_step_canonical")
            if psub.empty:
                continue
            ax.plot(
                psub["cumulative_tokens_m"],
                psub["validation_score"],
                marker="o",
                markersize=5.5,
                linewidth=2.1,
                color=POLICY_COLORS[policy],
                label=POLICY_DISPLAY[policy],
            )
        ax.set_title(MODEL_DISPLAY[model], fontweight="bold")
        ax.set_xlabel("Cumulative Generated Tokens (Millions)")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Validation Score (gsm8k accuracy)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Validation Score vs Cumulative Generated Tokens by Model and Policy", y=0.99, fontweight="bold")
    fig.legend(handles, labels, title="Policy", frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
