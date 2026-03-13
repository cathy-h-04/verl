"""Cumulative energy versus proxy-corrected total generated tokens.

For PPO/GRPO, denominator = logged rollout total tokens.
For ReMax, denominator adds a proxy for the hidden baseline-generation pass:
    corrected_tokens = rollout_total_tokens + rollout_prompt_tokens + est_genmax_output_tokens
where
    est_genmax_output_tokens = (gen_max_time / gen_time) * rollout_output_tokens
This approximates an extra prompt prefill plus greedy baseline output tokens.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/baselines/energy_vs_cumulative_generated_tokens.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")
TOKENS_TO_M = 1e6
ENERGY_TO_GJ = 1e9

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
    needed = ["run_id", "global_step_canonical", "step_total_energy_j", "step_rollout_total_tokens", "step_rollout_output_tokens"]
    steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    for col in needed[1:]:
        steps[col] = pd.to_numeric(steps[col], errors="coerce")
    steps = steps.dropna(subset=["global_step_canonical", "step_total_energy_j", "step_rollout_total_tokens", "step_rollout_output_tokens"]).copy()
    steps["global_step_canonical"] = steps["global_step_canonical"].astype(int)
    steps["step_rollout_prompt_tokens"] = steps["step_rollout_total_tokens"] - steps["step_rollout_output_tokens"]
    steps = steps.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    remax_ratio = _load_remax_gen_ratio(selected_runs[selected_runs["policy_norm"] == "remax"]["run_id"].astype(str).tolist())
    steps = steps.merge(remax_ratio, on=["run_id", "global_step_canonical"], how="left")
    steps["genmax_over_gen"] = pd.to_numeric(steps["genmax_over_gen"], errors="coerce")

    # fallback to run-level median ratio for missing ReMax steps
    run_medians = steps.groupby("run_id")["genmax_over_gen"].median().to_dict()
    steps["genmax_over_gen"] = steps.apply(
        lambda row: run_medians.get(row["run_id"], np.nan) if pd.isna(row["genmax_over_gen"]) else row["genmax_over_gen"],
        axis=1,
    )

    steps["corrected_step_tokens"] = steps["step_rollout_total_tokens"]
    remax_mask = steps["policy_norm"] == "remax"
    steps.loc[remax_mask, "corrected_step_tokens"] = (
        steps.loc[remax_mask, "step_rollout_total_tokens"]
        + steps.loc[remax_mask, "step_rollout_prompt_tokens"]
        + steps.loc[remax_mask, "step_rollout_output_tokens"] * steps.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )

    steps = steps[steps["corrected_step_tokens"] > 0].copy()
    steps = steps.sort_values(["run_id", "global_step_canonical"])
    steps["cumulative_energy_gj"] = steps.groupby("run_id")["step_total_energy_j"].cumsum() / ENERGY_TO_GJ
    steps["cumulative_corrected_tokens_m"] = steps.groupby("run_id")["corrected_step_tokens"].cumsum() / TOKENS_TO_M

    print("rows used:")
    print(
        steps[
            [
                "run_id",
                "global_step_canonical",
                "step_rollout_total_tokens",
                "step_rollout_prompt_tokens",
                "step_rollout_output_tokens",
                "genmax_over_gen",
                "corrected_step_tokens",
                "cumulative_energy_gj",
                "cumulative_corrected_tokens_m",
                "model_facet",
                "policy_norm",
            ]
        ].sort_values(["model_facet", "policy_norm", "global_step_canonical"]).to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharex=True, sharey=True)
    for ax, model in zip(axes, TARGET_MODEL_FACETS):
        sub = steps[steps["model_facet"] == model].copy()
        for policy in TARGET_POLICIES:
            psub = sub[sub["policy_norm"] == policy].sort_values("global_step_canonical")
            if psub.empty:
                continue
            ax.plot(
                psub["cumulative_corrected_tokens_m"],
                psub["cumulative_energy_gj"],
                linewidth=2.0,
                color=POLICY_COLORS[policy],
                label=POLICY_DISPLAY[policy],
            )
        ax.set_title(MODEL_DISPLAY[model], fontweight="bold")
        ax.set_xlabel("Cumulative Generated Tokens (Millions)")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cumulative Energy (GJ)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cumulative Energy vs Cumulative Generated Tokens by Model and Policy", y=0.99, fontweight="bold")
    fig.legend(handles, labels, title="Policy", frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="energy_vs_cumulative_generated_tokens",
        run_ids=selected_run_ids,
        data_sources={
            "views": ["run_summary_view", "runs", "step_fact_view", "phase_timings_long"],
            "proxy_formula": "rollout_total + rollout_prompt + (gen_max/gen)*rollout_output for ReMax only",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
