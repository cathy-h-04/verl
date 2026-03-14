"""Stacked phase J/proxy-generated-token by policy, grouped by reward mechanism.

Proxy correction mirrors baselines:
- PPO/GRPO denominator uses logged rollout total tokens.
- ReMax denominator uses:
    rollout_total + rollout_prompt + (gen_max/gen) * rollout_output
"""

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
OUTPATH = Path("plots/out/reward_model/j_per_output_token.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_EXPERIMENT_FACETS = ("Llama Reward Function", "Llama Reward Model")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama Reward Function": "llama_new_baseline",
    "Llama Reward Model": "llama_rm_gsm8k",
}
LOGICAL_GROUP_PREFIXES_BY_FACET = {
    "Llama Reward Function": ("stage1_llama8b_",),
    "Llama Reward Model": ("llama8b_",),
}
PHASE_STACK_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
PHASE_COLORS = {
    "rollout": "#4C78A8",
    "rl_policy": "#54A24B",
    "training": "#F58518",
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
    return selected_runs[["run_id", "policy_norm", "experiment_facet"]].drop_duplicates()


def _load_phase_energy(selected_run_ids: list[str]) -> pd.DataFrame:
    phase_df, _ = load_view("phase_fact_view")
    required_cols = ["run_id", "phase_name", "total_energy_j"]
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
    missing_required = [col for col in required_cols if col not in phase_df.columns]
    if missing_required:
        raise ValueError(
            "phase_fact_view is missing required columns "
            f"{missing_required}. Available columns: {list(phase_df.columns)}"
        )

    use_cols = [col for col in required_cols + filter_cols_optional if col in phase_df.columns]
    phase_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)][use_cols].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    filtering = explain_filtering(before, phase_df)
    print(f"phase filtering={filtering}")

    if not INCLUDE_VALIDATION:
        phase_df = phase_df[phase_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
    phase_df["total_energy_j"] = pd.to_numeric(phase_df["total_energy_j"], errors="coerce")
    phase_df = phase_df[phase_df["phase_bucket"].isin(PHASE_STACK_ORDER)].dropna(subset=["total_energy_j"]).copy()
    return phase_df


def _load_remax_gen_ratio(run_ids: list[str]) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame(columns=["run_id", "global_step_canonical", "genmax_over_gen"])

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


def _load_corrected_output_tokens(selected_runs: pd.DataFrame) -> pd.DataFrame:
    step_df, _ = load_view("step_fact_view")
    needed = ["run_id", "global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens"]
    missing = [col for col in needed if col not in step_df.columns]
    if missing:
        raise ValueError(
            "step_fact_view is missing required columns "
            f"{missing}. Available columns: {list(step_df.columns)}"
        )

    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    out = step_df[step_df["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    out["global_step_canonical"] = pd.to_numeric(out["global_step_canonical"], errors="coerce")
    out["step_rollout_total_tokens"] = pd.to_numeric(out["step_rollout_total_tokens"], errors="coerce")
    out["step_rollout_output_tokens"] = pd.to_numeric(out["step_rollout_output_tokens"], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out["step_rollout_prompt_tokens"] = out["step_rollout_total_tokens"] - out["step_rollout_output_tokens"]

    out = out.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    remax_ids = selected_runs[selected_runs["policy_norm"] == "remax"]["run_id"].astype(str).tolist()
    remax_ratio = _load_remax_gen_ratio(remax_ids)
    out = out.merge(remax_ratio, on=["run_id", "global_step_canonical"], how="left")
    out["genmax_over_gen"] = pd.to_numeric(out["genmax_over_gen"], errors="coerce")

    run_medians = out.groupby("run_id")["genmax_over_gen"].median().to_dict()
    out["genmax_over_gen"] = out.apply(
        lambda row: run_medians.get(row["run_id"], np.nan) if pd.isna(row["genmax_over_gen"]) else row["genmax_over_gen"],
        axis=1,
    )

    out["corrected_step_output_tokens"] = out["step_rollout_output_tokens"]
    remax_mask = out["policy_norm"] == "remax"
    out["corrected_step_output_tokens"] = out["step_rollout_total_tokens"]
    out.loc[remax_mask, "corrected_step_output_tokens"] = (
        out.loc[remax_mask, "step_rollout_total_tokens"]
        + out.loc[remax_mask, "step_rollout_prompt_tokens"]
        + out.loc[remax_mask, "step_rollout_output_tokens"] * out.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )

    out = out[out["corrected_step_output_tokens"] > 0].copy()
    per_run_tokens = (
        out.groupby(["run_id", "experiment_facet", "policy_norm"], dropna=False)["corrected_step_output_tokens"]
        .sum()
        .rename("corrected_output_tokens")
        .reset_index()
    )
    return per_run_tokens


def main() -> None:
    selected_runs = _select_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_energy = _load_phase_energy(selected_run_ids)
    phase_energy = phase_energy.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    phase_energy_run = (
        phase_energy.groupby(["run_id", "experiment_facet", "policy_norm", "phase_bucket"], dropna=False)["total_energy_j"]
        .sum()
        .reset_index()
    )

    per_run_tokens = _load_corrected_output_tokens(selected_runs)

    merged = phase_energy_run.merge(
        per_run_tokens[["run_id", "corrected_output_tokens"]],
        on="run_id",
        how="inner",
        validate="many_to_one",
    )
    merged = merged[merged["corrected_output_tokens"] > 0].copy()
    merged["j_per_output_token"] = merged["total_energy_j"] / merged["corrected_output_tokens"]

    run_counts = (
        selected_runs.groupby(["experiment_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["experiment_facet", "policy_norm"])
    )
    print("runs included by (experiment, policy):")
    print(run_counts.to_string(index=False))

    summary = (
        merged.groupby(["experiment_facet", "policy_norm", "phase_bucket"], dropna=False)["j_per_output_token"]
        .mean()
        .rename("mean_j_per_output_token")
        .reset_index()
    )

    totals = (
        summary.groupby(["experiment_facet", "policy_norm"], dropna=False)["mean_j_per_output_token"]
        .sum()
        .rename("total_j_per_output_token")
        .reset_index()
    )
    print("mean J/output-token totals by (experiment, policy):")
    print(totals.sort_values(["experiment_facet", "policy_norm"]).to_string(index=False))

    bar_positions = []
    bar_labels = []
    bar_pairs = []
    x = 0.0
    policy_spacing = 0.95
    facet_gap = 1.3
    for facet in TARGET_EXPERIMENT_FACETS:
        for policy in TARGET_POLICIES:
            bar_positions.append(x)
            bar_labels.append(policy.upper() if policy != "remax" else "ReMax")
            bar_pairs.append((facet, policy))
            x += policy_spacing
        x += facet_gap

    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    bar_width = 0.72
    bottoms = np.zeros(len(bar_positions), dtype=float)

    for phase in PHASE_STACK_ORDER:
        vals = []
        for facet, policy in bar_pairs:
            row = summary[
                (summary["experiment_facet"] == facet)
                & (summary["policy_norm"] == policy)
                & (summary["phase_bucket"] == phase)
            ]
            vals.append(float(row["mean_j_per_output_token"].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            bar_positions,
            vals,
            width=bar_width,
            bottom=bottoms,
            color=PHASE_COLORS[phase],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            label=PHASE_DISPLAY[phase],
        )
        bottoms += np.array(vals, dtype=float)

    for xpos, total in zip(bar_positions, bottoms):
        ax.text(
            xpos,
            total + max(bottoms.max(), 1e-9) * 0.018,
            f"{total:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    centers = []
    left = 0
    for _facet in TARGET_EXPERIMENT_FACETS:
        right = left + len(TARGET_POLICIES)
        centers.append(float(np.mean(bar_positions[left:right])))
        left = right
    y_top = max(bottoms.max(), 1e-9) * 1.09
    for center, facet in zip(centers, TARGET_EXPERIMENT_FACETS):
        ax.text(center, y_top, EXPERIMENT_DISPLAY[facet], ha="center", va="bottom", fontsize=10, fontweight="bold")

    boundary = (bar_positions[len(TARGET_POLICIES) - 1] + bar_positions[len(TARGET_POLICIES)]) / 2.0
    ax.axvline(boundary, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("J / generated token", fontweight="bold")
    ax.set_xlabel("Policy", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    phase_handles = [Patch(facecolor=PHASE_COLORS[p], edgecolor="black", label=PHASE_DISPLAY[p]) for p in PHASE_STACK_ORDER]
    fig.suptitle("Phase-Stacked Energy Intensity by Policy and Reward Mechanism", y=0.99, fontweight="bold")
    fig.legend(handles=phase_handles, title="Phase", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
