"""Stacked phase J/output-token aggregated across policies, grouped by dataset."""

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
OUTPATH = Path("plots/out/task/j_per_output_token.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_DATASETS = ("gsm8k", "rlhf-ff")
DATASET_DISPLAY = {
    "gsm8k": "gsm8k",
    "rlhf-ff": "full-hh-rlhf",
}
PHASE_STACK_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
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


def _select_runs() -> pd.DataFrame:
    runs_df, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "dataset"]
    missing = [col for col in required if col not in runs_df.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required selection columns: {missing}")

    runs_df = runs_df.copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["dataset_group"] = runs_df["dataset"].astype(str).str.lower()
    selected = runs_df[
        runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["dataset_group"].isin(TARGET_DATASETS)
    ].copy()
    if "is_checkpoint_continuation" in selected.columns:
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
    if selected.empty:
        raise ValueError("No task runs selected.")
    return selected[["run_id", "policy_norm", "dataset_group"]].drop_duplicates()


def _load_phase_energy(selected_run_ids: list[str]) -> pd.DataFrame:
    phase_df, _ = load_view("phase_fact_view")
    required_cols = ["run_id", "phase_name", "total_energy_j"]
    optional = [
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
        raise ValueError(f"phase_fact_view missing required columns: {missing_required}")

    use_cols = [col for col in required_cols + optional if col in phase_df.columns]
    phase_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)][use_cols].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"phase filtering={explain_filtering(before, phase_df)}")
    if not INCLUDE_VALIDATION:
        phase_df = phase_df[phase_df["phase_name"].astype(str).str.lower() != "validation"].copy()
    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
    phase_df["total_energy_j"] = pd.to_numeric(phase_df["total_energy_j"], errors="coerce")
    return phase_df[phase_df["phase_bucket"].isin(PHASE_STACK_ORDER)].dropna(subset=["total_energy_j"]).copy()


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
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    out = step_df[step_df["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    out["global_step_canonical"] = pd.to_numeric(out["global_step_canonical"], errors="coerce")
    out["step_rollout_total_tokens"] = pd.to_numeric(out["step_rollout_total_tokens"], errors="coerce")
    out["step_rollout_output_tokens"] = pd.to_numeric(out["step_rollout_output_tokens"], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out["step_rollout_prompt_tokens"] = out["step_rollout_total_tokens"] - out["step_rollout_output_tokens"]
    out = out.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    remax_ids = selected_runs[selected_runs["policy_norm"] == "remax"]["run_id"].astype(str).tolist()
    out = out.merge(_load_remax_gen_ratio(remax_ids), on=["run_id", "global_step_canonical"], how="left")
    out["genmax_over_gen"] = pd.to_numeric(out["genmax_over_gen"], errors="coerce")
    run_medians = out.groupby("run_id")["genmax_over_gen"].median().to_dict()
    out["genmax_over_gen"] = out.apply(
        lambda row: run_medians.get(row["run_id"], np.nan) if pd.isna(row["genmax_over_gen"]) else row["genmax_over_gen"],
        axis=1,
    )

    out["corrected_step_output_tokens"] = out["step_rollout_total_tokens"]
    remax_mask = out["policy_norm"] == "remax"
    out.loc[remax_mask, "corrected_step_output_tokens"] = (
        out.loc[remax_mask, "step_rollout_total_tokens"]
        + out.loc[remax_mask, "step_rollout_prompt_tokens"]
        + out.loc[remax_mask, "step_rollout_output_tokens"] * out.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )
    out.loc[~remax_mask, "corrected_step_output_tokens"] = out.loc[~remax_mask, "step_rollout_output_tokens"]
    out = out[out["corrected_step_output_tokens"] > 0].copy()
    return (
        out.groupby(["run_id", "dataset_group", "policy_norm"], dropna=False)["corrected_step_output_tokens"]
        .sum()
        .rename("corrected_output_tokens")
        .reset_index()
    )


def main() -> None:
    selected_runs = _select_runs()
    phase_energy = _load_phase_energy(selected_runs["run_id"].astype(str).tolist())
    phase_energy = phase_energy.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    phase_energy_run = (
        phase_energy.groupby(["run_id", "dataset_group", "policy_norm", "phase_bucket"], dropna=False)["total_energy_j"]
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

    summary = (
        merged.groupby(["dataset_group", "phase_bucket"], dropna=False)[["total_energy_j", "corrected_output_tokens"]]
        .sum()
        .reset_index()
    )
    summary = summary[summary["corrected_output_tokens"] > 0].copy()
    summary["j_per_output_token"] = summary["total_energy_j"] / summary["corrected_output_tokens"]

    totals = (
        summary.groupby("dataset_group", dropna=False)["j_per_output_token"]
        .sum()
        .rename("total_j_per_output_token")
        .reset_index()
    )
    print("aggregated J/output-token totals by dataset:")
    print(totals.sort_values("dataset_group").to_string(index=False))

    bar_positions = list(range(len(TARGET_DATASETS)))
    bar_labels = [DATASET_DISPLAY[dataset] for dataset in TARGET_DATASETS]

    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    bar_width = 0.72
    bottoms = np.zeros(len(bar_positions), dtype=float)
    for phase in PHASE_STACK_ORDER:
        vals = []
        for dataset in TARGET_DATASETS:
            row = summary[
                (summary["dataset_group"] == dataset)
                & (summary["phase_bucket"] == phase)
            ]
            vals.append(float(row["j_per_output_token"].iloc[0]) if not row.empty else 0.0)
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
        ax.text(xpos, total + max(bottoms.max(), 1e-9) * 0.018, f"{total:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Mean J / output token")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    phase_handles = [Patch(facecolor=PHASE_COLORS[p], edgecolor="black", label=PHASE_DISPLAY[p]) for p in PHASE_STACK_ORDER]
    fig.legend(handles=phase_handles, title="Phase", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("J per Output Token by Dataset (Policies Aggregated)", y=0.99, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
