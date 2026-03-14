"""Rollout throughput vs sync efficiency (RM gsm8k vs RM rlhf-ff).

Focused view showing only the key signal:
- Corrected Rollout Throughput (tokens/s) vs Rollout Sync Efficiency

This keeps policy faceting and dataset coloring while avoiding less informative
secondary metrics.

Token handling mirrors baseline correction:
- PPO/GRPO denominator = logged rollout total tokens
- ReMax denominator = rollout total + rollout prompt + estimated baseline output tokens
  where estimated baseline output tokens = (gen_max/gen) * rollout_output.
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


OUTPATH = Path("plots/out/reward_model/host_cpu_energy_intensity_vs_rollout_sync_efficiency.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_SLURM_TO_DATASET = {
    "llama_rm_gsm8k": "gsm8k",
    "llama_rm_rlhf": "rlhf-ff",
}
DATASET_ORDER = ("gsm8k", "rlhf-ff")
DATASET_COLOR = {"gsm8k": "#1f77b4", "rlhf-ff": "#d62728"}
DATASET_DISPLAY = {"gsm8k": "gsm8k", "rlhf-ff": "full-hh-rlhf"}
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")

    required = ["run_id", "policy", "logical_run_group"]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")

    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    df = run_summary.merge(runs[["run_id", "slurm_job_name"]], on="run_id", how="left", validate="one_to_one").copy()
    df["policy_norm"] = df["policy"].astype(str).str.lower()
    df["slurm_job_name_lc"] = df["slurm_job_name"].astype(str).str.lower()
    df["dataset_group"] = df["slurm_job_name_lc"].map(TARGET_SLURM_TO_DATASET)
    logical_group = df["logical_run_group"].astype(str).str.lower()

    target_mask = df["policy_norm"].isin(TARGET_POLICIES) & df["dataset_group"].isin(DATASET_ORDER)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    checkpoint_mask = (
        ~df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in df.columns
        else True
    )
    integrity_mask = (
        (pd.to_numeric(df["join_coverage_rate"], errors="coerce") == 1.0)
        & (pd.to_numeric(df["phase_boundary_integrity_rate"], errors="coerce") == 1.0)
        if {"join_coverage_rate", "phase_boundary_integrity_rate"}.issubset(df.columns)
        else True
    )

    selected = df[target_mask & non_rollout_knob_mask & checkpoint_mask & integrity_mask].copy()
    if selected.empty:
        raise ValueError("No target reward-model runs selected for gsm8k vs rlhf-ff.")

    expected = {(d, p) for d in DATASET_ORDER for p in TARGET_POLICIES}
    observed = set(zip(selected["dataset_group"], selected["policy_norm"]))
    if expected - observed:
        raise ValueError(f"Missing dataset-policy combinations: {sorted(expected - observed)}")

    return selected[["run_id", "policy_norm", "dataset_group"]].drop_duplicates()


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


def _load_corrected_tokens_and_sync(selected_runs: pd.DataFrame) -> pd.DataFrame:
    step, _ = load_view("step_fact_view")
    needed = [
        "run_id",
        "global_step_canonical",
        "step_rollout_total_tokens",
        "step_rollout_output_tokens",
        "sync_efficiency",
    ]
    missing = [c for c in needed if c not in step.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    out = step[step["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    before = out.copy()
    out = apply_analysis_ok(out)
    print(f"step filtering={explain_filtering(before, out)}")

    for col in ["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens", "sync_efficiency"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens", "sync_efficiency"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out = out[out["step_rollout_total_tokens"] > 0].copy()
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

    out["corrected_rollout_total_tokens"] = out["step_rollout_total_tokens"]
    remax_mask = out["policy_norm"] == "remax"
    out.loc[remax_mask, "corrected_rollout_total_tokens"] = (
        out.loc[remax_mask, "step_rollout_total_tokens"]
        + out.loc[remax_mask, "step_rollout_prompt_tokens"]
        + out.loc[remax_mask, "step_rollout_output_tokens"] * out.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )
    out = out[out["corrected_rollout_total_tokens"] > 0].copy()

    return out[["run_id", "global_step_canonical", "sync_efficiency", "corrected_rollout_total_tokens", "policy_norm", "dataset_group"]]


def _load_rollout_rapl_energy(selected_runs: pd.DataFrame) -> pd.DataFrame:
    phase, _ = load_view("phase_fact_view")
    needed = ["run_id", "global_step_canonical", "phase_name", "cpu_dram_energy_j", "phase_time_s"]
    missing = [c for c in needed if c not in phase.columns]
    if missing:
        raise ValueError(f"phase_fact_view missing required columns: {missing}")

    out = phase[phase["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    before = out.copy()
    out = apply_analysis_ok(out)
    print(f"phase filtering={explain_filtering(before, out)}")

    out["phase_name"] = out["phase_name"].astype(str).str.lower()
    out = out[out["phase_name"] == "rollout"].copy()

    out["global_step_canonical"] = pd.to_numeric(out["global_step_canonical"], errors="coerce")
    out["cpu_dram_energy_j"] = pd.to_numeric(out["cpu_dram_energy_j"], errors="coerce")
    out["phase_time_s"] = pd.to_numeric(out["phase_time_s"], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "cpu_dram_energy_j", "phase_time_s"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out = out[(out["cpu_dram_energy_j"] >= 0) & (out["phase_time_s"] > 0)].copy()
    out["phase_domain_energy_delta_uJ"] = out["cpu_dram_energy_j"] * 1e6

    return out[["run_id", "global_step_canonical", "phase_domain_energy_delta_uJ", "phase_time_s"]]


def main() -> None:
    selected_runs = _select_runs()
    print("selected runs:")
    print(selected_runs.sort_values(["dataset_group", "policy_norm", "run_id"]).to_string(index=False))

    step = _load_corrected_tokens_and_sync(selected_runs)
    phase = _load_rollout_rapl_energy(selected_runs)

    plot_df = phase.merge(
        step,
        on=["run_id", "global_step_canonical"],
        how="inner",
        validate="one_to_one",
    )
    plot_df["throughput_corrected_tokens_s"] = plot_df["corrected_rollout_total_tokens"] / plot_df["phase_time_s"]
    plot_df = plot_df[np.isfinite(plot_df["throughput_corrected_tokens_s"])].copy()
    plot_df = plot_df[np.isfinite(plot_df["sync_efficiency"])].copy()
    plot_df = plot_df[(plot_df["sync_efficiency"] > 0) & (plot_df["sync_efficiency"] <= 1.0)].copy()
    if plot_df.empty:
        raise ValueError("No rows available after joins and quality filters.")

    run_level = (
        plot_df.groupby(["run_id", "dataset_group", "policy_norm"], dropna=False)
        .agg(
            rollout_sync_efficiency_mean=("sync_efficiency", "mean"),
            throughput_corrected_tokens_s_mean=("throughput_corrected_tokens_s", "mean"),
        )
        .reset_index()
    )

    print("\nrollout-phase rows used by dataset:")
    print(plot_df.groupby("dataset_group", dropna=False).size().rename("n_rows").reset_index().to_string(index=False))
    print("\nrun-level aggregates:")
    print(
        run_level.sort_values(["dataset_group", "policy_norm", "run_id"])[
            [
                "run_id",
                "dataset_group",
                "policy_norm",
                "rollout_sync_efficiency_mean",
                "throughput_corrected_tokens_s_mean",
            ]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(1, len(TARGET_POLICIES), figsize=(18.0, 5.8), sharey=True)
    if len(TARGET_POLICIES) == 1:
        axes = [axes]

    for ax, policy in zip(axes, TARGET_POLICIES):
        psub = plot_df[plot_df["policy_norm"] == policy].copy()
        for dataset in DATASET_ORDER:
            sub = psub[psub["dataset_group"] == dataset].copy()
            if sub.empty:
                continue

            ax.scatter(
                sub["sync_efficiency"],
                sub["throughput_corrected_tokens_s"],
                s=22,
                alpha=0.4,
                color=DATASET_COLOR[dataset],
                edgecolors="none",
            )

            x = sub["sync_efficiency"].to_numpy(dtype=float)
            y_thr = sub["throughput_corrected_tokens_s"].to_numpy(dtype=float)

            finite_thr = np.isfinite(x) & np.isfinite(y_thr)
            xt = x[finite_thr]
            yt = y_thr[finite_thr]
            if xt.size >= 2 and np.unique(xt).size >= 2:
                slope_t, intercept_t = np.polyfit(xt, yt, 1)
                xline_t = np.linspace(float(np.nanmin(xt)), float(np.nanmax(xt)), 120)
                yline_t = slope_t * xline_t + intercept_t
                ax.plot(
                    xline_t,
                    yline_t,
                    color=DATASET_COLOR[dataset],
                    linewidth=2.5,
                    alpha=0.95,
                )
                label_x_idx_t = int(0.82 * (len(xline_t) - 1)) if dataset == "rlhf-ff" else int(0.18 * (len(xline_t) - 1))
                ax.text(
                    float(xline_t[label_x_idx_t]),
                    float(yline_t[label_x_idx_t]),
                    f"slope={slope_t:.1f}",
                    color="black",
                    fontsize=10,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "#888888"},
                )

        ax.set_title(POLICY_DISPLAY.get(policy, policy.upper()), fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlabel("Rollout Sync Efficiency")

    axes[0].set_ylabel("Corrected Rollout Throughput (tokens/s)")
    legend_handles = [
        Patch(facecolor=DATASET_COLOR["rlhf-ff"], edgecolor="none", label="full-hh-rlhf"),
        Patch(facecolor=DATASET_COLOR["gsm8k"], edgecolor="none", label="gsm8k"),
    ]
    fig.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Corrected Rollout Throughput vs. Rollout Sync Efficiency", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
