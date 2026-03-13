"""PCIe RX throughput vs rollout-only generation throughput for baseline runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/baselines/non_results/pcie_rx_vs_rollout_throughput.png")
INCLUDE_VALIDATION = False
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
MODEL_COLORS = {
    "Llama": "#C73E1D",
    "Qwen": "#1D4E89",
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
    required = ["run_id", "policy", "model", "logical_run_group"]
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

    selected = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & slurm_job_mask & checkpoint_mask
    ].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected[["run_id", "model_facet", "policy_norm"]].drop_duplicates()


def _build_plot_df() -> pd.DataFrame:
    selected_runs = _select_baseline_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_fact, _ = load_view("phase_fact_view")
    required_phase = ["run_id", "phase_name", "global_step_canonical", "phase_time_s", "rollout_output_tokens_total"]
    missing_phase = [c for c in required_phase if c not in phase_fact.columns]
    if missing_phase:
        raise ValueError(f"phase_fact_view missing required columns: {missing_phase}")
    eligible_steps = phase_fact[phase_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    eligible_steps = apply_analysis_ok(eligible_steps)
    if not INCLUDE_VALIDATION and "is_validation_step" in eligible_steps.columns:
        eligible_steps = eligible_steps[~eligible_steps["is_validation_step"].fillna(False)].copy()
    eligible_steps["phase_name"] = eligible_steps["phase_name"].astype(str).str.lower()
    eligible_steps = eligible_steps[eligible_steps["phase_name"] == "rollout"].copy()
    eligible_steps["phase_time_s"] = pd.to_numeric(eligible_steps["phase_time_s"], errors="coerce")
    eligible_steps["rollout_output_tokens_total"] = pd.to_numeric(
        eligible_steps["rollout_output_tokens_total"], errors="coerce"
    )
    eligible_steps = eligible_steps.dropna(
        subset=["global_step_canonical", "phase_time_s", "rollout_output_tokens_total"]
    ).copy()
    eligible_steps = eligible_steps[eligible_steps["phase_time_s"] > 0].copy()
    eligible_steps["rollout_output_tokens_per_s"] = (
        eligible_steps["rollout_output_tokens_total"] / eligible_steps["phase_time_s"]
    )
    eligible_steps = eligible_steps[
        ["run_id", "global_step_canonical", "rollout_output_tokens_per_s"]
    ].drop_duplicates()

    periodic, _ = load_view("hardware_periodic")
    needed = [
        "run_id",
        "phase_name",
        "global_step_canonical",
        "ts_monotonic_ns",
        "pcie_rx_bytes_s",
        "record_type",
        "source",
    ]
    missing = [c for c in needed if c not in periodic.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")

    df = periodic[periodic["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    df = df[df["record_type"].astype(str).str.upper() == "PERIODIC"].copy()
    df = df[df["source"].astype(str).str.lower() == "nvml"].copy()
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"] == "rollout"].copy()
    df["pcie_rx_bytes_s"] = pd.to_numeric(df["pcie_rx_bytes_s"], errors="coerce")
    df["ts_monotonic_ns"] = pd.to_numeric(df["ts_monotonic_ns"], errors="coerce")
    df = df.dropna(subset=["pcie_rx_bytes_s", "ts_monotonic_ns", "global_step_canonical"]).copy()

    per_ts = (
        df.groupby(["run_id", "global_step_canonical", "ts_monotonic_ns"], as_index=False)["pcie_rx_bytes_s"]
        .sum()
        .rename(columns={"pcie_rx_bytes_s": "pcie_rx_bytes_s_sum"})
    )
    per_ts = per_ts.merge(eligible_steps, on=["run_id", "global_step_canonical"], how="inner")
    per_ts = per_ts.merge(selected_runs[["run_id", "model_facet"]], on="run_id", how="inner")
    per_ts = per_ts.dropna(subset=["rollout_output_tokens_per_s", "pcie_rx_bytes_s_sum"]).copy()
    return per_ts


def _plot_density(ax: plt.Axes, values: np.ndarray, color: str, orientation: str) -> None:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return
    counts, edges = np.histogram(values, bins=28, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if orientation == "x":
        ax.fill_between(centers, counts, color=color, alpha=0.20)
        ax.plot(centers, counts, color=color, linewidth=1.5)
    else:
        ax.fill_betweenx(centers, counts, color=color, alpha=0.20)
        ax.plot(counts, centers, color=color, linewidth=1.5)


def main() -> None:
    plot_df = _build_plot_df()
    if plot_df.empty:
        raise ValueError("No rollout periodic PCIe samples available after filtering.")

    print("sample counts by model:")
    print(
        plot_df.groupby("model_facet", dropna=False)
        .size()
        .rename("n_periodic_samples")
        .reset_index()
        .to_string(index=False)
    )

    fig = plt.figure(figsize=(9.6, 7.2))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[4.4, 1.2],
        height_ratios=[1.2, 4.4],
        wspace=0.05,
        hspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    for model in TARGET_MODEL_FACETS:
        sub = plot_df[plot_df["model_facet"] == model].copy()
        if sub.empty:
            continue
        x = sub["pcie_rx_bytes_s_sum"].to_numpy(dtype=float)
        y = sub["rollout_output_tokens_per_s"].to_numpy(dtype=float)
        color = MODEL_COLORS[model]
        ax_main.scatter(
            x,
            y,
            s=12,
            alpha=0.28,
            color=color,
            edgecolors="none",
            label=MODEL_DISPLAY[model],
        )
        _plot_density(ax_top, x, color, orientation="x")
        _plot_density(ax_right, y, color, orientation="y")

    ax_main.grid(alpha=0.22)
    ax_main.set_axisbelow(True)
    ax_main.set_xlabel("pcie_rx_bytes_s (Host to Device)")
    ax_main.set_ylabel("Rollout output tokens / rollout phase_time_s (tokens/s)")

    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)

    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[model], markersize=7, label=MODEL_DISPLAY[model])
        for model in TARGET_MODEL_FACETS
    ]
    fig.suptitle("PCIe RX Throughput vs. Token Generation Velocity", y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.955,
        "Rollout-phase periodic NVML samples with rollout-only throughput from retained rollout phase records",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))

    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.10, top=0.89, wspace=0.05, hspace=0.05)
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
