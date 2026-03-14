"""Boundary-aligned PCIe RX throughput at the rollout -> rl_policy transition."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok
from plots.plotting.style import line_paper, savefig_paper


OUTPATH = Path("plots/out/reward_model/phase_transition_pcie_rx.png")
WINDOW_LEFT_S = -5.0
WINDOW_RIGHT_S = 8.0
BIN_WIDTH_S = 0.5
PCIE_TO_GBPS = 1e9

TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
TARGET_EXPERIMENT_FACETS = ("Llama Reward Function", "Llama Reward Model")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama Reward Function": "llama_new_baseline",
    "Llama Reward Model": "llama_rm_gsm8k",
}
LOGICAL_GROUP_PREFIXES_BY_FACET = {
    "Llama Reward Function": ("stage1_llama8b_",),
    "Llama Reward Model": ("llama8b_",),
}
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
EXPERIMENT_COLORS = {
    "Llama Reward Function": "#D04A1C",
    "Llama Reward Model": "#295894",
}
PRE_SHADE = "#ECECEC"
BOUNDARY_COLOR = "#444444"


def _experiment_facet(slurm_job_name: str, logical_run_group: str) -> str:
    slurm_text = str(slurm_job_name).strip().lower()
    logical_text = str(logical_run_group).strip().lower()
    for facet in TARGET_EXPERIMENT_FACETS:
        expected_slurm = TARGET_SLURM_JOB_NAME_BY_FACET[facet]
        logical_prefixes = LOGICAL_GROUP_PREFIXES_BY_FACET[facet]
        if slurm_text == expected_slurm and logical_text.startswith(logical_prefixes):
            return facet
    return "Other"


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")
    required = ["run_id", "policy", "logical_run_group"]
    missing = [col for col in required if col not in run_summary.columns]
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
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()
    runs_df["experiment_facet"] = [
        _experiment_facet(slurm_job_name=slurm_job_name, logical_run_group=logical_group_value)
        for slurm_job_name, logical_group_value in zip(runs_df["slurm_job_name"], runs_df["logical_run_group"])
    ]

    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["experiment_facet"].isin(TARGET_EXPERIMENT_FACETS)
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

    selected = runs_df[non_rollout_knob_mask & target_mask & checkpoint_mask & integrity_mask][
        ["run_id", "policy_norm", "experiment_facet"]
    ].drop_duplicates()
    if selected.empty:
        raise ValueError("No reward-model phase-transition runs selected.")
    return selected.sort_values(["experiment_facet", "policy_norm", "run_id"])


def _load_boundary_samples(selected_runs: pd.DataFrame) -> pd.DataFrame:
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_fact, _ = load_view("phase_fact_view")
    phase_fact = phase_fact[phase_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    phase_fact = apply_analysis_ok(phase_fact)
    phase_fact["phase_name"] = phase_fact["phase_name"].astype(str).str.lower()
    anchors = phase_fact[phase_fact["phase_name"] == "rl_policy"][
        ["run_id", "global_step_canonical", "phase_start_ts"]
    ].copy()
    anchors["phase_start_ts"] = pd.to_numeric(anchors["phase_start_ts"], errors="coerce")
    anchors = anchors.dropna(subset=["phase_start_ts"]).copy()
    anchors["phase_start_ts"] = anchors["phase_start_ts"].astype(np.int64)
    if anchors.empty:
        raise ValueError("No analysis-valid rl_policy phase starts found.")

    periodic, _ = load_view("hardware_periodic")
    needed = [
        "run_id",
        "global_step_canonical",
        "phase_name",
        "ts_monotonic_ns",
        "pcie_rx_bytes_s",
        "device_id",
        "device_kind",
        "source",
        "record_type",
    ]
    missing = [c for c in needed if c not in periodic.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")

    df = periodic[periodic["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    df = df[df["record_type"].astype(str).str.upper() == "PERIODIC"].copy()
    df = df[df["source"].astype(str).str.lower() == "nvml"].copy()
    df = df[df["device_kind"].astype(str).str.lower() == "gpu"].copy()
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(("rollout", "rl_policy"))].copy()

    df["ts_monotonic_ns"] = pd.to_numeric(df["ts_monotonic_ns"], errors="coerce")
    df["pcie_rx_bytes_s"] = pd.to_numeric(df["pcie_rx_bytes_s"], errors="coerce")
    df = df.dropna(subset=["ts_monotonic_ns", "pcie_rx_bytes_s"]).copy()
    df["ts_monotonic_ns"] = df["ts_monotonic_ns"].astype(np.int64)

    df = df.merge(anchors, on=["run_id", "global_step_canonical"], how="inner", validate="many_to_one")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    df["seconds_from_anchor"] = (df["ts_monotonic_ns"] - df["phase_start_ts"]) / 1e9
    df = df[
        (df["seconds_from_anchor"] >= WINDOW_LEFT_S)
        & (df["seconds_from_anchor"] <= WINDOW_RIGHT_S)
    ].copy()
    if df.empty:
        raise ValueError("No periodic PCIe RX samples in the requested boundary window.")

    # Aggregate across GPUs per timestamp so the plot reflects node-level host->device churn.
    per_ts = (
        df.groupby(
            [
                "experiment_facet",
                "policy_norm",
                "run_id",
                "global_step_canonical",
                "ts_monotonic_ns",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            seconds_from_anchor=("seconds_from_anchor", "mean"),
            pcie_rx_bytes_s_sum=("pcie_rx_bytes_s", "sum"),
        )
        .sort_values(["experiment_facet", "policy_norm", "run_id", "global_step_canonical", "ts_monotonic_ns"])
    )
    per_ts["pcie_rx_gbps"] = per_ts["pcie_rx_bytes_s_sum"] / PCIE_TO_GBPS
    per_ts["bin_center_s"] = (
        np.floor((per_ts["seconds_from_anchor"] - WINDOW_LEFT_S) / BIN_WIDTH_S) * BIN_WIDTH_S
        + WINDOW_LEFT_S
        + (BIN_WIDTH_S / 2.0)
    )
    return per_ts


def _aggregate_bins(per_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_facet, policy_norm, bin_center_s), g in per_ts.groupby(
        ["experiment_facet", "policy_norm", "bin_center_s"], dropna=False
    ):
        vals = pd.to_numeric(g["pcie_rx_gbps"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "experiment_facet": experiment_facet,
                "policy_norm": policy_norm,
                "bin_center_s": float(bin_center_s),
                "gbps_mean": float(vals.mean()),
                "gbps_q25": float(np.percentile(vals, 25)),
                "gbps_q75": float(np.percentile(vals, 75)),
                "gbps_p90": float(np.percentile(vals, 90)),
                "n_samples": int(vals.size),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No binned PCIe RX rows available for plotting.")
    return out.sort_values(["experiment_facet", "policy_norm", "bin_center_s"])


def _boundary_summary(per_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_facet, policy_norm), g in per_ts.groupby(["experiment_facet", "policy_norm"], dropna=False):
        pre = g[(g["seconds_from_anchor"] >= -2.0) & (g["seconds_from_anchor"] < 0.0)]["pcie_rx_gbps"]
        post = g[(g["seconds_from_anchor"] >= 0.0) & (g["seconds_from_anchor"] <= 2.0)]["pcie_rx_gbps"]
        peak_window = g[(g["seconds_from_anchor"] >= 0.0) & (g["seconds_from_anchor"] <= 3.0)].copy()
        peak_row = peak_window.loc[peak_window["pcie_rx_gbps"].idxmax()] if not peak_window.empty else None
        rows.append(
            {
                "experiment_facet": experiment_facet,
                "policy_norm": policy_norm,
                "pre_boundary_median_gbps": float(pre.median()) if not pre.empty else np.nan,
                "post_boundary_median_gbps": float(post.median()) if not post.empty else np.nan,
                "post_boundary_peak_gbps": float(peak_row["pcie_rx_gbps"]) if peak_row is not None else np.nan,
                "peak_time_s": float(peak_row["seconds_from_anchor"]) if peak_row is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["experiment_facet", "policy_norm"])


def main() -> None:
    selected_runs = _select_runs()
    per_ts = _load_boundary_samples(selected_runs)
    binned = _aggregate_bins(per_ts)
    summary = _boundary_summary(per_ts)

    print("selected runs:")
    print(selected_runs.to_string(index=False))
    print("\nwindowed timestamp sample counts:")
    print(
        per_ts.groupby(["experiment_facet", "policy_norm"], dropna=False)
        .size()
        .rename("n_timestamp_samples")
        .reset_index()
        .sort_values(["experiment_facet", "policy_norm"])
        .to_string(index=False)
    )
    print("\nrollout -> rl_policy PCIe RX summary (node-summed GB/s):")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(TARGET_POLICIES), figsize=(14.2, 4.9), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    y_max = float(max(binned["gbps_q75"].max(), binned["gbps_mean"].max()))
    y_upper = max(0.25, y_max * 1.12)

    for ax, policy in zip(axes, TARGET_POLICIES):
        sub = binned[binned["policy_norm"] == policy].copy()
        ax.axvspan(WINDOW_LEFT_S, 0.0, color=PRE_SHADE, alpha=0.6, zorder=0)
        ax.axvline(0.0, color=BOUNDARY_COLOR, linestyle="--", linewidth=1.1, zorder=1)

        for facet in TARGET_EXPERIMENT_FACETS:
            fsub = sub[sub["experiment_facet"] == facet].sort_values("bin_center_s")
            if fsub.empty:
                continue
            color = EXPERIMENT_COLORS[facet]
            ax.fill_between(
                fsub["bin_center_s"],
                fsub["gbps_q25"],
                fsub["gbps_q75"],
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                fsub["bin_center_s"],
                fsub["gbps_mean"],
                color=color,
                linewidth=2.4,
                zorder=3,
            )

        ax.set_title(POLICY_DISPLAY[policy], fontweight="bold")
        ax.set_xlim(WINDOW_LEFT_S, WINDOW_RIGHT_S)
        ax.set_ylim(0.0, y_upper)
        ax.set_xlabel("(s) from rl_policy start")
        line_paper(ax)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Node PCIe RX (GB/s)")

    legend_handles = [
        Line2D([0], [0], color=EXPERIMENT_COLORS[facet], linewidth=2.6, label=EXPERIMENT_DISPLAY[facet])
        for facet in TARGET_EXPERIMENT_FACETS
    ]
    legend_handles.append(
        Line2D([0], [0], color=BOUNDARY_COLOR, linestyle="--", linewidth=1.2, label="phase boundary")
    )

    fig.suptitle("PCIe RX at the Rollout to Preparation Boundary", y=0.99, fontweight="bold")
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        title="Experiment",
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.92),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83), w_pad=2.0)

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"\nwrote {saved}")


if __name__ == "__main__":
    main()
