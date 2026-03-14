"""Stacked rollout -> rl_policy boundary plots for VRAM occupancy and PCIe RX velocity."""

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


OUTPATH = Path("plots/out/reward_model/vram_vs_pcie_rx.png")
WINDOW_LEFT_S = -2.0
WINDOW_RIGHT_S = 5.0
BIN_WIDTH_S = 0.25

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
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
EXPERIMENT_LINESTYLE = {
    "Llama Reward Function": "-",
    "Llama Reward Model": "--",
}
EXPERIMENT_ALPHA = {
    "Llama Reward Function": 0.5,
    "Llama Reward Model": 1.0,
}
MEM_COLOR = "#295894"
PCIE_COLOR = "#D04A1C"
BOUNDARY_COLOR = "#111111"


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
        raise ValueError("No reward-model transition occupancy runs selected.")
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
        "mem_used_B",
        "pcie_rx_bytes_s",
        "device_kind",
        "source",
        "record_type",
    ]
    missing = [col for col in needed if col not in periodic.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")

    df = periodic[periodic["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    df = df[df["record_type"].astype(str).str.upper() == "PERIODIC"].copy()
    df = df[df["source"].astype(str).str.lower() == "nvml"].copy()
    df = df[df["device_kind"].astype(str).str.lower() == "gpu"].copy()
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(("rollout", "rl_policy"))].copy()

    df["ts_monotonic_ns"] = pd.to_numeric(df["ts_monotonic_ns"], errors="coerce")
    df["mem_used_B"] = pd.to_numeric(df["mem_used_B"], errors="coerce")
    df["pcie_rx_bytes_s"] = pd.to_numeric(df["pcie_rx_bytes_s"], errors="coerce")
    df = df.dropna(subset=["ts_monotonic_ns", "mem_used_B", "pcie_rx_bytes_s"]).copy()
    df["ts_monotonic_ns"] = df["ts_monotonic_ns"].astype(np.int64)

    df = df.merge(anchors, on=["run_id", "global_step_canonical"], how="inner", validate="many_to_one")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    df["seconds_from_anchor"] = (df["ts_monotonic_ns"] - df["phase_start_ts"]) / 1e9
    df = df[(df["seconds_from_anchor"] >= WINDOW_LEFT_S) & (df["seconds_from_anchor"] <= WINDOW_RIGHT_S)].copy()
    if df.empty:
        raise ValueError("No periodic mem_used_B / pcie_rx_bytes_s samples in the requested boundary window.")

    # Sum across GPUs per timestamp so both metrics reflect node-level occupancy and host->device traffic.
    per_ts = (
        df.groupby(
            ["experiment_facet", "run_id", "global_step_canonical", "ts_monotonic_ns"],
            dropna=False,
            as_index=False,
        )
        .agg(
            seconds_from_anchor=("seconds_from_anchor", "mean"),
            mem_used_B=("mem_used_B", "sum"),
            pcie_rx_bytes_s=("pcie_rx_bytes_s", "sum"),
        )
        .sort_values(["experiment_facet", "run_id", "global_step_canonical", "ts_monotonic_ns"])
    )
    per_ts["bin_center_s"] = (
        np.floor((per_ts["seconds_from_anchor"] - WINDOW_LEFT_S) / BIN_WIDTH_S) * BIN_WIDTH_S
        + WINDOW_LEFT_S
        + (BIN_WIDTH_S / 2.0)
    )
    return per_ts


def _aggregate_bins(per_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_facet, bin_center_s), g in per_ts.groupby(["experiment_facet", "bin_center_s"], dropna=False):
        rows.append(
            {
                "experiment_facet": experiment_facet,
                "bin_center_s": float(bin_center_s),
                "mem_used_B": float(pd.to_numeric(g["mem_used_B"], errors="coerce").mean()),
                "pcie_rx_bytes_s": float(pd.to_numeric(g["pcie_rx_bytes_s"], errors="coerce").mean()),
                "n_samples": int(len(g)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No binned occupancy rows available for plotting.")
    return out.sort_values(["experiment_facet", "bin_center_s"])


def main() -> None:
    selected_runs = _select_runs()
    per_ts = _load_boundary_samples(selected_runs)
    binned = _aggregate_bins(per_ts)

    print("selected runs:")
    print(selected_runs.to_string(index=False))
    print("\nwindowed timestamp sample counts by reward mechanism:")
    print(
        per_ts.groupby(["experiment_facet"], dropna=False)
        .size()
        .rename("n_timestamp_samples")
        .reset_index()
        .sort_values(["experiment_facet"])
        .to_string(index=False)
    )
    print("\nbinned stacked-line summary:")
    print(binned.to_string(index=False))

    fig, (ax_mem, ax_pcie) = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True)

    for facet in TARGET_EXPERIMENT_FACETS:
        fsub = binned[binned["experiment_facet"] == facet].sort_values("bin_center_s")
        if fsub.empty:
            continue
        linestyle = EXPERIMENT_LINESTYLE[facet]
        alpha = EXPERIMENT_ALPHA[facet]
        ax_mem.plot(
            fsub["bin_center_s"],
            fsub["mem_used_B"],
            color=MEM_COLOR,
            linestyle=linestyle,
            linewidth=2.4,
            alpha=alpha,
        )
        ax_pcie.plot(
            fsub["bin_center_s"],
            fsub["pcie_rx_bytes_s"],
            color=PCIE_COLOR,
            linestyle=linestyle,
            linewidth=2.4,
            alpha=alpha,
        )

    for ax in (ax_mem, ax_pcie):
        ax.axvline(0.0, color=BOUNDARY_COLOR, linestyle="--", linewidth=1.2, alpha=0.9)
        ax.set_xlim(WINDOW_LEFT_S, WINDOW_RIGHT_S)
        line_paper(ax)
        ax.set_axisbelow(True)

    ax_mem.set_ylabel("Used VRAM (B)")
    ax_pcie.set_ylabel("PCIe RX Throughput (B/s)")
    ax_pcie.set_xlabel("Time (s) from iteration start")

    legend_handles = [
        Line2D([0], [0], color=MEM_COLOR, linewidth=2.6, label="Used VRAM"),
        Line2D([0], [0], color=PCIE_COLOR, linewidth=2.6, label="PCIe RX Throughput"),
    ] + [
        Line2D([0], [0], color="#333333", linestyle=EXPERIMENT_LINESTYLE[facet], linewidth=2.6, label=EXPERIMENT_DISPLAY[facet])
        for facet in TARGET_EXPERIMENT_FACETS
    ]
    fig.suptitle("VRAM vs PCIe RX", fontweight="bold", y=0.98)
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"\nwrote {saved}")


if __name__ == "__main__":
    main()
