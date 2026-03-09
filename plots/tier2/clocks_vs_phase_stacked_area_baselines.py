"""Clocks vs phase (stacked area) for baseline runs.

Data source: device_timeseries_view (NVML periodic clock samples).
Output: one figure with two subplots (Llama baseline, Qwen baseline).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier2/clocks_vs_phase_stacked_area_baselines.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_MODEL_FACETS = ("Llama", "Qwen")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
BASE_PHASE_ORDER = ["rollout", "training", "rl_policy"]


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
    required = [
        "run_id",
        "policy",
        "model",
        "logical_run_group",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
    ]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    df = run_summary.merge(runs[["run_id", "slurm_job_name"]], on="run_id", how="left", validate="one_to_one").copy()
    df["policy_norm"] = df["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    df["model_facet"] = df["model"].map(_model_facet)
    df["logical_run_group"] = df["logical_run_group"].astype(str).str.lower()
    df["slurm_job_name"] = df["slurm_job_name"].astype(str).str.lower()
    df["is_checkpoint_continuation"] = df["is_checkpoint_continuation"].fillna(False).astype(bool)
    df["join_coverage_rate"] = pd.to_numeric(df["join_coverage_rate"], errors="coerce")
    df["phase_boundary_integrity_rate"] = pd.to_numeric(df["phase_boundary_integrity_rate"], errors="coerce")

    baseline_label_mask = df["logical_run_group"].str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    expected_slurm = df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_mask = df["slurm_job_name"] == expected_slurm
    integrity_mask = (df["join_coverage_rate"] == 1.0) & (df["phase_boundary_integrity_rate"] == 1.0)
    mask = (
        (~df["is_checkpoint_continuation"])
        & df["policy_norm"].isin(TARGET_POLICIES)
        & df["model_facet"].isin(TARGET_MODEL_FACETS)
        & baseline_label_mask
        & slurm_mask
        & integrity_mask
    )
    selected = df.loc[mask, ["run_id", "policy_norm", "model_facet"]].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def main() -> None:
    selected_runs = _select_baseline_runs()
    print("selected baseline runs:")
    print(selected_runs.sort_values(["model_facet", "policy_norm"]).to_string(index=False))

    dts, _ = load_view("device_timeseries_view")
    required = ["run_id", "phase_name", "source", "sm_clock_mhz", "mem_clock_mhz"]
    missing = [c for c in required if c not in dts.columns]
    if missing:
        raise ValueError(f"device_timeseries_view missing required columns: {missing}")

    df = dts[dts["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][required].copy()
    df = df.merge(selected_runs[["run_id", "model_facet"]], on="run_id", how="inner")
    df = df[df["source"].astype(str).str.lower() == "nvml"].copy()
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"] != "validation"].copy()
    df["sm_clock_mhz"] = pd.to_numeric(df["sm_clock_mhz"], errors="coerce")
    df["mem_clock_mhz"] = pd.to_numeric(df["mem_clock_mhz"], errors="coerce")
    df = df.dropna(subset=["sm_clock_mhz", "mem_clock_mhz"]).copy()
    if df.empty:
        raise ValueError("No clock samples after filtering.")

    # Sample means by model+phase for stacked-area visualization.
    phase_means = (
        df.groupby(["model_facet", "phase_name"], dropna=False)[["sm_clock_mhz", "mem_clock_mhz"]]
        .mean()
        .reset_index()
    )

    encountered = [p for p in BASE_PHASE_ORDER if p in set(phase_means["phase_name"])]
    extra = sorted(set(phase_means["phase_name"]) - set(BASE_PHASE_ORDER))
    phase_order = encountered + extra

    print("clock means by model/phase (MHz):")
    print(
        phase_means.sort_values(["model_facet", "phase_name"]).to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for ax, model in zip(axes, TARGET_MODEL_FACETS):
        m = phase_means[phase_means["model_facet"] == model].copy()
        m["phase_name"] = pd.Categorical(m["phase_name"], categories=phase_order, ordered=True)
        m = m.sort_values("phase_name")
        x = np.arange(len(phase_order), dtype=float)
        sm_vals = np.array(
            [float(m.loc[m["phase_name"] == p, "sm_clock_mhz"].iloc[0]) if (m["phase_name"] == p).any() else 0.0 for p in phase_order],
            dtype=float,
        )
        mem_vals = np.array(
            [float(m.loc[m["phase_name"] == p, "mem_clock_mhz"].iloc[0]) if (m["phase_name"] == p).any() else 0.0 for p in phase_order],
            dtype=float,
        )
        ax.stackplot(
            x,
            sm_vals,
            mem_vals,
            labels=["sm_clock_mhz", "mem_clock_mhz"],
            colors=["#4C78A8", "#F58518"],
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(phase_order)
        ax.set_title(f"{model} Baseline")
        ax.set_xlabel("phase_name")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Clock speed (MHz)")
    fig.suptitle("Clocks vs Phase (Stacked Area) - Baseline Runs", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
