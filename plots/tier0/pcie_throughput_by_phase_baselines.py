"""Baseline PCIe throughput by phase from periodic samples.

Uses periodic (non-boundary) records with per-sample phase alignment and computes:
1) Per run+timestamp: sum PCIe tx/rx across all GPUs/ranks.
2) Per run+phase: time-weighted mean and p95 of summed tx/rx using ts_monotonic_ns deltas.
3) Plot: grouped bars by phase_name for mean tx_sum_bytes_s and mean rx_sum_bytes_s.
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
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/figures/tier0/pcie_throughput_by_phase_baselines.png")
INCLUDE_VALIDATION = False

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
PHASE_ORDER = ["rollout", "training", "rl_policy"]


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = w.sum()
    if total <= 0:
        return float("nan")
    cdf = np.cumsum(w) / total
    idx = np.searchsorted(cdf, quantile, side="left")
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def _time_weights_from_timestamps(ts_ns: np.ndarray) -> np.ndarray:
    n = len(ts_ns)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    deltas = np.diff(ts_ns.astype(np.int64))
    deltas = np.where(deltas > 0, deltas, np.nan)
    finite = deltas[np.isfinite(deltas)]
    fallback = float(np.median(finite)) if finite.size > 0 else 1.0
    deltas = np.where(np.isfinite(deltas), deltas, fallback)
    weights = np.append(deltas, deltas[-1]).astype(float)
    return np.where(weights > 0, weights, fallback)


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
    return selected


def main() -> None:
    selected_runs = _select_baseline_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    selected_meta = selected_runs[["run_id", "model_facet", "policy_norm"]].drop_duplicates()

    step_fact, _ = load_view("step_fact_view")
    eligible_steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    eligible_steps = apply_analysis_ok(eligible_steps)
    if not INCLUDE_VALIDATION and "is_validation_step" in eligible_steps.columns:
        eligible_steps = eligible_steps[~eligible_steps["is_validation_step"].fillna(False)].copy()
    eligible_steps = eligible_steps[["run_id", "global_step_canonical"]].drop_duplicates()

    periodic, _ = load_view("hardware_periodic")
    needed = [
        "run_id",
        "node",
        "rank",
        "gpu_index",
        "phase_name",
        "global_step_canonical",
        "ts_monotonic_ns",
        "pcie_tx_bytes_s",
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
    df = df.merge(eligible_steps, on=["run_id", "global_step_canonical"], how="inner")
    if not INCLUDE_VALIDATION:
        df = df[df["phase_name"].astype(str).str.lower() != "validation"].copy()
    df = df[df["phase_name"].astype(str).str.lower().isin(PHASE_ORDER)].copy()
    df["pcie_tx_bytes_s"] = pd.to_numeric(df["pcie_tx_bytes_s"], errors="coerce")
    df["pcie_rx_bytes_s"] = pd.to_numeric(df["pcie_rx_bytes_s"], errors="coerce")
    df = df.dropna(subset=["pcie_tx_bytes_s", "pcie_rx_bytes_s", "ts_monotonic_ns"]).copy()

    per_ts = (
        df.groupby(["run_id", "phase_name", "ts_monotonic_ns"], as_index=False)[["pcie_tx_bytes_s", "pcie_rx_bytes_s"]]
        .sum()
        .rename(columns={"pcie_tx_bytes_s": "tx_sum_bytes_s", "pcie_rx_bytes_s": "rx_sum_bytes_s"})
    )
    per_ts = per_ts.merge(selected_meta, on="run_id", how="inner")

    run_phase_rows = []
    for (run_id, model_facet, policy_norm, phase_name), g in per_ts.groupby(
        ["run_id", "model_facet", "policy_norm", "phase_name"], dropna=False
    ):
        gg = g.sort_values("ts_monotonic_ns")
        ts = gg["ts_monotonic_ns"].to_numpy(dtype=np.int64)
        tx = gg["tx_sum_bytes_s"].to_numpy(dtype=float)
        rx = gg["rx_sum_bytes_s"].to_numpy(dtype=float)
        w = _time_weights_from_timestamps(ts)
        wsum = float(w.sum()) if w.size else 0.0
        tx_mean = float(np.average(tx, weights=w)) if wsum > 0 else float("nan")
        rx_mean = float(np.average(rx, weights=w)) if wsum > 0 else float("nan")
        tx_p95 = _weighted_quantile(tx, w, 0.95)
        rx_p95 = _weighted_quantile(rx, w, 0.95)
        run_phase_rows.append(
            {
                "run_id": run_id,
                "model_facet": model_facet,
                "policy_norm": policy_norm,
                "phase_name": phase_name,
                "tx_mean_bytes_s": tx_mean,
                "rx_mean_bytes_s": rx_mean,
                "tx_p95_bytes_s": tx_p95,
                "rx_p95_bytes_s": rx_p95,
                "n_timestamps": len(gg),
            }
        )
    run_phase = pd.DataFrame(run_phase_rows)
    if run_phase.empty:
        raise ValueError("No run-phase rows after periodic filtering.")

    phase_summary = (
        run_phase.groupby(["model_facet", "phase_name"], as_index=False)[
            ["tx_mean_bytes_s", "rx_mean_bytes_s", "tx_p95_bytes_s", "rx_p95_bytes_s"]
        ]
        .mean()
    )
    phase_summary["phase_name"] = phase_summary["phase_name"].astype(str).str.lower()
    phase_summary["phase_name"] = pd.Categorical(phase_summary["phase_name"], categories=PHASE_ORDER, ordered=True)
    phase_summary = phase_summary.sort_values(["model_facet", "phase_name"])

    print("selected run_ids:")
    print(selected_runs[["run_id", "model_facet", "policy_norm", "slurm_job_name"]].sort_values(["model_facet", "policy_norm"]).to_string(index=False))
    print("aggregation note: time-weighted using adjacent ts_monotonic_ns deltas.")
    print("run-phase stats (mean + p95):")
    print(run_phase.sort_values(["phase_name", "run_id"]).to_string(index=False))
    print("phase summary for plotting (means across runs, split by model):")
    print(phase_summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    facet_axes = dict(zip(TARGET_MODEL_FACETS, axes))
    wbar = 0.35
    for facet in TARGET_MODEL_FACETS:
        ax = facet_axes[facet]
        sub = phase_summary[phase_summary["model_facet"] == facet].copy()
        sub["phase_name"] = pd.Categorical(sub["phase_name"], categories=PHASE_ORDER, ordered=True)
        sub = sub.sort_values("phase_name")
        x = np.arange(len(PHASE_ORDER), dtype=float)
        tx_vals = [float(sub.loc[sub["phase_name"] == p, "tx_mean_bytes_s"].iloc[0]) if (sub["phase_name"] == p).any() else np.nan for p in PHASE_ORDER]
        rx_vals = [float(sub.loc[sub["phase_name"] == p, "rx_mean_bytes_s"].iloc[0]) if (sub["phase_name"] == p).any() else np.nan for p in PHASE_ORDER]
        ax.bar(x - wbar / 2, tx_vals, width=wbar, color="#1f77b4", edgecolor="black", linewidth=0.7)
        ax.bar(x + wbar / 2, rx_vals, width=wbar, color="#ff7f0e", edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(PHASE_ORDER)
        ax.set_xlabel("phase_name")
        ax.set_title(facet)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("bytes/sec")
    fig.suptitle("Baseline periodic PCIe throughput by phase, split by model\n(time-weighted run-level means, then averaged across runs)", y=0.99)
    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="mean tx_sum_bytes_s (device→host)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="mean rx_sum_bytes_s (host→device)"),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
