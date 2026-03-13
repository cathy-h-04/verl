"""Per-phase PCIe total data moved per step, split RX/TX, policy-matched baseline comparison."""

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


OUTPATH = Path("plots/out/baselines/phase_total_pcie.png")
INCLUDE_VALIDATION = False
BYTES_TO_GB = 1e9

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
PHASE_ORDER = ["rollout", "rl_policy", "training"]
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
TX_COLOR = "#1D4E89"
RX_COLOR = "#C73E1D"
MODEL_EDGE = {
    "Llama": "#222222",
    "Qwen": "#7F7F7F",
}
MODEL_ALPHA = {
    "Llama": 0.95,
    "Qwen": 0.35,
}


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


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
    weights = np.append(deltas, deltas[-1]).astype(float) / 1e9
    return np.where(weights > 0, weights, fallback / 1e9)


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
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(PHASE_ORDER)].copy()
    df["pcie_tx_bytes_s"] = pd.to_numeric(df["pcie_tx_bytes_s"], errors="coerce")
    df["pcie_rx_bytes_s"] = pd.to_numeric(df["pcie_rx_bytes_s"], errors="coerce")
    df["ts_monotonic_ns"] = pd.to_numeric(df["ts_monotonic_ns"], errors="coerce")
    df = df.dropna(subset=["pcie_tx_bytes_s", "pcie_rx_bytes_s", "ts_monotonic_ns"]).copy()

    per_ts = (
        df.groupby(["run_id", "global_step_canonical", "phase_name", "ts_monotonic_ns"], as_index=False)[
            ["pcie_tx_bytes_s", "pcie_rx_bytes_s"]
        ]
        .sum()
        .rename(columns={"pcie_tx_bytes_s": "tx_sum_bytes_s", "pcie_rx_bytes_s": "rx_sum_bytes_s"})
    )
    per_ts = per_ts.merge(selected_meta, on="run_id", how="inner")

    step_phase_rows = []
    for (run_id, step, phase_name, model_facet, policy_norm), g in per_ts.groupby(
        ["run_id", "global_step_canonical", "phase_name", "model_facet", "policy_norm"], dropna=False
    ):
        gg = g.sort_values("ts_monotonic_ns")
        ts = gg["ts_monotonic_ns"].to_numpy(dtype=np.int64)
        tx = gg["tx_sum_bytes_s"].to_numpy(dtype=float)
        rx = gg["rx_sum_bytes_s"].to_numpy(dtype=float)
        w_s = _time_weights_from_timestamps(ts)
        step_phase_rows.append(
            {
                "run_id": run_id,
                "global_step_canonical": int(step),
                "phase_name": phase_name,
                "model_facet": model_facet,
                "policy_norm": policy_norm,
                "tx_total_gb": float(np.sum(tx * w_s)) / BYTES_TO_GB,
                "rx_total_gb": float(np.sum(rx * w_s)) / BYTES_TO_GB,
                "n_samples": len(gg),
            }
        )
    step_phase = pd.DataFrame(step_phase_rows)
    if step_phase.empty:
        raise ValueError("No step-phase PCIe rows after periodic filtering.")

    summary = (
        step_phase.groupby(["policy_norm", "model_facet", "phase_name"], as_index=False)[["tx_total_gb", "rx_total_gb"]]
        .mean()
    )
    summary["policy_norm"] = pd.Categorical(summary["policy_norm"], categories=POLICY_ORDER, ordered=True)
    summary["phase_name"] = pd.Categorical(summary["phase_name"], categories=PHASE_ORDER, ordered=True)
    summary = summary.sort_values(["policy_norm", "phase_name", "model_facet"])

    print("selected run_ids:")
    print(
        selected_runs[["run_id", "model_facet", "policy_norm", "slurm_job_name"]]
        .sort_values(["policy_norm", "model_facet"])
        .to_string(index=False)
    )
    print("step-phase totals used for aggregation:")
    print(step_phase.sort_values(["policy_norm", "phase_name", "run_id", "global_step_canonical"]).to_string(index=False))
    print("plot summary (mean GB moved per step):")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), sharey=True)
    bar_w = 0.28
    x = np.arange(len(PHASE_ORDER), dtype=float)

    for ax, policy in zip(axes, POLICY_ORDER):
        sub = summary[summary["policy_norm"] == policy].copy()
        for i, model_facet in enumerate(TARGET_MODEL_FACETS):
            model_sub = sub[sub["model_facet"] == model_facet].copy()
            xpos = x + (i - 0.5) * (bar_w * 2.2)
            tx_vals = [
                float(model_sub.loc[model_sub["phase_name"] == phase, "tx_total_gb"].iloc[0])
                if (model_sub["phase_name"] == phase).any()
                else np.nan
                for phase in PHASE_ORDER
            ]
            rx_vals = [
                float(model_sub.loc[model_sub["phase_name"] == phase, "rx_total_gb"].iloc[0])
                if (model_sub["phase_name"] == phase).any()
                else np.nan
                for phase in PHASE_ORDER
            ]
            ax.bar(
                xpos,
                tx_vals,
                width=bar_w,
                color=TX_COLOR,
                edgecolor=MODEL_EDGE[model_facet],
                linewidth=1.1,
                alpha=MODEL_ALPHA[model_facet],
            )
            ax.bar(
                xpos,
                rx_vals,
                width=bar_w,
                bottom=tx_vals,
                color=RX_COLOR,
                edgecolor=MODEL_EDGE[model_facet],
                linewidth=1.1,
                alpha=MODEL_ALPHA[model_facet],
            )

        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_DISPLAY[p] for p in PHASE_ORDER])
        ax.set_xlabel("Phase")
        ax.set_title(POLICY_DISPLAY[policy], fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("PCIe data moved per step (GB)")
    fig.suptitle("Phase PCIe Data Moved per Step by Policy and Model", y=0.99, fontweight="bold")
    legend_handles = [
        Patch(facecolor=TX_COLOR, edgecolor="black", label="Transmit (device to host)"),
        Patch(facecolor=RX_COLOR, edgecolor="black", label="Receive (host to device)"),
        Patch(facecolor="#777777", edgecolor="black", alpha=MODEL_ALPHA["Llama"], label=MODEL_DISPLAY["Llama"]),
        Patch(facecolor="#777777", edgecolor="black", alpha=MODEL_ALPHA["Qwen"], label=MODEL_DISPLAY["Qwen"]),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
