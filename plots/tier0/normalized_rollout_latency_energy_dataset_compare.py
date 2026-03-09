"""Rollout-normalized latency + energy and straggler ratio (GSM8K vs RLHF dataset)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/normalized_rollout_latency_energy_dataset_compare.png")
GSM8K_RUN_ID = "stage1_llama8b_ppo_20260301_075906"
RLHF_RUN_ID = "llama31_8b_smoke_test_rlhf_ff_20260304_192533"
RUN_ORDER = [GSM8K_RUN_ID, RLHF_RUN_ID]
RUN_LABELS = {
    GSM8K_RUN_ID: "GSM8K dataset",
    RLHF_RUN_ID: "RLHF dataset",
}
RUN_COLORS = {
    GSM8K_RUN_ID: "#1f77b4",
    RLHF_RUN_ID: "#d62728",
}


def _box_with_jitter(ax, data_by_run: dict[str, np.ndarray], ylabel: str, title: str) -> None:
    vals = [data_by_run.get(r, np.array([])) for r in RUN_ORDER]
    bp = ax.boxplot(
        vals,
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.1},
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35, "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    for patch, rid in zip(bp["boxes"], RUN_ORDER):
        patch.set_facecolor(RUN_COLORS[rid])
        patch.set_alpha(0.25)
    rng = np.random.default_rng(42)
    for i, rid in enumerate(RUN_ORDER, start=1):
        v = data_by_run.get(rid, np.array([]))
        if v.size == 0:
            continue
        x = i + rng.uniform(-0.08, 0.08, size=v.size)
        ax.scatter(x, v, s=10, alpha=0.25, color=RUN_COLORS[rid], edgecolors="none")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([RUN_LABELS[r] for r in RUN_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)


def main() -> None:
    # Left metrics: rollout_time_per_token and rollout_energy_per_token from phase_fact_view.
    pf, _ = load_view("phase_fact_view")
    req_phase = ["run_id", "phase_name", "phase_time_s", "gpu_energy_j", "rollout_total_tokens"]
    miss_phase = [c for c in req_phase if c not in pf.columns]
    if miss_phase:
        raise ValueError(f"phase_fact_view missing required columns: {miss_phase}")

    phase = pf[pf["run_id"].astype(str).isin(RUN_ORDER)][req_phase].copy()
    before_phase = phase.copy()
    phase = apply_analysis_ok(phase)
    print(f"phase_filtering={explain_filtering(before_phase, phase)}")
    phase["phase_name"] = phase["phase_name"].astype(str).str.lower()
    phase = phase[phase["phase_name"] == "rollout"].copy()
    for c in ["phase_time_s", "gpu_energy_j", "rollout_total_tokens"]:
        phase[c] = pd.to_numeric(phase[c], errors="coerce")
    phase = phase.dropna(subset=["phase_time_s", "gpu_energy_j", "rollout_total_tokens"]).copy()
    phase = phase[phase["rollout_total_tokens"] > 0].copy()
    phase["rollout_time_per_token"] = phase["phase_time_s"] / phase["rollout_total_tokens"]
    phase["rollout_energy_per_token"] = phase["gpu_energy_j"] / phase["rollout_total_tokens"]

    # Right metric: straggler_ratio from step_fact_view.
    step, _ = load_view("step_fact_view")
    req_step = ["run_id", "global_step_canonical", "straggler_ratio"]
    miss_step = [c for c in req_step if c not in step.columns]
    if miss_step:
        raise ValueError(f"step_fact_view missing required columns: {miss_step}")
    step = step[step["run_id"].astype(str).isin(RUN_ORDER)][req_step].copy()
    before_step = step.copy()
    step = apply_analysis_ok(step)
    print(f"step_filtering={explain_filtering(before_step, step)}")
    step["global_step_canonical"] = pd.to_numeric(step["global_step_canonical"], errors="coerce")
    step["straggler_ratio"] = pd.to_numeric(step["straggler_ratio"], errors="coerce")
    step = step.dropna(subset=["global_step_canonical", "straggler_ratio"]).copy()

    # Logs
    print("rollout_time/energy_per_token summary by dataset:")
    s = phase.groupby("run_id", dropna=False).agg(
        n_rollout_phases=("rollout_time_per_token", "size"),
        time_per_token_mean=("rollout_time_per_token", "mean"),
        time_per_token_p95=("rollout_time_per_token", lambda x: x.quantile(0.95)),
        energy_per_token_mean=("rollout_energy_per_token", "mean"),
        energy_per_token_p95=("rollout_energy_per_token", lambda x: x.quantile(0.95)),
    ).reset_index()
    s["dataset"] = s["run_id"].map(RUN_LABELS)
    print(s[["run_id", "dataset", "n_rollout_phases", "time_per_token_mean", "time_per_token_p95", "energy_per_token_mean", "energy_per_token_p95"]].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    time_data = {rid: phase.loc[phase["run_id"] == rid, "rollout_time_per_token"].to_numpy(dtype=float) for rid in RUN_ORDER}
    energy_data = {rid: phase.loc[phase["run_id"] == rid, "rollout_energy_per_token"].to_numpy(dtype=float) for rid in RUN_ORDER}
    _box_with_jitter(
        axes[0],
        time_data,
        ylabel="rollout_time_per_token (s/token)",
        title="Normalized Rollout Latency",
    )
    _box_with_jitter(
        axes[1],
        energy_data,
        ylabel="rollout_energy_per_token (J/token)",
        title="Normalized Rollout Energy",
    )

    straggler_data = {rid: step.loc[step["run_id"] == rid, "straggler_ratio"].to_numpy(dtype=float) for rid in RUN_ORDER}
    _box_with_jitter(
        axes[2],
        straggler_data,
        ylabel="rollout/straggler_ratio",
        title="Straggler Ratio",
    )

    fig.suptitle("Rollout Efficiency and Imbalance\nPPO Llama: GSM8K vs RLHF dataset", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
