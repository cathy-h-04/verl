"""Rollout vs RL-policy phase energy+time share for PPO Llama: baseline vs RLHF dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/rollout_vs_rlpolicy_phase_energy_time_share_ppo_llama.png")
BASELINE_RUN_ID = "stage1_llama8b_ppo_20260301_075906"
RLHF_RUN_ID = "llama31_8b_smoke_test_rlhf_ff_20260304_192533"
TARGET_PHASES = ["rollout", "rl_policy", "training"]
RUN_LABELS = {
    BASELINE_RUN_ID: "GSM8K Dataset PPO Llama",
    RLHF_RUN_ID: "RLHF Dataset PPO Llama",
}
RUN_COLORS = {
    BASELINE_RUN_ID: "#1f77b4",
    RLHF_RUN_ID: "#d62728",
}
PHASE_COLORS = {
    "rollout": "#4C78A8",
    "rl_policy": "#F58518",
    "training": "#54A24B",
}


def _autopct_totals(values: np.ndarray, unit: str):
    total = float(np.sum(values))

    def _fmt(pct: float) -> str:
        if total <= 0:
            return ""
        val = pct * total / 100.0
        if unit == "J":
            return f"{val/1000.0:.1f}kJ" if val >= 1000 else f"{val:.0f}J"
        if unit == "s":
            return f"{val/60.0:.1f}m" if val >= 60 else f"{val:.1f}s"
        return f"{val:.1f}"

    return _fmt


def main() -> None:
    pf, _ = load_view("phase_fact_view")
    required = ["run_id", "phase_name", "total_energy_j", "phase_time_s"]
    missing = [c for c in required if c not in pf.columns]
    if missing:
        raise ValueError(f"phase_fact_view missing required columns: {missing}")

    df = pf[pf["run_id"].astype(str).isin([BASELINE_RUN_ID, RLHF_RUN_ID])].copy()
    if df.empty:
        raise ValueError("No rows found for requested run IDs.")

    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")

    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(TARGET_PHASES)].copy()
    df["total_energy_j"] = pd.to_numeric(df["total_energy_j"], errors="coerce")
    df["phase_time_s"] = pd.to_numeric(df["phase_time_s"], errors="coerce")
    df = df.dropna(subset=["total_energy_j", "phase_time_s"]).copy()
    if df.empty:
        raise ValueError("No phase energy/time-total rows after filtering.")

    agg = df.groupby(["run_id", "phase_name"], dropna=False)[["total_energy_j", "phase_time_s"]].sum().reset_index()
    print("phase totals by run:")
    print(agg.sort_values(["phase_name", "run_id"]).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={"aspect": "equal"})
    run_order = [BASELINE_RUN_ID, RLHF_RUN_ID]
    metric_order = [("total_energy_j", "Energy total", "J"), ("phase_time_s", "Time total", "s")]

    for r, (metric, metric_title, unit) in enumerate(metric_order):
        for c, run_id in enumerate(run_order):
            ax = axes[r][c]
            sub = agg[agg["run_id"] == run_id].set_index("phase_name")
            vals = np.array([float(sub.loc[p, metric]) if p in sub.index else 0.0 for p in TARGET_PHASES], dtype=float)
            colors = [PHASE_COLORS[p] for p in TARGET_PHASES]
            ax.pie(
                vals if vals.sum() > 0 else [1.0],
                labels=TARGET_PHASES if vals.sum() > 0 else [""],
                colors=colors if vals.sum() > 0 else ["#CCCCCC"],
                autopct=_autopct_totals(vals, unit) if vals.sum() > 0 else None,
                startangle=90,
                textprops={"fontsize": 8},
            )
            run_title = RUN_LABELS[run_id]
            if c == 0:
                ax.set_ylabel(f"{metric_title} ({unit})", fontsize=10)
            ax.set_title(run_title, fontsize=10)

    phase_handles = [plt.Line2D([0], [0], marker="o", linestyle="None", color=PHASE_COLORS[p], label=p) for p in TARGET_PHASES]
    fig.legend(handles=phase_handles, title="phase", frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("PPO Llama: Phase Energy and Time Totals (GSM8K vs RLHF dataset)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
