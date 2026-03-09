"""Scatter: comm_fraction/step vs utilization for Llama baseline vs smallmb.

x-axis: comm_fraction/step (iteration summary)
y-axis: time-weighted step utilization (sm_util_mean fallback gpu_util_mean) from phase_fact_view
color: mb setting (Baseline vs SmallMB)
marker: policy (PPO/GRPO/ReMax)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier1/util_vs_comm_fraction_smallmb_vs_baseline_llama_stage2.png")

RUNS = [
    {"run_id": "stage2_llama8b_ppo_20260301_120523", "policy": "PPO", "variant": "Baseline"},
    {"run_id": "stage2_llama8b_grpo_20260301_122602", "policy": "GRPO", "variant": "Baseline"},
    {"run_id": "stage2_llama8b_remax_20260301_121545", "policy": "ReMax", "variant": "Baseline"},
    {"run_id": "stage2_smallmb_llama8b_ppo_20260304_162933", "policy": "PPO", "variant": "SmallMB"},
    {"run_id": "stage2_smallmb_llama8b_grpo_20260304_171751", "policy": "GRPO", "variant": "SmallMB"},
    {"run_id": "stage2_smallmb_llama8b_remax_20260304_165430", "policy": "ReMax", "variant": "SmallMB"},
]

POLICY_COLORS = {"PPO": "#4e79a7", "GRPO": "#e15759", "ReMax": "#59a14f"}
VARIANT_MARKERS = {"Baseline": "o", "SmallMB": "s"}


def _choose_window(common_steps: list[int]) -> list[int]:
    preferred = [54, 55, 56, 57, 58]
    if all(s in common_steps for s in preferred):
        return preferred
    return sorted(common_steps)[-4:]


def _load_comm_fraction(run_id: str) -> pd.DataFrame:
    path = Path("results/monitoring_val") / run_id / f"{run_id}.jsonl"
    rows: list[dict[str, float | bool | str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            data = rec.get("data", {})
            if data.get("logging/record_scope") != "iteration_summary":
                continue
            step = data.get("training/global_step", rec.get("step"))
            if step is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "global_step_canonical": float(step),
                    "validation_logged": bool(data.get("logging/validation_logged", False)),
                    "comm_fraction_step": float(data.get("comm_fraction/step", np.nan)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["comm_fraction_step"] = pd.to_numeric(df["comm_fraction_step"], errors="coerce")
    df = df.dropna(subset=["global_step_canonical", "comm_fraction_step"]).copy()
    df = df[~df["validation_logged"]].copy()
    return df[["run_id", "global_step_canonical", "comm_fraction_step"]]


def main() -> None:
    meta = pd.DataFrame(RUNS)
    run_ids = meta["run_id"].tolist()

    # x metric from iteration summaries
    comm = pd.concat([_load_comm_fraction(rid) for rid in run_ids], ignore_index=True)
    comm = comm.merge(meta, on="run_id", how="inner")

    # y metric from phase_fact_view: time-weighted utilization per step
    phase, _ = load_view("phase_fact_view")
    phase = phase[phase["run_id"].astype(str).isin(run_ids)].copy()
    phase = phase[~phase["is_validation_step"].fillna(False)].copy()
    phase = phase[~phase["is_incomplete_phase"].fillna(False)].copy()
    phase = phase[~phase["is_warmup_idle"].fillna(False)].copy()
    phase = phase[phase["phase_name"].isin(["rollout", "rl_policy", "training"])].copy()

    # Compute chosen common late-step window.
    steps_by_run = {
        rid: set(comm.loc[comm["run_id"] == rid, "global_step_canonical"].dropna().astype(int).tolist()) for rid in run_ids
    }
    common_steps = sorted(set.intersection(*steps_by_run.values()))
    if not common_steps:
        raise ValueError("No common non-validation steps across selected runs.")
    chosen_steps = _choose_window(common_steps)

    comm = comm[comm["global_step_canonical"].astype(int).isin(chosen_steps)].copy()
    phase = phase[phase["global_step_canonical"].astype(int).isin(chosen_steps)].copy()

    phase["util_pct"] = pd.to_numeric(phase["sm_util_mean"], errors="coerce")
    phase["util_pct"] = phase["util_pct"].where(phase["util_pct"].notna(), pd.to_numeric(phase["gpu_util_mean"], errors="coerce"))
    phase["phase_time_s"] = pd.to_numeric(phase["phase_time_s"], errors="coerce")
    phase = phase.dropna(subset=["util_pct", "phase_time_s"]).copy()

    util_step = (
        phase.assign(weighted_util=phase["util_pct"] * phase["phase_time_s"])
        .groupby(["run_id", "global_step_canonical"], as_index=False)[["weighted_util", "phase_time_s"]]
        .sum()
    )
    util_step["util_pct_step_weighted"] = util_step["weighted_util"] / util_step["phase_time_s"].replace(0, np.nan)
    util_step = util_step[["run_id", "global_step_canonical", "util_pct_step_weighted"]]

    df = comm.merge(util_step, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(meta, on=["run_id", "policy", "variant"], how="inner")

    print("common steps across all runs:", common_steps)
    print("chosen matched late-step window:", chosen_steps)
    print("points per policy x variant:")
    print(df.groupby(["policy", "variant"]).size().rename("n_points").reset_index().to_string(index=False))
    print("means:")
    print(
        df.groupby(["policy", "variant"], as_index=False)[["comm_fraction_step", "util_pct_step_weighted"]]
        .mean()
        .to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for policy in ["PPO", "GRPO", "ReMax"]:
        for variant in ["Baseline", "SmallMB"]:
            sub = df[(df["policy"] == policy) & (df["variant"] == variant)].copy()
            if sub.empty:
                continue
            ax.scatter(
                sub["comm_fraction_step"],
                sub["util_pct_step_weighted"],
                s=58,
                alpha=0.9,
                c=POLICY_COLORS[policy],
                marker=VARIANT_MARKERS[variant],
                edgecolor="black",
                linewidth=0.5,
                label=f"{policy} - {variant}",
            )

    ax.set_xlabel("comm_fraction/step")
    ax.set_ylabel("Step Utilization (%)")
    ax.set_title("Are We Actually Idling? Utilization vs Comm Fraction\nLlama Stage2: Baseline vs SmallMB")
    ax.grid(alpha=0.2)

    # Compact unique legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
