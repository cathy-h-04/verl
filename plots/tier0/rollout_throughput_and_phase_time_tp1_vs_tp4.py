"""Side-by-side TP1 vs TP4: rollout throughput-by-step and phase-time decomposition."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/figures/tier0/rollout_throughput_and_phase_time_tp1_vs_tp4.png")

RUNS = [
    {"run_id": "stage1_llama8b_ppo_20260301_075906", "model": "Llama", "policy": "PPO", "tp": 1},
    {"run_id": "llama8b_ppo_tp4_20260304_234405", "model": "Llama", "policy": "PPO", "tp": 4},
    {"run_id": "stage1_llama8b_remax_20260301_083423", "model": "Llama", "policy": "ReMax", "tp": 1},
    {"run_id": "llama8b_remax_tp4_20260305_003135", "model": "Llama", "policy": "ReMax", "tp": 4},
    {"run_id": "qwen_sys_3b_ppo_20260301_094328", "model": "Qwen", "policy": "PPO", "tp": 1},
    {"run_id": "qwen_sys_3b_ppo_tp4_20260305_025050", "model": "Qwen", "policy": "PPO", "tp": 4},
    {"run_id": "qwen_sys_3b_remax_20260301_100809", "model": "Qwen", "policy": "ReMax", "tp": 1},
    {"run_id": "qwen_sys_3b_remax_tp4_20260305_043245", "model": "Qwen", "policy": "ReMax", "tp": 4},
]

PANELS = [("Llama", "PPO"), ("Llama", "ReMax"), ("Qwen", "PPO"), ("Qwen", "ReMax")]
TP_COLORS = {1: "#4e79a7", 4: "#e15759"}
PHASES = ["rollout", "rl_policy", "training"]
PHASE_COLORS = {"rollout": "#59a14f", "rl_policy": "#f28e2b", "training": "#4e79a7"}


def main() -> None:
    run_meta = pd.DataFrame(RUNS)
    run_ids = run_meta["run_id"].tolist()

    pf, _ = load_view("phase_fact_view")
    req = ["run_id", "global_step_canonical", "phase_name", "phase_time_s", "rollout_total_tokens"]
    missing = [c for c in req if c not in pf.columns]
    if missing:
        raise ValueError(f"phase_fact_view missing required columns: {missing}")

    df = pf[pf["run_id"].astype(str).isin(run_ids)][req].copy()
    df = apply_analysis_ok(df)
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["phase_time_s"] = pd.to_numeric(df["phase_time_s"], errors="coerce")
    df["rollout_total_tokens"] = pd.to_numeric(df["rollout_total_tokens"], errors="coerce")
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df.dropna(subset=["global_step_canonical", "phase_time_s", "phase_name"]).copy()
    df = df.merge(run_meta, on="run_id", how="inner")

    # Throughput metric choice: rollout_total_tokens / rollout phase_time_s (rollout-specific).
    roll = df[df["phase_name"] == "rollout"].copy()
    roll = roll[(roll["phase_time_s"] > 0) & (roll["rollout_total_tokens"] > 0)].copy()
    roll["rollout_tokens_per_s"] = roll["rollout_total_tokens"] / roll["phase_time_s"]
    roll = roll.sort_values(["run_id", "global_step_canonical"])

    phase_means = (
        df[df["phase_name"].isin(PHASES)]
        .groupby(["model", "policy", "tp", "phase_name"], as_index=False)["phase_time_s"]
        .mean()
    )

    print("rollout throughput summary (tokens/s):")
    print(
        roll.groupby(["model", "policy", "tp"], as_index=False)["rollout_tokens_per_s"]
        .agg(["mean", "median"])
        .reset_index()
        .to_string(index=False)
    )
    print("phase mean time summary (s):")
    print(
        phase_means.pivot_table(
            index=["model", "policy", "tp"],
            columns="phase_name",
            values="phase_time_s",
            aggfunc="mean",
        )
        .reset_index()
        .to_string(index=False)
    )

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 16))

    for i, (model, policy) in enumerate(PANELS):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        # Left: per-step rollout throughput (line), TP1 vs TP4.
        sub_roll = roll[(roll["model"] == model) & (roll["policy"] == policy)].copy()
        for tp in [1, 4]:
            s = sub_roll[sub_roll["tp"] == tp].sort_values("global_step_canonical")
            if s.empty:
                continue
            ax_l.plot(
                s["global_step_canonical"],
                s["rollout_tokens_per_s"],
                color=TP_COLORS[tp],
                linewidth=1.6,
                alpha=0.9,
                label=f"TP={tp}",
            )
            ax_l.scatter(
                s["global_step_canonical"],
                s["rollout_tokens_per_s"],
                color=TP_COLORS[tp],
                s=10,
                alpha=0.6,
            )
        ax_l.set_title(f"{model} - {policy}: Rollout Throughput")
        ax_l.set_xlabel("step")
        ax_l.set_ylabel("rollout_total_tokens / rollout_phase_time_s (tokens/s)")
        ax_l.grid(alpha=0.2)
        ax_l.legend(frameon=False, loc="best")

        # Right: mean phase-time decomposition (stacked bars), TP1 vs TP4.
        sub_phase = phase_means[(phase_means["model"] == model) & (phase_means["policy"] == policy)].copy()
        x = np.array([0.0, 1.0])
        bottoms = np.zeros_like(x)
        phase_vals_by_tp: dict[str, dict[int, float]] = {ph: {1: 0.0, 4: 0.0} for ph in PHASES}
        for ph in PHASES:
            vals = []
            for tp in [1, 4]:
                m = sub_phase[(sub_phase["tp"] == tp) & (sub_phase["phase_name"] == ph)]
                vals.append(float(m["phase_time_s"].iloc[0]) if not m.empty else 0.0)
                phase_vals_by_tp[ph][tp] = vals[-1]
            v = np.array(vals, dtype=float)
            ax_r.bar(
                x,
                v,
                bottom=bottoms,
                width=0.62,
                color=PHASE_COLORS[ph],
                edgecolor="black",
                linewidth=0.6,
                label=ph,
            )
            bottoms += v
        # Annotate TP=4 stack segments with percent change vs TP=1 baseline.
        y_cursor = 0.0
        for ph in PHASES:
            base = phase_vals_by_tp[ph][1]
            tp4 = phase_vals_by_tp[ph][4]
            if tp4 <= 0:
                continue
            if base > 0:
                pct = (tp4 - base) / base * 100.0
                txt = f"{ph} {pct:+.0f}%"
            else:
                txt = f"{ph} NA"
            y_mid = y_cursor + tp4 / 2.0
            if tp4 >= 0.8:
                ax_r.text(
                    1.0,
                    y_mid,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
            else:
                ax_r.text(
                    1.12,
                    y_mid,
                    txt,
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            y_cursor += tp4
        for j, total in enumerate(bottoms):
            ax_r.text(x[j], total + max(0.02, 0.02 * float(bottoms.max() if bottoms.max() > 0 else 1.0)), f"{total:.2f}s", ha="center", va="bottom", fontsize=9)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(["TP=1", "TP=4"])
        ax_r.set_title(f"{model} - {policy}: Mean Phase Time per Step")
        ax_r.set_ylabel("seconds")
        ax_r.grid(axis="y", alpha=0.2)

        # Keep right-plot legend once (top row only).
        if i == 0:
            ax_r.legend(frameon=False, loc="best")

    fig.suptitle(
        "TP1 vs TP4: Rollout Throughput by Step (Left) and Phase-Time Decomposition (Right)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
