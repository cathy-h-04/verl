"""Training MFU per GPU: TP=1 vs TP=4 for Llama/Qwen PPO/ReMax."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier2/training_mfu_per_gpu_tp1_vs_tp4.png")

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

PANEL_ORDER = [("Llama", "PPO"), ("Llama", "ReMax"), ("Qwen", "PPO"), ("Qwen", "ReMax")]
TP_COLORS = {1: "#4e79a7", 4: "#e15759"}


def main() -> None:
    step, _ = load_view("step_fact_view")
    required = ["run_id", "global_step_canonical", "validation_logged", "mfu_actor"]
    missing = [c for c in required if c not in step.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    run_meta = pd.DataFrame(RUNS)
    run_ids = run_meta["run_id"].tolist()

    df = step[step["run_id"].astype(str).isin(run_ids)][required].copy()
    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"step_filtering={explain_filtering(before, df)}")

    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["mfu_actor"] = pd.to_numeric(df["mfu_actor"], errors="coerce")
    df["validation_logged"] = df["validation_logged"].fillna(False).astype(bool)
    df = df.dropna(subset=["global_step_canonical", "mfu_actor"]).copy()
    df = df[~df["validation_logged"]].copy()
    df = df.merge(run_meta, on="run_id", how="inner")

    agg = (
        df.groupby(["run_id", "model", "policy", "tp"], as_index=False)
        .agg(
            mfu_mean=("mfu_actor", "mean"),
            mfu_std=("mfu_actor", "std"),
            n_steps=("mfu_actor", "size"),
        )
    )
    agg["mfu_std"] = agg["mfu_std"].fillna(0.0)

    print("training mfu summary:")
    print(
        agg.sort_values(["model", "policy", "tp"])[
            ["run_id", "model", "policy", "tp", "n_steps", "mfu_mean", "mfu_std"]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes_flat = axes.flatten()
    ymax = max(0.1, float(agg["mfu_mean"].max()) * 1.25) if not agg.empty else 1.0

    for ax, (model, policy) in zip(axes_flat, PANEL_ORDER):
        sub = agg[(agg["model"] == model) & (agg["policy"] == policy)].copy()
        sub = sub.sort_values("tp")
        x = np.array([0, 1], dtype=float)
        labels = ["TP=1", "TP=4"]
        y = np.array(
            [
                float(sub.loc[sub["tp"] == 1, "mfu_mean"].iloc[0]) if (sub["tp"] == 1).any() else np.nan,
                float(sub.loc[sub["tp"] == 4, "mfu_mean"].iloc[0]) if (sub["tp"] == 4).any() else np.nan,
            ]
        )
        yerr = np.array(
            [
                float(sub.loc[sub["tp"] == 1, "mfu_std"].iloc[0]) if (sub["tp"] == 1).any() else 0.0,
                float(sub.loc[sub["tp"] == 4, "mfu_std"].iloc[0]) if (sub["tp"] == 4).any() else 0.0,
            ]
        )

        bars = ax.bar(
            x,
            np.nan_to_num(y, nan=0.0),
            yerr=yerr,
            width=0.62,
            color=[TP_COLORS[1], TP_COLORS[4]],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
            capsize=4,
        )
        for i, b in enumerate(bars):
            if np.isnan(y[i]):
                ax.text(b.get_x() + b.get_width() / 2, 0.01, "NA", ha="center", va="bottom", fontsize=9)
            else:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + ymax * 0.02,
                    f"{y[i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{model} - {policy}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, ymax)

    axes[0, 0].set_ylabel("Training MFU per GPU (mfu_actor)")
    axes[1, 0].set_ylabel("Training MFU per GPU (mfu_actor)")
    fig.suptitle("Training MFU per GPU: TP=1 vs TP=4 (Baselines + TP4 Runs)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
