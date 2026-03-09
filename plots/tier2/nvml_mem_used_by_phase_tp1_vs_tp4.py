"""Phase-scoped NVML GPU memory used (GB): TP=1 vs TP=4 for the same 8 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPATH = Path("plots/out/figures/tier2/nvml_mem_used_by_phase_tp1_vs_tp4.png")
BASE = Path("results/monitoring_val")

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

PHASES = ["rollout", "rl_policy", "training"]
PANEL_ORDER = [("Llama", "PPO"), ("Llama", "ReMax"), ("Qwen", "PPO"), ("Qwen", "ReMax")]
TP_COLORS = {1: "#4e79a7", 4: "#e15759"}


def _load_nvml_boundary(run_id: str) -> pd.DataFrame:
    path = BASE / run_id / "nvml_boundary.jsonl"
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            phase = str(rec.get("phase_name", "")).lower()
            mem_used_b = rec.get("mem_used_B")
            if phase not in PHASES or not isinstance(mem_used_b, (int, float)):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "phase_name": phase,
                    "mem_used_gb": float(mem_used_b) / 1e9,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    run_meta = pd.DataFrame(RUNS)
    raw = pd.concat([_load_nvml_boundary(r["run_id"]) for r in RUNS], ignore_index=True)
    raw = raw.merge(run_meta, on="run_id", how="inner")

    if raw.empty:
        raise ValueError("No NVML boundary memory rows loaded.")

    agg = (
        raw.groupby(["run_id", "model", "policy", "tp", "phase_name"], as_index=False)
        .agg(
            mem_mean_gb=("mem_used_gb", "mean"),
            mem_p95_gb=("mem_used_gb", lambda x: float(np.nanquantile(x, 0.95))),
            mem_max_gb=("mem_used_gb", "max"),
            n_samples=("mem_used_gb", "size"),
        )
        .sort_values(["model", "policy", "tp", "phase_name"])
    )

    print("phase-scoped NVML mem_used_B summary (GB):")
    print(agg.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    ymax = max(1.0, float(agg["mem_mean_gb"].max()) * 1.2)
    width = 0.36
    x = np.arange(len(PHASES), dtype=float)

    for i, (model, policy) in enumerate(PANEL_ORDER):
        ax = axes[i // 2, i % 2]
        sub = agg[(agg["model"] == model) & (agg["policy"] == policy)].copy()

        for tp, offset in [(1, -width / 2), (4, width / 2)]:
            tsub = sub[sub["tp"] == tp]
            means = []
            errs = []
            for ph in PHASES:
                row = tsub[tsub["phase_name"] == ph]
                if row.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                else:
                    m = float(row["mem_mean_gb"].iloc[0])
                    p95 = float(row["mem_p95_gb"].iloc[0])
                    means.append(m)
                    errs.append(max(0.0, p95 - m))
            ax.bar(
                x + offset,
                np.nan_to_num(np.array(means), nan=0.0),
                width=width,
                yerr=np.array(errs),
                capsize=3,
                color=TP_COLORS[tp],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.7,
                label=f"TP={tp}",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(PHASES)
        ax.set_title(f"{model} - {policy}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, ymax)
        if i == 0:
            ax.legend(frameon=False)

    axes[0, 0].set_ylabel("NVML mem_used_B (GB)")
    axes[1, 0].set_ylabel("NVML mem_used_B (GB)")
    fig.suptitle("Phase-Scoped NVML GPU Memory Used: TP=1 vs TP=4", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
