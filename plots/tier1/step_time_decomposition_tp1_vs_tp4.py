"""Communication-only step decomposition for TP=1 vs TP=4 using the same 8 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/figures/tier1/step_time_decomposition_tp1_vs_tp4.png")

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

COMM_COLS = [
    "comm_s/gen",
    "comm_s/values",
    "comm_s/old_log_prob",
    "comm_s/update_actor",
    "comm_s/update_critic",
]
COMM_COLORS = {
    "comm_s/gen": "#4e79a7",
    "comm_s/values": "#f28e2b",
    "comm_s/old_log_prob": "#59a14f",
    "comm_s/update_actor": "#e15759",
    "comm_s/update_critic": "#76b7b2",
}


def _load_run_jsonl_steps(run_id: str) -> pd.DataFrame:
    path = Path("results/monitoring_val") / run_id / f"{run_id}.jsonl"
    rows: list[dict[str, float | bool | str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = rec.get("data", {})
            step = data.get("training/global_step", rec.get("step"))
            if step is None:
                continue
            row: dict[str, float | bool | str] = {
                "run_id": run_id,
                "global_step_canonical": float(step),
                "validation_logged": bool(data.get("logging/validation_logged", False)),
                "timing_s/step": data.get("timing_s/step", np.nan),
                "timing_s/gen": data.get("timing_s/gen", np.nan),
                "comm_s/step": data.get("comm_s/step", np.nan),
            }
            for c in COMM_COLS:
                row[c] = data.get(c, np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["global_step_canonical", "timing_s/step", "timing_s/gen", "comm_s/step"] + COMM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    run_meta = pd.DataFrame(RUNS)
    run_ids = run_meta["run_id"].tolist()

    # Use step_fact_view only for filtering parity (startup exclusion, integrity, validation mask).
    sf, _ = load_view("step_fact_view")
    req = ["run_id", "global_step_canonical", "validation_logged"]
    filt = sf[sf["run_id"].astype(str).isin(run_ids)][req].copy()
    filt = apply_analysis_ok(filt)
    filt["global_step_canonical"] = pd.to_numeric(filt["global_step_canonical"], errors="coerce")
    filt["validation_logged"] = filt["validation_logged"].fillna(False).astype(bool)
    filt = filt.dropna(subset=["global_step_canonical"]).copy()
    filt = filt[~filt["validation_logged"]].copy()
    filt = filt[["run_id", "global_step_canonical"]].drop_duplicates()

    raw = pd.concat([_load_run_jsonl_steps(rid) for rid in run_ids], ignore_index=True)
    raw = raw.dropna(subset=["global_step_canonical", "timing_s/step"]).copy()
    raw = raw.merge(filt, on=["run_id", "global_step_canonical"], how="inner")
    raw = raw.merge(run_meta, on="run_id", how="inner")

    for c in COMM_COLS:
        raw[c] = raw[c].fillna(0.0)
    raw["comm_sum_components"] = raw[COMM_COLS].sum(axis=1)
    raw["comm_step_ref"] = pd.to_numeric(raw["comm_s/step"], errors="coerce")
    raw["comm_total_used"] = raw["comm_step_ref"].where(raw["comm_step_ref"].notna(), raw["comm_sum_components"])
    agg_cols = COMM_COLS + ["timing_s/step", "timing_s/gen", "comm_total_used"]
    agg = raw.groupby(["model", "policy", "tp"], as_index=False)[agg_cols].mean()

    print("mean step decomposition (s/step):")
    print(
        agg.sort_values(["model", "policy", "tp"])[
            ["model", "policy", "tp"] + COMM_COLS + ["comm_total_used", "timing_s/step"]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes_flat = axes.flatten()
    y_max = float((agg["comm_total_used"].max() if not agg.empty else 1.0) * 1.35)

    legend_handles = None
    legend_labels = None
    seen_legend_labels: set[str] = set()

    for ax, (model, policy) in zip(axes_flat, PANELS):
        sub = agg[(agg["model"] == model) & (agg["policy"] == policy)].copy().sort_values("tp")
        tp_to_x = {1: 0.0, 4: 1.0}
        for tp in [1, 4]:
            x = tp_to_x[tp]
            row = sub[sub["tp"] == tp]
            if row.empty:
                ax.text(x, 0.02 * y_max, "NA", ha="center", va="bottom", fontsize=10)
                continue
            row0 = row.iloc[0]
            bottom = 0.0
            for c in COMM_COLS:
                val = float(row0[c])
                ax.bar(
                    x,
                    val,
                    bottom=bottom,
                    width=0.6,
                    color=COMM_COLORS[c],
                    edgecolor="black",
                    linewidth=0.5,
                    label=c if c not in seen_legend_labels else "_nolegend_",
                )
                seen_legend_labels.add(c)
                bottom += val
            ax.text(x, bottom + 0.02 * y_max, f"{bottom:.3f}s", ha="center", va="bottom", fontsize=9)

        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["TP=1", "TP=4"])
        ax.set_title(f"{model} - {policy}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, y_max)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0, 0].set_ylabel("Communication seconds per step (mean)")
    axes[1, 0].set_ylabel("Communication seconds per step (mean)")
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Step Communication Decomposition: TP=1 vs TP=4 (Comm Components Only)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
