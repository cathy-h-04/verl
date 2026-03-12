"""Plot 2.2: CPU-GPU Interaction (DRAM energy normalized by rollout tokens).

Data grain:
- Phase-level rows from phase_fact_view.

Metric:
- dram_uJ_per_rollout_token = dram_energy_j * 1e6 / step_rollout_output_tokens

Grouping:
- Platform mapping from run_id:
  - 2gpu_h200 -> Sapphire Rapids/DDR5
  - 4gpu_a100 -> Ice Lake/DDR4
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier2/cpu_gpu_interaction_dram_energy_by_pcie_gen_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PHASE_ORDER = ["rollout", "training", "rl_policy"]
PLATFORM_ORDER = ["Sapphire Rapids/DDR5", "Ice Lake/DDR4"]
PLATFORM_COLORS = {
    "Sapphire Rapids/DDR5": "#4c78a8",
    "Ice Lake/DDR4": "#f58518",
}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "Sapphire Rapids/DDR5"
    if "4gpu_a100" in rid:
        return "Ice Lake/DDR4"
    return "other"


def _phase_bucket(phase_name: str) -> str:
    s = str(phase_name).strip().lower()
    if s in {"rollout", "training", "rl_policy"}:
        return s
    return "other"


def main() -> None:
    pf, _ = load_view("phase_fact_view")
    sf, _ = load_view("step_fact_view")

    pf = pf[pf["run_id"].astype(str).isin(RUN_IDS)].copy()
    sf = sf[sf["run_id"].astype(str).isin(RUN_IDS)].copy()
    if pf.empty or sf.empty:
        raise ValueError("Selected RUN_IDS missing from phase_fact_view or step_fact_view.")

    pf_before = pf.copy()
    sf_before = sf.copy()
    pf = apply_analysis_ok(pf)
    sf = apply_analysis_ok(sf)
    print(f"phase_filtering={explain_filtering(pf_before, pf)}")
    print(f"step_filtering={explain_filtering(sf_before, sf)}")

    pf = pf[["run_id", "global_step_canonical", "phase_name", "dram_energy_j", "total_energy_j"]].copy()
    sf = sf[["run_id", "global_step_canonical", "step_rollout_output_tokens", "policy"]].copy()

    pf["phase_name"] = pf["phase_name"].map(_phase_bucket)
    pf = pf[pf["phase_name"].isin(PHASE_ORDER)].copy()
    pf["dram_energy_j"] = pd.to_numeric(pf["dram_energy_j"], errors="coerce")
    pf["total_energy_j"] = pd.to_numeric(pf["total_energy_j"], errors="coerce")

    sf["step_rollout_output_tokens"] = pd.to_numeric(sf["step_rollout_output_tokens"], errors="coerce")
    sf["policy_norm"] = sf["policy"].astype(str).str.lower().replace({"remx": "remax"})

    df = pf.merge(
        sf[["run_id", "global_step_canonical", "step_rollout_output_tokens", "policy_norm"]],
        on=["run_id", "global_step_canonical"],
        how="inner",
    )
    df = df[df["step_rollout_output_tokens"] > 0].copy()
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df = df[df["platform"].isin(PLATFORM_ORDER)].copy()
    df["dram_uj_per_rollout_token"] = (df["dram_energy_j"] * 1_000_000.0) / df["step_rollout_output_tokens"]
    df["dram_energy_share"] = df["dram_energy_j"] / df["total_energy_j"].replace(0, np.nan)
    df = df.dropna(subset=["dram_uj_per_rollout_token"]).copy()

    if df.empty:
        raise ValueError("No valid rows for DRAM-per-token metric after filtering.")

    summary = (
        df.groupby(["platform", "phase_name"], dropna=False)
        .agg(
            n_rows=("dram_uj_per_rollout_token", "size"),
            mean_dram_uj_per_token=("dram_uj_per_rollout_token", "mean"),
            median_dram_uj_per_token=("dram_uj_per_rollout_token", "median"),
            mean_dram_energy_share=("dram_energy_share", "mean"),
        )
        .reset_index()
        .sort_values(["phase_name", "platform"])
    )
    print("platform x phase summary:")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(PHASE_ORDER))
    w = 0.34
    pos_data = []
    pos_vals = []
    pos_colors = []

    for i, phase in enumerate(PHASE_ORDER):
        for platform in PLATFORM_ORDER:
            vals = pd.to_numeric(
                df.loc[(df["phase_name"] == phase) & (df["platform"] == platform), "dram_uj_per_rollout_token"],
                errors="coerce",
            ).dropna()
            if vals.empty:
                continue
            pos = i + (-0.5 if platform == PLATFORM_ORDER[0] else 0.5) * w
            pos_data.append(vals.to_numpy())
            pos_vals.append(pos)
            pos_colors.append(PLATFORM_COLORS[platform])

    bp = ax.boxplot(
        pos_data,
        positions=pos_vals,
        widths=w * 0.82,
        showfliers=False,
        patch_artist=True,
        boxprops={"edgecolor": "black", "linewidth": 0.75},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.75},
        capprops={"color": "black", "linewidth": 0.75},
    )
    for patch, c in zip(bp["boxes"], pos_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.38)

    # Annotate medians above each box.
    for vals, pos in zip(pos_data, pos_vals):
        med = float(np.median(vals))
        ax.text(pos, med, f"{med:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_ORDER)
    ax.set_xlabel("Phase")
    ax.set_ylabel("DRAM energy normalized by rollout tokens (uJ/token)")
    ax.set_title("Plot 2.2: CPU-GPU Interaction (DRAM Energy by Host/Memory Platform)")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PLATFORM_COLORS[p], edgecolor="black", alpha=0.5, label=p)
        for p in PLATFORM_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, title="Platform")

    fig.text(
        0.5,
        0.01,
        "Note: platform mapping inferred from run IDs (2gpu_h200 -> Sapphire Rapids/DDR5, 4gpu_a100 -> Ice Lake/DDR4).",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
