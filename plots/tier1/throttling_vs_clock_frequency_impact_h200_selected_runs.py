"""Throttling vs Clock Frequency Impact (H200, selected runs).

Design:
- Violin plot of sm_clock_MHz split by:
  - Not Throttled (clocks_throttle_reasons_raw == 0)
  - Throttled (clocks_throttle_reasons_raw > 0)

Telemetry source:
- periodic NVML samples from hardware_periodic.parquet
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


OUTPATH = Path("plots/out/figures/tier1/throttling_vs_clock_frequency_impact_h200_selected_runs.png")

H200_RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
]


def main() -> None:
    # Use shared step-level analysis filter to get eligible run-step keys.
    sf, _ = load_view("step_fact_view")
    sf = sf[sf["run_id"].astype(str).isin(H200_RUN_IDS)].copy()
    sf_before = sf.copy()
    sf = apply_analysis_ok(sf)
    print(f"step_filtering={explain_filtering(sf_before, sf)}")
    keys = sf[["run_id", "global_step_canonical"]].drop_duplicates().copy()

    hp = pd.read_parquet(
        "DATASETS/hardware_periodic.parquet",
        columns=[
            "run_id",
            "global_step_canonical",
            "source",
            "device_kind",
            "sm_clock_MHz",
            "clocks_throttle_reasons_raw",
        ],
    )
    hp = hp[hp["run_id"].astype(str).isin(H200_RUN_IDS)].copy()
    hp = hp[hp["source"].astype(str).str.lower() == "nvml"].copy()
    hp = hp[hp["device_kind"].astype(str).str.lower() == "gpu"].copy()
    hp = hp.merge(keys, on=["run_id", "global_step_canonical"], how="inner")

    hp["sm_clock_MHz"] = pd.to_numeric(hp["sm_clock_MHz"], errors="coerce")
    hp["clocks_throttle_reasons_raw"] = pd.to_numeric(hp["clocks_throttle_reasons_raw"], errors="coerce").fillna(0.0)
    hp = hp.dropna(subset=["sm_clock_MHz"]).copy()
    hp = hp[hp["sm_clock_MHz"] > 0].copy()
    if hp.empty:
        raise ValueError("No valid H200 periodic rows after filtering.")

    hp["throttle_state"] = np.where(
        hp["clocks_throttle_reasons_raw"] > 0,
        "Throttled",
        "Not Throttled",
    )

    print("bitmask counts:")
    print(hp["clocks_throttle_reasons_raw"].value_counts().sort_index().to_string())

    summary = (
        hp.groupby("throttle_state", dropna=False)["sm_clock_MHz"]
        .agg(n_samples="size", mean="mean", median="median", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90))
        .reset_index()
        .sort_values("throttle_state")
    )
    print("clock summary by throttle state:")
    print(summary.to_string(index=False))

    # Optional policy-level summary for audit.
    runs, _ = load_view("run_summary_view")
    policy_map = runs[runs["run_id"].astype(str).isin(H200_RUN_IDS)][["run_id", "policy"]].drop_duplicates("run_id").copy()
    policy_map["policy"] = policy_map["policy"].astype(str).str.lower().replace({"remx": "remax"})
    hp_pol = hp.merge(policy_map, on="run_id", how="left")
    print("policy x state median clocks:")
    print(
        hp_pol.groupby(["policy", "throttle_state"], dropna=False)["sm_clock_MHz"]
        .median()
        .reset_index(name="median_sm_clock_MHz")
        .sort_values(["policy", "throttle_state"])
        .to_string(index=False)
    )

    order = ["Not Throttled", "Throttled"]
    vals = [hp.loc[hp["throttle_state"] == s, "sm_clock_MHz"].to_numpy() for s in order]

    fig, ax = plt.subplots(figsize=(8.3, 5.6))
    vp = ax.violinplot(
        vals,
        positions=[0, 1],
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    fill_colors = ["#4c78a8", "#e45756"]
    for body, c in zip(vp["bodies"], fill_colors):
        body.set_facecolor(c)
        body.set_edgecolor("black")
        body.set_alpha(0.45)
        body.set_linewidth(0.8)

    bp = ax.boxplot(
        vals,
        positions=[0, 1],
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.2},
    )
    _ = bp

    for i, state in enumerate(order):
        s = summary[summary["throttle_state"] == state]
        if s.empty:
            continue
        row = s.iloc[0]
        txt = f"n={int(row['n_samples'])}\nmedian={row['median']:.0f} MHz"
        ax.text(
            i,
            float(row["p90"]) * 1.01,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85, "pad": 0.25},
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(order)
    ax.set_xlabel("Throttle status from clocks_throttle_reasons_raw")
    ax.set_ylabel("sm_clock_MHz")
    ax.set_title("Throttling vs Clock Frequency Impact (H200)")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    fig.text(
        0.5,
        0.01,
        "Throttled defined as clocks_throttle_reasons_raw > 0 on NVML periodic GPU samples.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
