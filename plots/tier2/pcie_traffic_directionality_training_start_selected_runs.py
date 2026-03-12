"""PCIe traffic directionality at Training START boundary (selected runs).

Goal:
- Inspect potential host->device bottlenecks at phase boundary.

Metrics:
- pcie_tx_bytes_s (Device -> Host)
- pcie_rx_bytes_s (Host -> Device)

Boundary focus:
- Training phase START, using the first periodic GPU sample at/after start.
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


OUTPATH = Path("plots/out/figures/tier2/pcie_traffic_directionality_training_start_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PLATFORM_ORDER = ["2xH200", "4xA100"]
PLATFORM_COLORS = {"2xH200": "#4c78a8", "4xA100": "#f58518"}
DIRECTION_COLORS = {
    "Device -> Host (TX)": "#4c78a8",
    "Host -> Device (RX)": "#f58518",
}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def main() -> None:
    dts, _ = load_view("device_timeseries_view")
    pf, _ = load_view("phase_fact_view")

    dts = dts[dts["run_id"].astype(str).isin(RUN_IDS)].copy()
    pf = pf[pf["run_id"].astype(str).isin(RUN_IDS)].copy()
    dts_before = dts.copy()
    pf_before = pf.copy()
    dts = apply_analysis_ok(dts)
    pf = apply_analysis_ok(pf)
    print(f"dts_filtering={explain_filtering(dts_before, dts)}")
    print(f"pf_filtering={explain_filtering(pf_before, pf)}")

    # Training phase starts from phase_fact_view.
    starts = pf[["run_id", "global_step_canonical", "phase_name", "phase_start_ts"]].copy()
    starts["phase_name"] = starts["phase_name"].astype(str).str.lower()
    starts = starts[starts["phase_name"] == "training"].copy()
    starts["phase_start_ts"] = pd.to_numeric(starts["phase_start_ts"], errors="coerce")
    starts = starts.dropna(subset=["phase_start_ts"]).copy()
    starts = starts.rename(columns={"phase_start_ts": "training_start_ts_ns"})

    # NVML GPU periodic rows in training.
    g = dts.copy()
    g = g[g["phase_name"].astype(str).str.lower() == "training"].copy()
    if "device_kind" in g.columns:
        g = g[g["device_kind"].astype(str).str.lower() == "gpu"].copy()
    if "source" in g.columns:
        g = g[g["source"].astype(str).str.lower() == "nvml"].copy()
    g["ts_monotonic_ns"] = pd.to_numeric(g["ts_monotonic_ns"], errors="coerce")
    g["pcie_tx_bytes_s"] = pd.to_numeric(g["pcie_tx_bytes_s"], errors="coerce")
    g["pcie_rx_bytes_s"] = pd.to_numeric(g["pcie_rx_bytes_s"], errors="coerce")
    g = g.dropna(subset=["ts_monotonic_ns", "pcie_tx_bytes_s", "pcie_rx_bytes_s"]).copy()

    merged = g.merge(
        starts[["run_id", "global_step_canonical", "training_start_ts_ns"]],
        on=["run_id", "global_step_canonical"],
        how="inner",
    )
    merged = merged[merged["ts_monotonic_ns"] >= merged["training_start_ts_ns"]].copy()
    merged["start_offset_s"] = (merged["ts_monotonic_ns"] - merged["training_start_ts_ns"]) / 1_000_000_000.0
    merged = merged[merged["start_offset_s"] >= 0].copy()

    if merged.empty:
        raise ValueError("No periodic samples at/after training START for selected runs.")

    # H200 runs may poll all GPUs while only 2 are active.
    # Select active devices per run using training utilization, then keep top-2 for H200 runs.
    g_active = g.copy()
    g_active["gpu_util_pct"] = pd.to_numeric(g_active.get("gpu_util_pct"), errors="coerce")
    util_by_dev = (
        g_active.groupby(["run_id", "device_id"], dropna=False)["gpu_util_pct"]
        .median()
        .reset_index(name="median_gpu_util_pct")
        .fillna({"median_gpu_util_pct": 0.0})
    )
    active_rows = []
    for run_id, grp in util_by_dev.groupby("run_id", dropna=False):
        platform = _platform_from_run_id(str(run_id))
        gsort = grp.sort_values("median_gpu_util_pct", ascending=False).copy()
        if platform == "2xH200":
            gsel = gsort.head(2)
        else:
            # Keep likely active devices for other platforms.
            gsel = gsort[gsort["median_gpu_util_pct"] > 0]
            if gsel.empty:
                gsel = gsort
        for _, r in gsel.iterrows():
            active_rows.append({"run_id": run_id, "device_id": r["device_id"], "median_gpu_util_pct": float(r["median_gpu_util_pct"])})
    active_devices = pd.DataFrame(active_rows).drop_duplicates(["run_id", "device_id"])
    print("active device selection:")
    print(active_devices.sort_values(["run_id", "median_gpu_util_pct"], ascending=[True, False]).to_string(index=False))

    merged = merged.merge(active_devices[["run_id", "device_id"]], on=["run_id", "device_id"], how="inner")
    if merged.empty:
        raise ValueError("No rows remain after active-device filtering.")

    # Keep first sample at/after START for each run-step-device.
    merged = merged.sort_values(["run_id", "global_step_canonical", "device_id", "ts_monotonic_ns"]).copy()
    first = (
        merged.groupby(["run_id", "global_step_canonical", "device_id"], dropna=False)
        .head(1)
        .copy()
    )
    first["platform"] = first["run_id"].map(_platform_from_run_id)
    first = first[first["platform"].isin(PLATFORM_ORDER)].copy()

    # Aggregate across GPUs to run-step level boundary snapshot.
    step_boundary = (
        first.groupby(["run_id", "global_step_canonical", "platform"], dropna=False)
        .agg(
            pcie_tx_bytes_s_start=("pcie_tx_bytes_s", "mean"),
            pcie_rx_bytes_s_start=("pcie_rx_bytes_s", "mean"),
            offset_s_mean=("start_offset_s", "mean"),
            n_gpu_samples=("device_id", "nunique"),
        )
        .reset_index()
    )
    step_boundary["rx_to_tx_ratio"] = step_boundary["pcie_rx_bytes_s_start"] / step_boundary["pcie_tx_bytes_s_start"].replace(0, np.nan)

    print("boundary sample timing stats (s from training START):")
    print(step_boundary.groupby("platform")["offset_s_mean"].describe().to_string())
    print("run-step boundary summary:")
    print(
        step_boundary.groupby("platform")[["pcie_tx_bytes_s_start", "pcie_rx_bytes_s_start", "rx_to_tx_ratio"]]
        .median()
        .rename(columns={
            "pcie_tx_bytes_s_start": "median_tx_bytes_s",
            "pcie_rx_bytes_s_start": "median_rx_bytes_s",
            "rx_to_tx_ratio": "median_rx_to_tx_ratio",
        })
        .to_string()
    )

    # Long format for split direction boxplots.
    long = pd.concat(
        [
            step_boundary[["platform", "pcie_tx_bytes_s_start"]]
            .rename(columns={"pcie_tx_bytes_s_start": "pcie_bytes_s"})
            .assign(direction="Device -> Host (TX)"),
            step_boundary[["platform", "pcie_rx_bytes_s_start"]]
            .rename(columns={"pcie_rx_bytes_s_start": "pcie_bytes_s"})
            .assign(direction="Host -> Device (RX)"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # Panel A: TX vs RX by platform (boxplot).
    ax = axes[0]
    x = np.arange(len(PLATFORM_ORDER))
    w = 0.34
    positions = []
    data = []
    colors = []
    hatches = []
    labels = []
    for i, platform in enumerate(PLATFORM_ORDER):
        for direction, dx in [("Device -> Host (TX)", -0.5), ("Host -> Device (RX)", 0.5)]:
            vals = pd.to_numeric(
                long.loc[(long["platform"] == platform) & (long["direction"] == direction), "pcie_bytes_s"],
                errors="coerce",
            ).dropna()
            if vals.empty:
                continue
            positions.append(i + dx * w)
            data.append(vals.to_numpy())
            colors.append(DIRECTION_COLORS[direction])
            hatches.append("" if platform == "2xH200" else "//")
            labels.append((platform, direction))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=w * 0.85,
        showfliers=False,
        patch_artist=True,
        boxprops={"edgecolor": "black", "linewidth": 0.7},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.7},
        capprops={"color": "black", "linewidth": 0.7},
    )
    for patch, c, h in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(PLATFORM_ORDER)
    ax.set_ylabel("PCIe bytes/s at Training START")
    ax.set_title("A) Directional PCIe Throughput at Training START")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    # Panel B: RX/TX ratio by platform.
    ax2 = axes[1]
    ratio_data = [
        pd.to_numeric(step_boundary.loc[step_boundary["platform"] == p, "rx_to_tx_ratio"], errors="coerce").dropna().to_numpy()
        for p in PLATFORM_ORDER
    ]
    bp2 = ax2.boxplot(
        ratio_data,
        positions=[0, 1],
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"edgecolor": "black", "linewidth": 0.7},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.7},
        capprops={"color": "black", "linewidth": 0.7},
    )
    for patch, p in zip(bp2["boxes"], PLATFORM_ORDER):
        patch.set_facecolor(PLATFORM_COLORS[p])
        patch.set_alpha(0.45)
    ax2.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(PLATFORM_ORDER)
    ax2.set_ylabel("RX / TX ratio at Training START")
    ax2.set_title("B) Host->Device vs Device->Host Directionality")
    ax2.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    # Median annotations.
    for i, p in enumerate(PLATFORM_ORDER):
        med_tx = step_boundary.loc[step_boundary["platform"] == p, "pcie_tx_bytes_s_start"].median()
        med_rx = step_boundary.loc[step_boundary["platform"] == p, "pcie_rx_bytes_s_start"].median()
        med_ratio = step_boundary.loc[step_boundary["platform"] == p, "rx_to_tx_ratio"].median()
        ax.text(i, max(med_tx, med_rx) * 1.02 if np.isfinite(max(med_tx, med_rx)) else 0, f"med TX={med_tx:.2e}\nmed RX={med_rx:.2e}", ha="center", va="bottom", fontsize=8)
        ax2.text(i, med_ratio, f"{med_ratio:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DIRECTION_COLORS["Device -> Host (TX)"], edgecolor="black", alpha=0.45, label="Device -> Host (TX)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIRECTION_COLORS["Host -> Device (RX)"], edgecolor="black", alpha=0.45, label="Host -> Device (RX)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="", label="2xH200"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//", label="4xA100"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        title="Direction (color) and Platform (hatch)",
    )

    fig.suptitle("PCIe Traffic Directionality by Phase Boundary (Training START)", y=0.99)
    fig.text(0.5, 0.01, "Boundary sample = first periodic GPU sample at/after training START per device.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
