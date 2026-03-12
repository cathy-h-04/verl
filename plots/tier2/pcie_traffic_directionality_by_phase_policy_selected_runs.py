"""PCIe traffic directionality split by phase and policy (selected runs).

Design:
- Grid facets: rows=phase, cols=policy
- Within each panel: TX/RX distributions split by platform
- Boundary sampling: first periodic GPU sample at/after phase START per device
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier2/pcie_traffic_directionality_by_phase_policy_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PHASE_ORDER = ["rollout", "training", "rl_policy"]
POLICY_ORDER = ["ppo", "remax", "grpo"]
PLATFORM_ORDER = ["2xH200", "4xA100"]
PLATFORM_COLORS = {"2xH200": "#4c78a8", "4xA100": "#f58518"}
DIRECTION_ORDER = ["Device -> Host (TX)", "Host -> Device (RX)"]
DIRECTION_HATCH = {"Device -> Host (TX)": "", "Host -> Device (RX)": "//"}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def _target_active_gpu_count(run_id: str) -> int:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return 2
    if "4gpu_a100" in rid:
        return 4
    return 8


def _norm_policy(x: str) -> str:
    return str(x).strip().lower().replace("remx", "remax")


def _build_boundary_snapshot() -> pd.DataFrame:
    dts, _ = load_view("device_timeseries_view")
    pf, _ = load_view("phase_fact_view")
    runs, _ = load_view("run_summary_view")

    dts = dts[dts["run_id"].astype(str).isin(RUN_IDS)].copy()
    pf = pf[pf["run_id"].astype(str).isin(RUN_IDS)].copy()
    dts_before = dts.copy()
    pf_before = pf.copy()
    dts = apply_analysis_ok(dts)
    pf = apply_analysis_ok(pf)
    print(f"dts_filtering={explain_filtering(dts_before, dts)}")
    print(f"pf_filtering={explain_filtering(pf_before, pf)}")

    policy_map = runs[runs["run_id"].astype(str).isin(RUN_IDS)][["run_id", "policy"]].drop_duplicates("run_id").copy()
    policy_map["policy_norm"] = policy_map["policy"].map(_norm_policy)

    starts = pf[["run_id", "global_step_canonical", "phase_name", "phase_start_ts"]].copy()
    starts["phase_name"] = starts["phase_name"].astype(str).str.lower()
    starts = starts[starts["phase_name"].isin(PHASE_ORDER)].copy()
    starts["phase_start_ts"] = pd.to_numeric(starts["phase_start_ts"], errors="coerce")
    starts = starts.dropna(subset=["phase_start_ts"]).rename(columns={"phase_start_ts": "phase_start_ts_ns"})

    g = dts.copy()
    if "device_kind" in g.columns:
        g = g[g["device_kind"].astype(str).str.lower() == "gpu"].copy()
    if "source" in g.columns:
        g = g[g["source"].astype(str).str.lower() == "nvml"].copy()
    g["phase_name"] = g["phase_name"].astype(str).str.lower()
    g = g[g["phase_name"].isin(PHASE_ORDER)].copy()
    g["ts_monotonic_ns"] = pd.to_numeric(g["ts_monotonic_ns"], errors="coerce")
    g["gpu_util_pct"] = pd.to_numeric(g["gpu_util_pct"], errors="coerce")
    g["pcie_tx_bytes_s"] = pd.to_numeric(g["pcie_tx_bytes_s"], errors="coerce")
    g["pcie_rx_bytes_s"] = pd.to_numeric(g["pcie_rx_bytes_s"], errors="coerce")
    g = g.dropna(subset=["ts_monotonic_ns", "pcie_tx_bytes_s", "pcie_rx_bytes_s"]).copy()

    # Active-device selection: top-K by median util per run.
    util_by_dev = (
        g.groupby(["run_id", "device_id"], dropna=False)["gpu_util_pct"]
        .median()
        .reset_index(name="median_gpu_util_pct")
        .fillna({"median_gpu_util_pct": 0.0})
    )
    active_rows = []
    for run_id, grp in util_by_dev.groupby("run_id", dropna=False):
        k = _target_active_gpu_count(str(run_id))
        gsel = grp.sort_values("median_gpu_util_pct", ascending=False).head(k)
        for _, r in gsel.iterrows():
            active_rows.append({"run_id": run_id, "device_id": r["device_id"], "median_gpu_util_pct": float(r["median_gpu_util_pct"])})
    active_devices = pd.DataFrame(active_rows).drop_duplicates(["run_id", "device_id"])
    print("active device selection:")
    print(active_devices.sort_values(["run_id", "median_gpu_util_pct"], ascending=[True, False]).to_string(index=False))

    merged = g.merge(
        starts[["run_id", "global_step_canonical", "phase_name", "phase_start_ts_ns"]],
        on=["run_id", "global_step_canonical", "phase_name"],
        how="inner",
    )
    merged = merged.merge(active_devices[["run_id", "device_id"]], on=["run_id", "device_id"], how="inner")
    merged = merged[merged["ts_monotonic_ns"] >= merged["phase_start_ts_ns"]].copy()
    merged["start_offset_s"] = (merged["ts_monotonic_ns"] - merged["phase_start_ts_ns"]) / 1_000_000_000.0
    merged = merged[merged["start_offset_s"] >= 0].copy()
    if merged.empty:
        raise ValueError("No periodic rows after start-boundary + active-device filtering.")

    # First sample at/after phase start for each device.
    merged = merged.sort_values(["run_id", "global_step_canonical", "phase_name", "device_id", "ts_monotonic_ns"]).copy()
    first = merged.groupby(["run_id", "global_step_canonical", "phase_name", "device_id"], dropna=False).head(1).copy()
    first["platform"] = first["run_id"].map(_platform_from_run_id)
    first = first[first["platform"].isin(PLATFORM_ORDER)].copy()
    first = first.merge(policy_map[["run_id", "policy_norm"]], on="run_id", how="left")
    first = first[first["policy_norm"].isin(POLICY_ORDER)].copy()

    # Aggregate across active GPUs to run-step-phase boundary snapshot.
    step_phase = (
        first.groupby(["run_id", "global_step_canonical", "phase_name", "policy_norm", "platform"], dropna=False)
        .agg(
            pcie_tx_bytes_s_start=("pcie_tx_bytes_s", "mean"),
            pcie_rx_bytes_s_start=("pcie_rx_bytes_s", "mean"),
            n_gpu_samples=("device_id", "nunique"),
            offset_s_mean=("start_offset_s", "mean"),
        )
        .reset_index()
    )
    return step_phase


def main() -> None:
    df = _build_boundary_snapshot()
    print("boundary snapshot summary:")
    print(
        df.groupby(["phase_name", "policy_norm", "platform"], dropna=False)[["pcie_tx_bytes_s_start", "pcie_rx_bytes_s_start"]]
        .median()
        .reset_index()
        .to_string(index=False)
    )

    long = pd.concat(
        [
            df[["phase_name", "policy_norm", "platform", "pcie_tx_bytes_s_start"]]
            .rename(columns={"pcie_tx_bytes_s_start": "pcie_bytes_s"})
            .assign(direction="Device -> Host (TX)"),
            df[["phase_name", "policy_norm", "platform", "pcie_rx_bytes_s_start"]]
            .rename(columns={"pcie_rx_bytes_s_start": "pcie_bytes_s"})
            .assign(direction="Host -> Device (RX)"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(len(PHASE_ORDER), len(POLICY_ORDER), figsize=(15.5, 11), sharey=False)
    for r, phase in enumerate(PHASE_ORDER):
        for c, policy in enumerate(POLICY_ORDER):
            ax = axes[r, c]
            panel = long[(long["phase_name"] == phase) & (long["policy_norm"] == policy)].copy()
            if panel.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue

            xbase = np.arange(len(DIRECTION_ORDER))
            w = 0.34
            positions = []
            data = []
            colors = []
            hatches = []
            for i, direction in enumerate(DIRECTION_ORDER):
                for platform, dx in [("2xH200", -0.5), ("4xA100", 0.5)]:
                    vals = pd.to_numeric(
                        panel.loc[(panel["direction"] == direction) & (panel["platform"] == platform), "pcie_bytes_s"],
                        errors="coerce",
                    ).dropna()
                    if vals.empty:
                        continue
                    positions.append(i + dx * w)
                    data.append(vals.to_numpy())
                    colors.append(PLATFORM_COLORS[platform])
                    hatches.append(DIRECTION_HATCH[direction])

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=w * 0.82,
                showfliers=False,
                patch_artist=True,
                boxprops={"edgecolor": "black", "linewidth": 0.7},
                medianprops={"color": "black", "linewidth": 1.1},
                whiskerprops={"color": "black", "linewidth": 0.7},
                capprops={"color": "black", "linewidth": 0.7},
            )
            for patch, col, hatch in zip(bp["boxes"], colors, hatches):
                patch.set_facecolor(col)
                patch.set_alpha(0.45)
                patch.set_hatch(hatch)

            # RX/TX medians annotation (compact).
            med2_tx = panel[(panel["direction"] == DIRECTION_ORDER[0]) & (panel["platform"] == "2xH200")]["pcie_bytes_s"].median()
            med2_rx = panel[(panel["direction"] == DIRECTION_ORDER[1]) & (panel["platform"] == "2xH200")]["pcie_bytes_s"].median()
            med4_tx = panel[(panel["direction"] == DIRECTION_ORDER[0]) & (panel["platform"] == "4xA100")]["pcie_bytes_s"].median()
            med4_rx = panel[(panel["direction"] == DIRECTION_ORDER[1]) & (panel["platform"] == "4xA100")]["pcie_bytes_s"].median()
            r2 = (med2_rx / med2_tx) if pd.notna(med2_rx) and pd.notna(med2_tx) and med2_tx > 0 else np.nan
            r4 = (med4_rx / med4_tx) if pd.notna(med4_rx) and pd.notna(med4_tx) and med4_tx > 0 else np.nan
            n = int(len(panel) // 2)
            ax.text(
                0.03,
                0.97,
                f"n={n}\nRX/TX 2x={r2:.2f}\nRX/TX 4x={r4:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85, "pad": 0.2},
            )

            ax.set_xticks(xbase)
            ax.set_xticklabels(["TX", "RX"])
            if c == 0:
                ax.set_ylabel(f"{phase}\nPCIe bytes/s at phase START")
            if r == 0:
                ax.set_title(policy.upper(), pad=8)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.22)

    legend_handles = [
        Patch(facecolor=PLATFORM_COLORS["2xH200"], edgecolor="black", alpha=0.45, label="2xH200"),
        Patch(facecolor=PLATFORM_COLORS["4xA100"], edgecolor="black", alpha=0.45, label="4xA100"),
        Patch(facecolor="white", edgecolor="black", hatch=DIRECTION_HATCH["Device -> Host (TX)"], label="TX"),
        Patch(facecolor="white", edgecolor="black", hatch=DIRECTION_HATCH["Host -> Device (RX)"], label="RX"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        title="Platform (color) and Direction (hatch)",
    )
    fig.suptitle("PCIe Traffic Directionality by Phase and Policy (Boundary Snapshot)", y=0.995)
    fig.text(
        0.5,
        0.01,
        "Boundary snapshot: first periodic GPU sample at/after phase START, using active GPUs per run.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
