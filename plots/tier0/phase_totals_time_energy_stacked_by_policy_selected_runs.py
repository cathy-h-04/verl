"""Total phase time and energy by policy/platform (stacked bars).

Companion to the phase-dominance plots, but uses absolute totals:
- Panel A: total phase_time_s
- Panel B: total total_energy_j

Stacks: rollout / training / rl_policy
Groups: policy x platform
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


OUTPATH = Path("plots/out/figures/tier0/phase_totals_time_energy_stacked_by_policy_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

POLICY_ORDER = ["ppo", "remax", "grpo"]
PLATFORM_ORDER = ["2xH200", "4xA100"]
PHASE_ORDER = ["rollout", "training", "rl_policy"]

PHASE_COLORS = {
    "rollout": "#4c78a8",
    "training": "#f58518",
    "rl_policy": "#54a24b",
}
PLATFORM_HATCH = {
    "2xH200": "",
    "4xA100": "//",
}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def _phase_bucket(name: str) -> str:
    s = str(name).strip().lower()
    if s in PHASE_ORDER:
        return s
    return "other"


def _norm_policy(x: str) -> str:
    return str(x).strip().lower().replace("remx", "remax")


def _plot_stacked_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(POLICY_ORDER))
    w = 0.36

    per_policy_phase_pct: dict[tuple[str, str], float | None] = {}
    for policy in POLICY_ORDER:
        for phase in PHASE_ORDER:
            row2 = data[
                (data["policy_norm"] == policy)
                & (data["platform"] == "2xH200")
                & (data["phase_bucket"] == phase)
            ]
            row4 = data[
                (data["policy_norm"] == policy)
                & (data["platform"] == "4xA100")
                & (data["phase_bucket"] == phase)
            ]
            v2 = float(row2[value_col].iloc[0]) if not row2.empty else 0.0
            v4 = float(row4[value_col].iloc[0]) if not row4.empty else 0.0
            per_policy_phase_pct[(policy, phase)] = (100.0 * (v4 - v2) / v2) if v2 > 0 else None

    for i, platform in enumerate(PLATFORM_ORDER):
        xloc = x + (i - 0.5) * w
        bottom = np.zeros(len(POLICY_ORDER), dtype=float)
        for phase in PHASE_ORDER:
            vals = []
            for policy in POLICY_ORDER:
                row = data[
                    (data["policy_norm"] == policy)
                    & (data["platform"] == platform)
                    & (data["phase_bucket"] == phase)
                ]
                vals.append(float(row[value_col].iloc[0]) if not row.empty else 0.0)
            vals_arr = np.array(vals, dtype=float)
            ax.bar(
                xloc,
                vals_arr,
                bottom=bottom,
                width=w * 0.92,
                color=PHASE_COLORS[phase],
                edgecolor="black",
                linewidth=0.6,
                hatch=PLATFORM_HATCH[platform],
                alpha=0.92,
            )
            # Put percent-diff annotation inside each segment.
            for j, (xi, yi, bi, policy) in enumerate(zip(xloc, vals_arr, bottom, POLICY_ORDER)):
                if yi <= 0:
                    continue
                if platform != "4xA100":
                    continue
                pct = per_policy_phase_pct.get((policy, phase))
                if pct is None:
                    continue
                # Skip ultra-thin segments to avoid illegible overlaps.
                if yi < 0.06 * max(1.0, np.nanmax(vals_arr)):
                    continue
                ax.text(
                    xi,
                    bi + yi * 0.5,
                    f"{pct:+.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                    fontweight="bold",
                    bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.28, "pad": 0.12},
                )
            bottom += vals_arr

        for xi, yi in zip(xloc, bottom):
            ax.text(
                xi,
                yi,
                f"{yi:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.18},
            )

    # Expand y-limit to keep top totals visible.
    y_max_seen = 0.0
    for i, policy in enumerate(POLICY_ORDER):
        total_2 = 0.0
        total_4 = 0.0
        for phase in PHASE_ORDER:
            row2 = data[
                (data["policy_norm"] == policy)
                & (data["platform"] == "2xH200")
                & (data["phase_bucket"] == phase)
            ]
            row4 = data[
                (data["policy_norm"] == policy)
                & (data["platform"] == "4xA100")
                & (data["phase_bucket"] == phase)
            ]
            v2 = float(row2[value_col].iloc[0]) if not row2.empty else 0.0
            v4 = float(row4[value_col].iloc[0]) if not row4.empty else 0.0
            total_2 += v2
            total_4 += v4
        y_anchor = max(total_2, total_4)
        y_max_seen = max(y_max_seen, y_anchor)

    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POLICY_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_ylim(0, y_max_seen * 1.30 if y_max_seen > 0 else 1.0)


def main() -> None:
    pf, _ = load_view("phase_fact_view")
    if pf.empty:
        raise ValueError("phase_fact_view is empty.")

    required = ["run_id", "policy", "phase_name", "phase_time_s", "total_energy_j"]
    missing = [c for c in required if c not in pf.columns]
    if missing:
        raise ValueError(f"phase_fact_view missing required columns: {missing}")

    df = pf[pf["run_id"].astype(str).isin(RUN_IDS)].copy()
    if df.empty:
        raise ValueError("No phase_fact_view rows found for selected RUN_IDS.")

    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")

    df["policy_norm"] = df["policy"].map(_norm_policy)
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df["phase_bucket"] = df["phase_name"].map(_phase_bucket)
    df["phase_time_s"] = pd.to_numeric(df["phase_time_s"], errors="coerce")
    df["total_energy_j"] = pd.to_numeric(df["total_energy_j"], errors="coerce")
    df = df[
        df["policy_norm"].isin(POLICY_ORDER)
        & df["platform"].isin(PLATFORM_ORDER)
        & df["phase_bucket"].isin(PHASE_ORDER)
    ].copy()
    df = df.dropna(subset=["phase_time_s", "total_energy_j"]).copy()
    if df.empty:
        raise ValueError("No valid rows after filtering and column normalization.")

    agg = (
        df.groupby(["policy_norm", "platform", "phase_bucket"], dropna=False)
        .agg(
            total_phase_time_s=("phase_time_s", "sum"),
            total_phase_energy_j=("total_energy_j", "sum"),
            n_rows=("run_id", "size"),
        )
        .reset_index()
        .sort_values(["policy_norm", "platform", "phase_bucket"])
    )
    print("aggregates:")
    print(agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6))
    _plot_stacked_panel(
        axes[0],
        agg,
        value_col="total_phase_time_s",
        ylabel="Total phase time (s)",
        title="A) Total Time Across Phases",
    )
    _plot_stacked_panel(
        axes[1],
        agg,
        value_col="total_phase_energy_j",
        ylabel="Total phase energy (J)",
        title="B) Total Energy Across Phases",
    )

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[p], edgecolor="black", label=p) for p in PHASE_ORDER
    ]
    platform_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=PLATFORM_HATCH[p], label=p) for p in PLATFORM_ORDER
    ]
    leg1 = fig.legend(
        phase_handles,
        [p for p in PHASE_ORDER],
        title="Phase (color)",
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.34, 0.97),
    )
    fig.add_artist(leg1)
    fig.legend(
        platform_handles,
        [p for p in PLATFORM_ORDER],
        title="Platform (hatch)",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.77, 0.97),
    )

    fig.suptitle("Total Time and Energy by Phase Across Policies (Selected Runs)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
