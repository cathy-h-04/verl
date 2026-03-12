"""Clear summary view for MFU/throttling by platform x policy.

Replaces dense scatter with bar-chart aggregates:
1) SW power-cap hit rate (% periodic samples with thr_sw_power_cap=True)
2) Median SM utilization (%)
3) Median GPU power (mW)
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


OUTPATH = Path("plots/out/figures/tier1/mfu_and_throttling_summary_bars_platform_policy_selected_runs.png")

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
PLATFORM_COLORS = {"2xH200": "#4c78a8", "4xA100": "#f58518"}
ACTIVE_SAMPLE_SM_UTIL_THRESHOLD = 0.0


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def _norm_policy(x: str) -> str:
    return str(x).strip().lower().replace("remx", "remax")


def _active_gpu_count_from_run_id(run_id: str) -> int:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return 2
    if "4gpu_a100" in rid:
        return 4
    return 1


def _safe_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.fillna(False).astype(bool)


def _grouped_bars(ax: plt.Axes, summary: pd.DataFrame, value_col: str, ylabel: str, title: str) -> None:
    x = np.arange(len(POLICY_ORDER))
    w = 0.36
    for i, platform in enumerate(PLATFORM_ORDER):
        vals = []
        for policy in POLICY_ORDER:
            row = summary[(summary["policy_norm"] == policy) & (summary["platform"] == platform)]
            vals.append(float(row[value_col].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + (i - 0.5) * w,
            vals,
            width=w,
            color=PLATFORM_COLORS[platform],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=platform,
        )
        for xi, yi in zip(x + (i - 0.5) * w, vals):
            if pd.isna(yi):
                continue
            ax.text(xi, yi, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POLICY_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)


def _grouped_boxplots(ax: plt.Axes, df: pd.DataFrame, value_col: str, ylabel: str, title: str) -> None:
    x = np.arange(len(POLICY_ORDER))
    w = 0.34
    positions: list[float] = []
    data: list[np.ndarray] = []
    colors: list[str] = []

    for i, policy in enumerate(POLICY_ORDER):
        for platform in PLATFORM_ORDER:
            vals = pd.to_numeric(
                df.loc[(df["policy_norm"] == policy) & (df["platform"] == platform), value_col],
                errors="coerce",
            ).dropna()
            if vals.empty:
                continue
            positions.append(i + (-0.5 if platform == "2xH200" else 0.5) * w)
            data.append(vals.to_numpy())
            colors.append(PLATFORM_COLORS[platform])

    if data:
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
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.38)

    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POLICY_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)


def main() -> None:
    dts, _ = load_view("device_timeseries_view")
    runs, _ = load_view("run_summary_view")

    dts = dts[dts["run_id"].astype(str).isin(RUN_IDS)].copy()
    before = dts.copy()
    dts = apply_analysis_ok(dts)
    print(f"filtering={explain_filtering(before, dts)}")

    policy_map = runs[runs["run_id"].astype(str).isin(RUN_IDS)][["run_id", "policy"]].drop_duplicates("run_id").copy()
    policy_map["policy_norm"] = policy_map["policy"].map(_norm_policy)

    df = dts.merge(policy_map[["run_id", "policy_norm"]], on="run_id", how="inner")
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df = df[df["platform"].isin(PLATFORM_ORDER) & df["policy_norm"].isin(POLICY_ORDER)].copy()

    if "device_kind" in df.columns:
        df = df[df["device_kind"].astype(str).str.lower() == "gpu"].copy()
    if "source" in df.columns:
        df = df[df["source"].astype(str).str.lower() == "nvml"].copy()

    df["sm_util_pct"] = pd.to_numeric(df["sm_util_pct"], errors="coerce")
    df["gpu_power_mw"] = pd.to_numeric(df["gpu_power_mw"], errors="coerce")
    df["thr_sw_power_cap"] = _safe_bool(df["thr_sw_power_cap"])
    df = df.dropna(subset=["sm_util_pct", "gpu_power_mw"]).copy()
    df = df[(df["sm_util_pct"] >= 0) & (df["sm_util_pct"] <= 100) & (df["gpu_power_mw"] >= 0)].copy()

    # Keep only active device IDs per run (2 for 2xH200, 4 for 4xA100).
    df["_active_gpu_count"] = df["run_id"].map(_active_gpu_count_from_run_id).astype(int)
    before_device_filter = len(df)
    device_rank = (
        df.groupby(["run_id", "device_id"], dropna=False)
        .agg(
            n_samples=("sm_util_pct", "size"),
            active_rate=("sm_util_pct", lambda s: float((s > ACTIVE_SAMPLE_SM_UTIL_THRESHOLD).mean())),
            mean_sm_util=("sm_util_pct", "mean"),
            mean_gpu_power_mw=("gpu_power_mw", "mean"),
        )
        .reset_index()
    )
    expected = df[["run_id", "_active_gpu_count"]].drop_duplicates("run_id")
    device_rank = device_rank.merge(expected, on="run_id", how="left")
    device_rank = device_rank.sort_values(
        ["run_id", "active_rate", "mean_sm_util", "mean_gpu_power_mw"],
        ascending=[True, False, False, False],
    )
    device_rank["_rank"] = device_rank.groupby("run_id", dropna=False).cumcount()
    keep_devices = device_rank[device_rank["_rank"] < device_rank["_active_gpu_count"]][["run_id", "device_id"]].copy()
    df = df.merge(keep_devices.assign(_keep=True), on=["run_id", "device_id"], how="inner")
    print(
        "active_device_filter="
        f"{{'rows_before': {before_device_filter}, 'rows_after': {len(df)}, "
        f"'rows_removed': {before_device_filter - len(df)}}}"
    )
    print(
        "selected_active_devices:\n"
        + keep_devices.sort_values(["run_id", "device_id"]).to_string(index=False)
    )

    # During-active-work filter: keep samples where selected GPUs are doing work.
    before_sample_active = len(df)
    df = df[df["sm_util_pct"] > ACTIVE_SAMPLE_SM_UTIL_THRESHOLD].copy()
    print(
        "active_sample_filter="
        f"{{'sm_util_threshold': {ACTIVE_SAMPLE_SM_UTIL_THRESHOLD}, 'rows_before': {before_sample_active}, "
        f"'rows_after': {len(df)}, 'rows_removed': {before_sample_active - len(df)}}}"
    )
    df = df.drop(columns=["_active_gpu_count"])

    if df.empty:
        raise ValueError("No valid periodic GPU rows after filtering.")

    summary = (
        df.groupby(["platform", "policy_norm"], dropna=False)
        .agg(
            n_samples=("sm_util_pct", "size"),
            cap_rate_pct=("thr_sw_power_cap", lambda s: 100.0 * float(_safe_bool(s).mean())),
            sm_util_median=("sm_util_pct", "median"),
            gpu_power_mw_median=("gpu_power_mw", "median"),
            capped_n=("thr_sw_power_cap", lambda s: int(_safe_bool(s).sum())),
        )
        .reset_index()
        .sort_values(["platform", "policy_norm"])
    )
    print("summary table:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))

    _grouped_bars(
        axes[0],
        summary,
        value_col="cap_rate_pct",
        ylabel="SW cap hit rate (%)",
        title="A) Throttling Frequency",
    )
    _grouped_boxplots(
        axes[1],
        df,
        value_col="sm_util_pct",
        ylabel="sm_util_pct",
        title="B) Compute Utilization (Active GPUs Only)",
    )
    _grouped_boxplots(
        axes[2],
        df,
        value_col="gpu_power_mw",
        ylabel="gpu_power_mW",
        title="C) Power Draw (Active GPUs Only)",
    )

    # Per-policy tax annotation in panel A (relative change from 2xH200 -> 4xA100).
    x = np.arange(len(POLICY_ORDER))
    for i, policy in enumerate(POLICY_ORDER):
        r2 = summary[(summary["policy_norm"] == policy) & (summary["platform"] == "2xH200")]
        r4 = summary[(summary["policy_norm"] == policy) & (summary["platform"] == "4xA100")]
        if r2.empty or r4.empty:
            continue
        v2 = float(r2["cap_rate_pct"].iloc[0])
        v4 = float(r4["cap_rate_pct"].iloc[0])
        if v2 <= 0:
            continue
        pct = 100.0 * (v4 - v2) / v2
        y = max(v2, v4)
        axes[0].text(
            i,
            y * 1.12 + 0.2,
            f"{pct:+.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.2},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.97), title="Platform")
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.suptitle("Plot 1.2 (Clear View): MFU/Throttling Summary by Platform x Policy (Active GPUs)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
