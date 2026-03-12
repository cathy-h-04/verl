"""Straggler tail distribution (inter-GPU variance) by platform.

Requested metric pair:
- X: rollout/straggler_ratio
- Y: timing_dist_s/update_actor/imbalance

Dataset note:
- timing_dist_s/update_actor/imbalance is not present in current DATASETS.
- Fallback used here: timing_per_token_ms/update_actor (explicitly labeled).
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


OUTPATH = Path("plots/out/figures/tier1/straggler_tail_distribution_inter_gpu_variance_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PLATFORM_ORDER = ["2xH200", "4xA100"]
PLATFORM_COLORS = {
    "2xH200": "#4c78a8",
    "4xA100": "#f58518",
}

X_KEY = "rollout/straggler_ratio"
Y_KEY_FALLBACK = "timing_per_token_ms/update_actor"


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def main() -> None:
    sf, _ = load_view("step_fact_view")
    ml, _ = load_view("step_metrics_long")

    sf = sf[sf["run_id"].astype(str).isin(RUN_IDS)].copy()
    before = sf.copy()
    sf = apply_analysis_ok(sf)
    print(f"filtering={explain_filtering(before, sf)}")
    if sf.empty:
        raise ValueError("No step rows remain after filtering.")

    step_keys = sf[["run_id", "global_step_canonical"]].drop_duplicates().copy()

    m = ml[ml["run_id"].astype(str).isin(RUN_IDS)].copy()
    m = m[m["metric_key"].isin([X_KEY, Y_KEY_FALLBACK])].copy()
    m["metric_value_float"] = pd.to_numeric(m["metric_value_float"], errors="coerce")
    m = m.dropna(subset=["metric_value_float"]).copy()

    wide = (
        m.pivot_table(
            index=["run_id", "global_step_canonical"],
            columns="metric_key",
            values="metric_value_float",
            aggfunc="mean",
        )
        .reset_index()
    )
    for col in [X_KEY, Y_KEY_FALLBACK]:
        if col not in wide.columns:
            wide[col] = np.nan

    df = step_keys.merge(
        wide[["run_id", "global_step_canonical", X_KEY, Y_KEY_FALLBACK]],
        on=["run_id", "global_step_canonical"],
        how="inner",
    )
    df = df.dropna(subset=[X_KEY, Y_KEY_FALLBACK]).copy()
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df = df[df["platform"].isin(PLATFORM_ORDER)].copy()
    if df.empty:
        raise ValueError("No points available after metric join.")

    # Guard to reasonable ranges for cleaner tails view.
    df = df[(df[X_KEY] >= 0) & (df[Y_KEY_FALLBACK] >= 0)].copy()

    summary_rows = []
    for platform in PLATFORM_ORDER:
        s = df[df["platform"] == platform]
        if s.empty:
            continue
        summary_rows.append(
            {
                "platform": platform,
                "n_steps": int(len(s)),
                "straggler_mean": float(s[X_KEY].mean()),
                "straggler_p90": float(s[X_KEY].quantile(0.90)),
                "straggler_p95": float(s[X_KEY].quantile(0.95)),
                "update_actor_tpt_ms_mean": float(s[Y_KEY_FALLBACK].mean()),
                "update_actor_tpt_ms_p90": float(s[Y_KEY_FALLBACK].quantile(0.90)),
                "update_actor_tpt_ms_p95": float(s[Y_KEY_FALLBACK].quantile(0.95)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("platform")
    print("tail summary by platform:")
    print(summary.to_string(index=False))

    run_means = (
        df.groupby(["run_id", "platform"], dropna=False)[[X_KEY, Y_KEY_FALLBACK]]
        .mean()
        .reset_index()
        .rename(columns={X_KEY: "run_mean_x", Y_KEY_FALLBACK: "run_mean_y"})
        .sort_values(["platform", "run_id"])
    )
    print("run means:")
    print(run_means.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharex=True, sharey=True)
    for ax, platform in zip(axes, PLATFORM_ORDER):
        s = df[df["platform"] == platform].copy()
        if s.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.scatter(
            s[X_KEY],
            s[Y_KEY_FALLBACK],
            s=20,
            alpha=0.25,
            color=PLATFORM_COLORS[platform],
            edgecolors="none",
            zorder=1,
        )

        # Tail reference lines (p90).
        x90 = float(s[X_KEY].quantile(0.90))
        y90 = float(s[Y_KEY_FALLBACK].quantile(0.90))
        ax.axvline(x90, color=PLATFORM_COLORS[platform], linestyle="--", linewidth=1.1, alpha=0.9, zorder=2)
        ax.axhline(y90, color=PLATFORM_COLORS[platform], linestyle="--", linewidth=1.1, alpha=0.9, zorder=2)

        rm = run_means[run_means["platform"] == platform]
        if not rm.empty:
            ax.scatter(
                rm["run_mean_x"],
                rm["run_mean_y"],
                s=90,
                marker="D",
                color=PLATFORM_COLORS[platform],
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )

        n = int(len(s))
        ax.text(
            0.03,
            0.97,
            f"n={n}\nP90 x={x90:.3f}\nP90 y={y90:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85, "pad": 0.25},
        )
        ax.set_title(platform, pad=8)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    axes[0].set_ylabel("timing_per_token_ms/update_actor")
    for ax in axes:
        ax.set_xlabel("rollout/straggler_ratio")

    fig.suptitle("The Straggler Tail Distribution (Platform Comparison)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 1))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
