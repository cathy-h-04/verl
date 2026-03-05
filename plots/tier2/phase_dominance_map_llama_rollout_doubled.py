"""Phase dominance map for Llama PPO/ReMax: baseline vs doubled rollout tokens.

Panels:
- Left: baseline runs (rollout_max_batched_tokens=8192)
- Right: doubled runs (rollout_max_batched_tokens=16384)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier2/phase_dominance_map_llama_rollout_doubled.png")
INCLUDE_VALIDATION = False

LEFT_BASELINE_RUN_IDS = {
    "stage1_llama8b_ppo_20260301_075906",
    "stage1_llama8b_remax_20260301_083423",
}
RIGHT_DOUBLED_RUN_IDS = {
    "llama8b_ppo_16384_20260301_105643",
    "llama8b_remax_16384_20260301_113139",
}
TARGET_POLICIES = {"ppo", "remax"}

POLICY_COLORS = {
    "ppo": "#1f77b4",
    "remax": "#ff7f0e",
}
PHASE_MARKERS = {
    "rollout": "o",
    "training": "s",
    "rl_policy": "^",
}


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = ["run_id", "phase_name", "energy_share", "time_share", "policy", "model"]
    filter_cols_optional = [
        "global_step_canonical",
        "global_step",
        "analysis_ok",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    df, _ = load_view("phase_fact_view")
    needed = [col for col in required_cols + filter_cols_optional if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            "phase_fact_view is missing required columns "
            f"{missing_required}. Available columns: {list(df.columns)}"
        )
    return df[needed].copy()


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    phase_df["run_id"] = phase_df["run_id"].astype(str)
    phase_df["policy_norm"] = phase_df["policy"].astype(str).str.lower()

    expected_runs = LEFT_BASELINE_RUN_IDS | RIGHT_DOUBLED_RUN_IDS
    missing_runs = sorted(expected_runs - set(phase_df["run_id"].unique().tolist()))
    if missing_runs:
        raise ValueError(f"Missing required run IDs in phase_fact_view: {missing_runs}")

    plot_df = phase_df[phase_df["run_id"].isin(expected_runs)].copy()
    plot_df = plot_df[plot_df["policy_norm"].isin(TARGET_POLICIES)].copy()

    plot_df_before_filter = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    filtering = explain_filtering(plot_df_before_filter, plot_df)
    print(f"filtering={filtering}")

    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[~plot_df["phase_bucket"].isin(["other", "validation"])].copy()
    plot_df["panel"] = plot_df["run_id"].map(
        lambda rid: "Baseline (8192)" if rid in LEFT_BASELINE_RUN_IDS else "Doubled (16384)"
    )

    run_counts = (
        plot_df.groupby(["panel", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["panel", "policy_norm"])
    )
    print("runs included by (panel, policy):")
    print(run_counts.to_string(index=False))

    point_counts = (
        plot_df.groupby(["panel", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["panel", "policy_norm", "phase_bucket"])
    )
    print("points plotted by (panel, policy, phase_bucket):")
    print(point_counts.to_string(index=False))

    panel_order = ["Baseline (8192)", "Doubled (16384)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panel_axes = dict(zip(panel_order, axes))

    for panel in panel_order:
        ax = panel_axes[panel]
        panel_df = plot_df[plot_df["panel"] == panel]
        for policy in sorted(TARGET_POLICIES):
            policy_df = panel_df[panel_df["policy_norm"] == policy]
            for phase_bucket, bucket_df in policy_df.groupby("phase_bucket", dropna=False):
                ax.scatter(
                    bucket_df["time_share"],
                    bucket_df["energy_share"],
                    s=52,
                    marker=PHASE_MARKERS.get(str(phase_bucket), "o"),
                    color=POLICY_COLORS.get(policy, "#333333"),
                    edgecolor="black",
                    linewidth=0.45,
                    alpha=0.75,
                    zorder=3,
                )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("time_share")
        ax.set_title(panel)
        ax.grid(alpha=0.2)

    policy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=POLICY_COLORS[policy],
            markerfacecolor=POLICY_COLORS[policy],
            markeredgecolor="black",
            markersize=8,
            label=policy,
        )
        for policy in sorted(TARGET_POLICIES)
    ]
    phase_handles = [
        Line2D(
            [0],
            [0],
            marker=PHASE_MARKERS[phase],
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label=phase,
        )
        for phase in ["rollout", "training", "rl_policy"]
    ]

    axes[0].set_ylabel("energy_share")
    fig.suptitle("Llama phase dominance: baseline vs doubled rollout tokens", y=0.99)
    fig.legend(handles=policy_handles, title="policy (color)", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.33, 0.955))
    fig.legend(handles=phase_handles, title="phase (shape)", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.76, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
