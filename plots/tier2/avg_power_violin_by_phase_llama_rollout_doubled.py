"""Violin+box avg_power_w by phase_name for Llama PPO/ReMax baseline vs doubled rollout tokens."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier2/avg_power_violin_by_phase_llama_rollout_doubled.png")
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
PHASE_ORDER = ["rollout", "training", "rl_policy"]
RUN_DISPLAY_ORDER = {
    "Baseline (8192)": [
        "stage1_llama8b_ppo_20260301_075906",
        "stage1_llama8b_remax_20260301_083423",
    ],
    "Doubled (16384)": [
        "llama8b_ppo_16384_20260301_105643",
        "llama8b_remax_16384_20260301_113139",
    ],
}
PANEL_ORDER = ["Baseline (8192)", "Doubled (16384)"]
POLICY_ORDER = ["ppo", "remax"]
POLICY_COLORS = {
    "ppo": "#1f77b4",
    "remax": "#ff7f0e",
}


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = ["run_id", "phase_name", "avg_power_w", "policy", "model"]
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
    needed = [c for c in required_cols + filter_cols_optional if c in df.columns]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            "phase_fact_view is missing required columns "
            f"{missing_required}. Available columns: {list(df.columns)}"
        )
    return df[needed].copy()


def main() -> None:
    df = _load_phase_fact_for_plot()
    df["run_id"] = df["run_id"].astype(str)
    df["policy_norm"] = df["policy"].astype(str).str.lower()

    expected_runs = LEFT_BASELINE_RUN_IDS | RIGHT_DOUBLED_RUN_IDS
    missing_runs = sorted(expected_runs - set(df["run_id"].unique().tolist()))
    if missing_runs:
        raise ValueError(f"Missing required run IDs in phase_fact_view: {missing_runs}")

    plot_df = df[df["run_id"].isin(expected_runs)].copy()
    plot_df = plot_df[plot_df["policy_norm"].isin(TARGET_POLICIES)].copy()
    plot_df = plot_df[plot_df["model"].astype(str).str.contains("llama", case=False, na=False)].copy()

    before = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    filtering = explain_filtering(before, plot_df)
    print(f"filtering={filtering}")

    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[plot_df["phase_bucket"].isin(PHASE_ORDER)].copy()
    plot_df["avg_power_w"] = pd.to_numeric(plot_df["avg_power_w"], errors="coerce")
    plot_df = plot_df.dropna(subset=["avg_power_w"]).copy()

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
        plot_df.groupby(["phase_bucket", "panel", "run_id", "policy_norm"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["phase_bucket", "panel", "policy_norm"])
    )
    print("points plotted by (phase_bucket, panel, run_id, policy):")
    print(point_counts.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, phase in zip(axes, PHASE_ORDER):
        phase_df = plot_df[plot_df["phase_bucket"] == phase].copy()
        if phase_df.empty:
            raise ValueError(f"No rows available for phase '{phase}' after filtering.")

        data: list[list[float]] = []
        positions: list[float] = []
        colors: list[str] = []

        for cond_idx, panel in enumerate(PANEL_ORDER):
            for policy in POLICY_ORDER:
                run_id = RUN_DISPLAY_ORDER[panel][0] if policy == "ppo" else RUN_DISPLAY_ORDER[panel][1]
                vals = phase_df.loc[
                    (phase_df["panel"] == panel) & (phase_df["policy_norm"] == policy) & (phase_df["run_id"] == run_id),
                    "avg_power_w",
                ].tolist()
                if not vals:
                    raise ValueError(
                        f"Empty distribution for phase={phase}, panel={panel}, policy={policy}, run_id={run_id}."
                    )
                data.append(vals)
                positions.append((cond_idx + 1) + (-0.16 if policy == "ppo" else 0.16))
                colors.append(POLICY_COLORS[policy])

        vp = ax.violinplot(data, positions=positions, widths=0.28, showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_linewidth(0.6)
            body.set_alpha(0.4)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.09,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.1},
            flierprops={"marker": ".", "markersize": 2.0, "alpha": 0.35, "markerfacecolor": "black", "markeredgecolor": "black"},
        )

        y_lo = float(phase_df["avg_power_w"].quantile(0.02))
        y_hi = float(phase_df["avg_power_w"].quantile(0.98))
        if y_hi <= y_lo:
            y_lo = float(phase_df["avg_power_w"].min())
            y_hi = float(phase_df["avg_power_w"].max())
        pad = max((y_hi - y_lo) * 0.08, 1.0)
        ax.set_ylim(y_lo - pad, y_hi + pad)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(PANEL_ORDER)
        ax.set_xlabel("run condition")
        ax.set_title(phase)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("avg_power_w")
    fig.suptitle("Llama avg_power_w distributions by phase", y=0.99)
    run_handles = [
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=POLICY_COLORS["ppo"], markeredgecolor="black", markersize=8, label="PPO run"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=POLICY_COLORS["remax"], markeredgecolor="black", markersize=8, label="ReMax run"),
    ]
    fig.legend(handles=run_handles, title="run color", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
