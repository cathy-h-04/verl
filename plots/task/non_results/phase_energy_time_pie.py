"""Phase energy/time pie chart for task comparison."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


INCLUDE_VALIDATION = False
OUTPATH = Path("plots/out/task/non_results/phase_energy_time_pie.png")
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
TARGET_DATASETS = ("gsm8k", "rlhf-ff")
DATASET_DISPLAY = {
    "gsm8k": "gsm8k",
    "rlhf-ff": "full-hh-rlhf",
}
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "rollout",
    "rl_policy": "preparation",
    "training": "training",
}
PHASE_COLORS = {
    "rollout": "#4C78A8",
    "training": "#F58518",
    "rl_policy": "#54A24B",
}
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}


def _autopct_fmt(pct: float) -> str:
    return f"{pct:.0f}%" if pct > 0 else ""


def _center_autotexts(wedges, autotexts, radius: float, width: float) -> None:
    label_radius = radius - (width / 2.0)
    for wedge, autotext in zip(wedges, autotexts):
        theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        autotext.set_position((label_radius * math.cos(theta), label_radius * math.sin(theta)))
        autotext.set_ha("center")
        autotext.set_va("center")
        autotext.set_clip_on(False)


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = ["run_id", "phase_name", "energy_share", "time_share", "policy"]
    optional_cols = [
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
    needed = [col for col in required_cols + optional_cols if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"phase_fact_view is missing required columns {missing_required}")
    return df[needed].copy()


def _select_runs() -> pd.DataFrame:
    runs_df, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "dataset"]
    missing = [col for col in required if col not in runs_df.columns]
    if missing:
        raise ValueError(f"run_summary_view is missing required selection columns {missing}")
    runs_df = runs_df.copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["dataset_group"] = runs_df["dataset"].astype(str).str.lower()
    selected = runs_df[
        runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["dataset_group"].isin(TARGET_DATASETS)
    ].copy()
    if "is_checkpoint_continuation" in selected.columns:
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
    if selected.empty:
        raise ValueError("No task-comparison runs selected.")
    return selected[["run_id", "policy_norm", "dataset_group"]].drop_duplicates()


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    selected_runs = _select_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    plot_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    if plot_df.empty:
        raise ValueError(f"Selected run_ids produced no rows in phase_fact_view: {selected_run_ids}")

    before = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    print(f"filtering={explain_filtering(before, plot_df)}")
    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    plot_df = plot_df.merge(selected_runs, on="run_id", how="left", validate="many_to_one")
    plot_df["policy_norm"] = plot_df["policy"].astype(str).str.lower()
    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[~plot_df["phase_bucket"].isin(["other", "validation"])].copy()

    run_counts = (
        selected_runs.groupby(["dataset_group", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["dataset_group", "policy_norm"])
    )
    print("runs included by (dataset, policy):")
    print(run_counts.to_string(index=False))

    point_counts = (
        plot_df.groupby(["dataset_group", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["dataset_group", "policy_norm", "phase_bucket"])
    )
    print("points plotted by (dataset, policy, phase_bucket):")
    print(point_counts.to_string(index=False))

    fig, axes = plt.subplots(len(TARGET_DATASETS), len(POLICY_ORDER), figsize=(14, 9), subplot_kw={"aspect": "equal"})
    if len(TARGET_DATASETS) == 1:
        axes = [axes]

    for row_idx, dataset in enumerate(TARGET_DATASETS):
        facet_df = plot_df[plot_df["dataset_group"] == dataset]
        for col_idx, policy in enumerate(POLICY_ORDER):
            ax = axes[row_idx][col_idx]
            combo_df = facet_df[facet_df["policy_norm"] == policy]
            phase_means = (
                combo_df.groupby("phase_bucket", dropna=False)[["time_share", "energy_share"]]
                .mean()
                .reindex(PHASE_ORDER)
                .fillna(0.0)
            )
            time_vals = phase_means["time_share"].clip(lower=0.0).to_numpy()
            energy_vals = phase_means["energy_share"].clip(lower=0.0).to_numpy()
            colors = [PHASE_COLORS[p] for p in PHASE_ORDER]
            nonzero_total = float(time_vals.sum() + energy_vals.sum())
            if nonzero_total <= 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.set_axis_off()
                continue

            outer_wedges, _, outer_autotexts = ax.pie(
                energy_vals if energy_vals.sum() > 0 else [1.0],
                labels=None,
                colors=colors if energy_vals.sum() > 0 else ["#D9D9D9"],
                radius=1.0,
                wedgeprops={"width": 0.32, "edgecolor": "white", "linewidth": 0.8},
                autopct=_autopct_fmt if energy_vals.sum() > 0 else None,
                pctdistance=0.83,
                textprops={"fontsize": 9, "color": "white", "weight": "bold"},
            )
            inner_wedges, _, inner_autotexts = ax.pie(
                time_vals if time_vals.sum() > 0 else [1.0],
                labels=None,
                colors=colors if time_vals.sum() > 0 else ["#BDBDBD"],
                radius=0.66,
                wedgeprops={"width": 0.32, "edgecolor": "white", "linewidth": 0.8},
                autopct=_autopct_fmt if time_vals.sum() > 0 else None,
                pctdistance=0.52,
                textprops={"fontsize": 9, "color": "white", "weight": "bold"},
            )
            if energy_vals.sum() > 0:
                _center_autotexts(outer_wedges, outer_autotexts, radius=1.0, width=0.32)
            if time_vals.sum() > 0:
                _center_autotexts(inner_wedges, inner_autotexts, radius=0.66, width=0.32)

            ax.set_title(f"{DATASET_DISPLAY[dataset]} | {POLICY_DISPLAY[policy]}", fontsize=10, fontweight="bold")

    phase_handles = [Patch(facecolor=PHASE_COLORS[phase], edgecolor="black", label=PHASE_DISPLAY[phase]) for phase in PHASE_ORDER]
    fig.suptitle("Phase Energy and Time Share by Policy and Dataset", y=0.995, fontweight="bold")
    fig.text(0.5, 0.942, "outer=energy share, inner=time_share", ha="center", va="center", fontsize=10)
    fig.legend(handles=phase_handles, title="phase colors", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.905))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
