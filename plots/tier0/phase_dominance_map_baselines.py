"""Phase dominance map for baseline RLHF runs.

Selection policy:
- Keep only baseline-labeled runs for Llama/Qwen with policy in {ppo, remax, grpo}.
- Exclude checkpoint-continuation runs.
- Exclude rollout-knob runs via label heuristics.

Filtering policy:
- Apply shared integrity/mask filtering via plots.plotting.filters.apply_analysis_ok.
- Exclude validation phase by default (INCLUDE_VALIDATION=False).
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


INCLUDE_VALIDATION = False
OUTPATH = Path("plots/out/figures/tier0/phase_dominance_map_baselines.png")
TARGET_POLICIES = {"ppo", "remax", "grpo"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")

POLICY_COLORS = {
    "ppo": "#1f77b4",
    "remax": "#ff7f0e",
    "grpo": "#2ca02c",
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


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = ["run_id", "phase_name", "energy_share", "time_share", "policy", "model"]
    label_cols_optional = ["logical_run_group"]
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
    needed = [col for col in required_cols + label_cols_optional + filter_cols_optional if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            "phase_fact_view is missing required columns "
            f"{missing_required}. Available columns: {list(df.columns)}"
        )
    return df[needed].copy()


def _load_run_summary_for_selection() -> pd.DataFrame:
    df_runs, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "model"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(
            "run_summary_view is missing required selection columns "
            f"{missing}. Available columns: {list(df_runs.columns)}"
        )
    return df_runs.copy()


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    runs_df = _load_run_summary_for_selection()

    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    if "logical_run_group" in runs_df.columns:
        logical_group = runs_df["logical_run_group"].astype(str).str.lower()
    else:
        raise ValueError(
            "run_summary_view is missing logical_run_group; cannot enforce 'explicitly baseline-labeled' selection."
        )

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected_runs = runs_df[baseline_label_mask & non_rollout_knob_mask & target_pair_mask & checkpoint_mask].copy()
    if selected_runs.empty:
        raise ValueError(
            "No baseline runs selected. Debug hints: "
            f"target_policies={sorted(TARGET_POLICIES)}, "
            f"target_model_facets={TARGET_MODEL_FACETS}, "
            f"requires logical_run_group prefixes={BASELINE_GROUP_PREFIXES}, "
            "excludes logical_run_group containing rollout/knob/cap, "
            f"available logical_run_group sample={runs_df['logical_run_group'].dropna().astype(str).unique()[:15].tolist()}, "
            f"available (model, policy) sample={runs_df[['model','policy']].dropna().drop_duplicates().head(15).to_dict(orient='records')}"
        )

    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    plot_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    if plot_df.empty:
        raise ValueError(
            "Selected run_ids were found in run_summary_view but produced no rows in phase_fact_view. "
            f"selected_run_ids={selected_run_ids}"
        )

    # Apply shared integrity + startup filters and log auditable breakdown.
    plot_df_before_filter = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    filtering = explain_filtering(plot_df_before_filter, plot_df)
    print(f"filtering={filtering}")

    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    plot_df["policy_norm"] = plot_df["policy"].astype(str).str.lower()
    plot_df["model_facet"] = plot_df["model"].map(_model_facet)
    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[~plot_df["phase_bucket"].isin(["other", "validation"])].copy()

    # Console summaries required by spec.
    run_counts = (
        selected_runs.groupby(["model_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["model_facet", "policy_norm"])
    )
    print("runs included by (model, policy):")
    print(run_counts.to_string(index=False))

    point_counts = (
        plot_df.groupby(["model_facet", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["model_facet", "policy_norm", "phase_bucket"])
    )
    print("points plotted by (model, policy, phase_bucket):")
    print(point_counts.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    facet_axes = dict(zip(TARGET_MODEL_FACETS, axes))

    for facet in TARGET_MODEL_FACETS:
        ax = facet_axes[facet]
        facet_df = plot_df[plot_df["model_facet"] == facet]

        for policy in sorted(TARGET_POLICIES):
            policy_df = facet_df[facet_df["policy_norm"] == policy]
            for bucket, bucket_df in policy_df.groupby("phase_bucket", dropna=False):
                ax.scatter(
                    bucket_df["time_share"],
                    bucket_df["energy_share"],
                    s=52,
                    marker=PHASE_MARKERS.get(str(bucket), "o"),
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
        ax.set_title(facet)
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
    fig.suptitle("Phase dominance map: time share vs energy share", y=0.99)
    fig.legend(handles=policy_handles, title="policy (color)", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.33, 0.955))
    fig.legend(handles=phase_handles, title="phase (shape)", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.76, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
