"""Baseline phase energy/time pie chart.

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
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


INCLUDE_VALIDATION = False
OUTPATH = Path("plots/out/baselines/phase_energy_time_pie.png")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
TARGET_MODEL_FACETS = ("Llama", "Qwen")
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "rollout",
    "rl_policy": "preparation",
    "training": "training",
}
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}

POLICY_COLORS = {
    "ppo": "#1f77b4",
    "remax": "#ff7f0e",
    "grpo": "#2ca02c",
}

PHASE_COLORS = {
    "rollout": "#4C78A8",
    "training": "#F58518",
    "rl_policy": "#54A24B",
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


def _load_runs_with_slurm_metadata() -> pd.DataFrame:
    df_runs, _ = load_view("runs")
    required = ["run_id", "slurm_job_name"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(
            "runs is missing required slurm metadata columns "
            f"{missing}. Available columns: {list(df_runs.columns)}"
        )
    return df_runs[required].copy()


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    runs_df = _load_run_summary_for_selection()
    runs_meta_df = _load_runs_with_slurm_metadata()
    runs_df = runs_df.merge(runs_meta_df, on="run_id", how="left", validate="one_to_one")

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
    expected_slurm_job_name = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm_job_name
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected_runs = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & checkpoint_mask & slurm_job_mask
    ].copy()
    if selected_runs.empty:
        raise ValueError(
            "No baseline runs selected. Debug hints: "
            f"requires slurm_job_name_by_model_facet={TARGET_SLURM_JOB_NAME_BY_FACET}, "
            f"target_policies={sorted(TARGET_POLICIES)}, "
            f"target_model_facets={TARGET_MODEL_FACETS}, "
            f"requires logical_run_group prefixes={BASELINE_GROUP_PREFIXES}, "
            "excludes logical_run_group containing rollout/knob/cap, "
            f"available logical_run_group sample={runs_df['logical_run_group'].dropna().astype(str).unique()[:15].tolist()}, "
            f"available slurm_job_name sample={runs_df['slurm_job_name'].dropna().astype(str).unique()[:15].tolist()}, "
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
    print("selected run_ids:")
    print(selected_runs[["run_id", "slurm_job_name", "model_facet", "policy_norm"]].sort_values(["model_facet", "policy_norm"]).to_string(index=False))

    point_counts = (
        plot_df.groupby(["model_facet", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["model_facet", "policy_norm", "phase_bucket"])
    )
    print("points plotted by (model, policy, phase_bucket):")
    print(point_counts.to_string(index=False))

    fig, axes = plt.subplots(
        len(TARGET_MODEL_FACETS),
        len(POLICY_ORDER),
        figsize=(14, 9),
        subplot_kw={"aspect": "equal"},
    )
    if len(TARGET_MODEL_FACETS) == 1:
        axes = [axes]

    for row_idx, facet in enumerate(TARGET_MODEL_FACETS):
        facet_df = plot_df[plot_df["model_facet"] == facet]
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
            phase_keys = list(PHASE_ORDER)
            phase_labels = [PHASE_DISPLAY[p] for p in phase_keys]
            colors = [PHASE_COLORS[p] for p in phase_keys]
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

            ax.set_title(
                f"{MODEL_DISPLAY[facet]} | {POLICY_DISPLAY[policy]}",
                fontsize=10,
                fontweight="bold",
            )

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[phase], edgecolor="black", label=PHASE_DISPLAY[phase]) for phase in PHASE_ORDER
    ]

    fig.suptitle(
        "Phase Energy and Time Share by Model and Policy",
        y=0.995,
        fontweight="bold",
    )
    fig.text(0.5, 0.942, "outer=energy share, inner=time_share", ha="center", va="center", fontsize=10)
    fig.legend(handles=phase_handles, title="phase colors", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.905))
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
