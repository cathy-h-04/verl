"""Violin+box plot of avg_power_w by phase_name, faceted by model (baseline runs).

Run selection intentionally mirrors:
plots/tier0/phase_dominance_map_baselines.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/avg_power_violin_by_phase_baselines.png")
INCLUDE_VALIDATION = False

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
PHASE_ORDER = ["rollout", "training", "rl_policy"]
POLICY_ORDER = ["ppo", "remax", "grpo"]
POLICY_COLORS = {
    "ppo": "#4c78a8",
    "remax": "#f58518",
    "grpo": "#54a24b",
}


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


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
    needed = [col for col in required_cols + ["logical_run_group"] + filter_cols_optional if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            "phase_fact_view is missing required columns "
            f"{missing_required}. Available columns: {list(df.columns)}"
        )
    return df[needed].copy()


def _load_run_summary_for_selection() -> pd.DataFrame:
    df_runs, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "model", "logical_run_group"]
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
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

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
        raise ValueError("No baseline runs selected.")

    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    plot_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    if plot_df.empty:
        raise ValueError(f"No phase_fact_view rows found for selected runs: {selected_run_ids}")

    plot_df_before_filter = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    filtering = explain_filtering(plot_df_before_filter, plot_df)
    print(f"filtering={filtering}")

    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df["model_facet"] = plot_df["model"].map(_model_facet)
    plot_df = plot_df[plot_df["phase_bucket"].isin(PHASE_ORDER)].copy()
    plot_df["policy_norm"] = plot_df["policy"].astype(str).str.lower()
    plot_df["avg_power_w"] = pd.to_numeric(plot_df["avg_power_w"], errors="coerce")
    plot_df = plot_df.dropna(subset=["avg_power_w"]).copy()

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
        plot_df.groupby(["model_facet", "phase_bucket", "policy_norm"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["model_facet", "phase_bucket", "policy_norm"])
    )
    print("points plotted by (model, phase_bucket, policy):")
    print(point_counts.to_string(index=False))

    phase_means = (
        plot_df.groupby(["phase_bucket", "model_facet"], dropna=False)["avg_power_w"]
        .mean()
        .unstack("model_facet")
        .reindex(PHASE_ORDER)
    )
    phase_pct_diff = ((phase_means.get("Qwen") - phase_means.get("Llama")) / phase_means.get("Llama")) * 100.0
    print("phase mean avg_power_w and %diff ((Qwen-Llama)/Llama*100):")
    print(
        pd.DataFrame(
            {
                "phase_bucket": PHASE_ORDER,
                "llama_mean_power_w": phase_means.get("Llama"),
                "qwen_mean_power_w": phase_means.get("Qwen"),
                "pct_diff_qwen_vs_llama": phase_pct_diff,
            }
        ).to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    facet_axes = dict(zip(TARGET_MODEL_FACETS, axes))

    for facet in TARGET_MODEL_FACETS:
        ax = facet_axes[facet]
        facet_df = plot_df[plot_df["model_facet"] == facet].copy()

        data = []
        positions = []
        color_by_violin = []
        for phase_i, phase in enumerate(PHASE_ORDER, start=1):
            for policy_i, policy in enumerate(POLICY_ORDER):
                vals = facet_df.loc[
                    (facet_df["phase_bucket"] == phase) & (facet_df["policy_norm"] == policy),
                    "avg_power_w",
                ].tolist()
                if not vals:
                    continue
                offset = (policy_i - 1) * 0.24
                data.append(vals)
                positions.append(phase_i + offset)
                color_by_violin.append(POLICY_COLORS[policy])

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.2,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(vp["bodies"], color_by_violin):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_linewidth(0.6)
            body.set_alpha(0.45)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.08,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.1},
            flierprops={"marker": ".", "markersize": 2.0, "alpha": 0.35, "markerfacecolor": "black", "markeredgecolor": "black"},
        )

        ax.set_xticks(range(1, len(PHASE_ORDER) + 1))
        ax.set_xticklabels(PHASE_ORDER)
        ax.set_xlabel("phase_name")
        ax.set_title(facet)
        ax.grid(axis="y", alpha=0.2)

    y_min, y_max = axes[0].get_ylim()
    y_span = max(y_max - y_min, 1e-9)
    for ax in axes:
        ax.set_ylim(y_min, y_max + 0.12 * y_span)
    annot_y = y_max + 0.08 * y_span
    for phase_i, phase in enumerate(PHASE_ORDER, start=1):
        pct = phase_pct_diff.get(phase)
        label = f"Qwen vs Llama: {pct:+.1f}%" if pd.notna(pct) else "Qwen vs Llama: n/a"
        axes[0].text(phase_i, annot_y, label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_ylabel("avg_power_w")
    fig.suptitle("avg_power_w by phase_name (baseline runs)", y=0.99)
    policy_handles = [Patch(facecolor=POLICY_COLORS[p], edgecolor="black", alpha=0.45, label=p) for p in POLICY_ORDER]
    fig.legend(handles=policy_handles, title="policy", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
