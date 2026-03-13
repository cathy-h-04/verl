"""Fraction of NVML periodic samples with SW power-cap throttle active, by phase and policy.

Bar chart with per-step jitter overlay. One panel per phase (rollout / preparation / training).
X-axis: policy. Paired bars: Llama vs Qwen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/baselines/phase_throttle_cap_rate.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")
INCLUDE_VALIDATION = False

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {"Llama": "Llama-3.1-8B-Inst", "Qwen": "Qwen2.5-3B-Inst"}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {"rollout": "Rollout", "rl_policy": "Preparation", "training": "Training"}

MODEL_COLORS = {"Llama": "#1D4E89", "Qwen": "#C73E1D"}
JITTER_ALPHA = 0.35
JITTER_SIZE = 3.5
BAR_WIDTH = 0.32


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _select_baseline_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")
    required = ["run_id", "policy", "model", "logical_run_group"]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    ).copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = (
        runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    )
    expected_slurm = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & slurm_job_mask & checkpoint_mask
    ].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def main() -> None:
    selected_runs = _select_baseline_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    selected_meta = selected_runs[["run_id", "model_facet", "policy_norm"]].drop_duplicates()

    phase_fact, _ = load_view("phase_fact_view")
    df_before = phase_fact[phase_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    df = apply_analysis_ok(df_before)
    filter_info = explain_filtering(df_before, df)
    print("[filtering]", filter_info)

    if not INCLUDE_VALIDATION and "is_validation_step" in df.columns:
        df = df[~df["is_validation_step"].fillna(False)].copy()

    df["phase_n"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_n"].isin(PHASE_ORDER)].copy()
    df["throttle_rate"] = pd.to_numeric(df["throttle_sw_power_cap_rate"], errors="coerce")
    df = df.dropna(subset=["throttle_rate"]).copy()
    df = df.merge(selected_meta, on="run_id", how="inner")

    # per-step mean across all GPUs already captured in phase_fact_view rate
    step_df = df.groupby(
        ["run_id", "global_step_canonical", "phase_n", "model_facet", "policy_norm"],
        as_index=False,
    )["throttle_rate"].mean()

    # summary: mean and std over steps per (phase, model, policy)
    # (one run per policy×model combination, so step-level spread is the right error bar)
    summary = step_df.groupby(["phase_n", "model_facet", "policy_norm"], as_index=False).agg(
        mean=("throttle_rate", "mean"),
        std=("throttle_rate", "std"),
        n=("throttle_rate", "count"),
    )

    print("\nplot summary (mean throttle cap rate):")
    print(summary.sort_values(["policy_norm", "phase_n", "model_facet"]).to_string(index=False))

    # ── plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)
    x = np.arange(len(POLICY_ORDER), dtype=float)

    for ax, phase in zip(axes, PHASE_ORDER):
        phase_sum = summary[summary["phase_n"] == phase].copy()
        phase_steps = step_df[step_df["phase_n"] == phase].copy()

        for i, model_facet in enumerate(TARGET_MODEL_FACETS):
            # tight side-by-side: centres exactly BAR_WIDTH apart, zero gap
            offset = (i - 0.5) * BAR_WIDTH
            xpos = x + offset
            color = MODEL_COLORS[model_facet]

            means = []
            stds = []
            for policy in POLICY_ORDER:
                row = phase_sum[(phase_sum["model_facet"] == model_facet) & (phase_sum["policy_norm"] == policy)]
                means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
                stds.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

            ax.bar(
                xpos,
                means,
                width=BAR_WIDTH,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                label=MODEL_DISPLAY[model_facet] if phase == PHASE_ORDER[0] else None,
                zorder=2,
            )
            ax.errorbar(
                xpos,
                means,
                yerr=stds,
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1.0,
                zorder=3,
            )

            # std text annotation clearly above the errorbar top
            for xp, m, s in zip(xpos, means, stds):
                if np.isnan(m):
                    continue
                top = m + s
                ax.text(
                    xp, top + 0.055,
                    f"\u00b1{s:.0%}",
                    ha="center", va="bottom",
                    fontsize=8, color="black", fontweight="bold",
                    zorder=5,
                )

            # per-step jitter
            rng = np.random.default_rng(42 + i)
            for j, policy in enumerate(POLICY_ORDER):
                pts = phase_steps[
                    (phase_steps["model_facet"] == model_facet) & (phase_steps["policy_norm"] == policy)
                ]["throttle_rate"].to_numpy()
                if pts.size == 0:
                    continue
                jx = xpos[j] + rng.uniform(-BAR_WIDTH * 0.4, BAR_WIDTH * 0.4, size=len(pts))
                ax.scatter(jx, pts, s=JITTER_SIZE, color=color, alpha=JITTER_ALPHA, zorder=4, linewidths=0)

        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_DISPLAY[p] for p in POLICY_ORDER])
        ax.set_xlabel("Policy")
        ax.set_title(PHASE_DISPLAY[phase], fontweight="bold")
        ax.set_ylim(-0.04, 1.28)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.6)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(10)
        ax.title.set_size(11)

    axes[0].set_ylabel("Fraction of samples\nwith SW power-cap throttle", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.97),
        fontsize=9,
    )
    fig.suptitle(
        "SW Power-Cap Throttle Rate by Phase and Model",
        y=1.02,
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"\nwrote {saved}")

    manifest = build_run_manifest(
        plot_name="phase_throttle_cap_rate",
        run_ids=selected_run_ids,
        data_sources={"views": ["phase_fact_view", "run_summary_view", "runs"]},
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
