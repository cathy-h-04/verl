"""Energy to best validation and best validation score, policy-matched baseline comparison.

Bars: energy consumed before reaching best validation checkpoint (MJ).
Overlaid markers: best validation score on secondary axis.
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
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/baselines/non_results/energy_to_best_validation_baselines.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")

# Match baselines color scheme (pcie / throttle plots)
MODEL_COLORS = {"Llama": "#1D4E89", "Qwen": "#C73E1D"}
ENERGY_SCALE = 1e6  # J → MJ
BAR_WIDTH = 0.36


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

    needed = ["run_id", "model_facet", "policy_norm", "energy_to_best_validation_j", "best_validation_metric"]
    plot_df = selected_runs[needed].copy()
    for col in ["energy_to_best_validation_j", "best_validation_metric"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["energy_to_best_validation_j"]).copy()

    print("rows used:")
    print(
        plot_df[["model_facet", "policy_norm", "energy_to_best_validation_j", "best_validation_metric"]]
        .sort_values(["model_facet", "policy_norm"])
        .to_string(index=False)
    )

    x_positions = {policy: i for i, policy in enumerate(TARGET_POLICIES)}
    offsets = {"Llama": -BAR_WIDTH / 2, "Qwen": BAR_WIDTH / 2}

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax2 = ax.twinx()

    energy_max = 0.0
    validation_vals: list[float] = []

    # pre-compute annotation offset from full validation range across both models
    all_metric_vals = [
        float(row["best_validation_metric"])
        for _, row in plot_df.iterrows()
        if pd.notna(row["best_validation_metric"])
    ]
    if all_metric_vals:
        vspan = max(max(all_metric_vals) - min(all_metric_vals), 0.02)
        annotation_offset = vspan * 0.30
    else:
        annotation_offset = 0.01

    for model in TARGET_MODEL_FACETS:
        model_df = plot_df[plot_df["model_facet"] == model]
        color = MODEL_COLORS[model]
        xs: list[float] = []
        energy_vals: list[float] = []
        metric_vals: list[float] = []

        for policy in TARGET_POLICIES:
            rows = model_df[model_df["policy_norm"] == policy]
            if rows.empty:
                continue
            row = rows.iloc[0]
            x = x_positions[policy] + offsets[model]
            e = float(row["energy_to_best_validation_j"]) / ENERGY_SCALE
            m = float(row["best_validation_metric"]) if pd.notna(row["best_validation_metric"]) else float("nan")
            xs.append(x)
            energy_vals.append(e)
            metric_vals.append(m)
            energy_max = max(energy_max, e)
            if pd.notna(m):
                validation_vals.append(m)

        ax.bar(
            xs,
            energy_vals,
            width=BAR_WIDTH,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            label=MODEL_DISPLAY[model],
            zorder=2,
        )
        ax2.plot(
            xs,
            metric_vals,
            linestyle="None",
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.0,
            zorder=4,
        )

        # validation score annotations — bold, lifted above the marker
        for xp, mv in zip(xs, metric_vals):
            if np.isnan(mv):
                continue
            ax2.text(
                xp,
                mv + annotation_offset,
                f"{mv:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=5,
            )

    ax.set_xticks([x_positions[p] for p in TARGET_POLICIES])
    ax.set_xticklabels([POLICY_DISPLAY[p] for p in TARGET_POLICIES], fontsize=10)
    ax.set_xlabel("Policy", fontsize=11)
    ax.set_ylabel("Energy to Best Validation (MJ)", fontsize=11)
    ax.set_ylim(0, energy_max * 1.22 if energy_max > 0 else 1.0)
    ax2.set_ylabel("Best Validation Score", fontsize=11)
    if validation_vals:
        vmin, vmax = min(validation_vals), max(validation_vals)
        vpad = max((vmax - vmin) * 0.55, 0.02)
        ax2.set_ylim(vmin - vpad * 0.3, vmax + vpad)

    ax.set_title(
        "Energy to Best Validation and Validation Score by Policy and Model",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9)
    ax2.tick_params(labelsize=9)

    # Legend: model color patches from bar handles only
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper right",
        fontsize=9,
    )

    fig.tight_layout()
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="energy_to_best_validation_baselines",
        run_ids=selected_run_ids,
        data_sources={"views": ["run_summary_view", "runs"]},
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
