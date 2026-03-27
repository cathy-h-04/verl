"""Absolute phase energy/time totals in bar-chart form for llama scaling runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


INCLUDE_VALIDATION = False
OUTPATH = Path("plots/out/scale/phase_energy_time_total.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}

CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
CONFIG_COLORS = {
    "2xA100": "#E76F51",
    "2xH200": "#2A9D8F",
    "4xA100": "#E9C46A",
    "4xH200": "#457B9D",
}

PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "rollout",
    "rl_policy": "preparation",
    "training": "training",
}

ENERGY_SCALE = 1000.0
TIME_SCALE = 60.0
FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _config_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_a100" in rid:
        return "2xA100"
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    if "4gpu_h200" in rid:
        return "4xH200"
    return "Unknown"


def _select_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")
    required_runs = ["run_id", "run_dir"]
    missing_runs = [col for col in required_runs if col not in runs_df.columns]
    if missing_runs:
        raise ValueError(f"runs is missing required selection columns {missing_runs}")
    required_summary = ["run_id", "policy"]
    missing_summary = [col for col in required_summary if col not in summary_df.columns]
    if missing_summary:
        raise ValueError(f"run_summary_view is missing required selection columns {missing_summary}")

    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower()
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[selected["policy_norm"].isin(POLICY_ORDER) & selected["config"].isin(CONFIG_ORDER)].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        checkpoint_flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(checkpoint_flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    if selected.empty:
        raise ValueError("No llama_scaling runs selected.")
    return selected[["run_id", "policy_norm", "config"]].drop_duplicates()


def _load_phase_fact_for_plot(selected_run_ids: list[str]) -> pd.DataFrame:
    required_cols = ["run_id", "phase_name", "phase_time_s", "total_energy_j"]
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
    plot_df = df[df["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    if plot_df.empty:
        raise ValueError(f"Selected run_ids produced no rows in phase_fact_view: {selected_run_ids}")
    before = plot_df.copy()
    plot_df = apply_analysis_ok(plot_df)
    print(f"filtering={explain_filtering(before, plot_df)}")
    if not INCLUDE_VALIDATION:
        plot_df = plot_df[plot_df["phase_name"].astype(str).str.lower() != "validation"].copy()
    plot_df["phase_bucket"] = plot_df["phase_name"].map(_phase_bucket)
    plot_df = plot_df[plot_df["phase_bucket"].isin(PHASE_ORDER)].copy()
    plot_df["total_energy_j"] = pd.to_numeric(plot_df["total_energy_j"], errors="coerce")
    plot_df["phase_time_s"] = pd.to_numeric(plot_df["phase_time_s"], errors="coerce")
    plot_df = plot_df.dropna(subset=["total_energy_j", "phase_time_s"]).copy()
    return plot_df


def main() -> None:
    selected_runs = _select_scaling_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    phase_df = _load_phase_fact_for_plot(selected_run_ids)
    phase_df = phase_df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    run_counts = (
        selected_runs.groupby(["policy_norm", "config"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["policy_norm", "config"])
    )
    print("runs included by (policy, config):")
    print(run_counts.to_string(index=False))

    totals = (
        phase_df.groupby(["policy_norm", "config", "phase_bucket"], dropna=False)[["total_energy_j", "phase_time_s"]]
        .sum()
        .reset_index()
    )
    print("phase totals by (policy, config, phase):")
    print(totals.sort_values(["policy_norm", "config", "phase_bucket"]).to_string(index=False))

    fig, axes = plt.subplots(2, len(POLICY_ORDER), figsize=(18.0, 9.4), sharex=False)
    x = np.arange(len(PHASE_ORDER), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONFIG_ORDER))
    global_energy_max_kj = max(float(totals["total_energy_j"].max() / ENERGY_SCALE), 1.0)
    global_time_max_min = max(float(totals["phase_time_s"].max() / TIME_SCALE), 1.0)

    for col_idx, policy in enumerate(POLICY_ORDER):
        policy_df = totals[totals["policy_norm"] == policy]
        ax_energy = axes[0][col_idx]
        ax_time = axes[1][col_idx]

        for cfg_idx, config in enumerate(CONFIG_ORDER):
            combo_df = (
                policy_df[policy_df["config"] == config]
                .set_index("phase_bucket")
                .reindex(PHASE_ORDER)
                .fillna(0.0)
            )
            energy_vals_kj = combo_df["total_energy_j"].to_numpy(dtype=float) / ENERGY_SCALE
            time_vals_min = combo_df["phase_time_s"].to_numpy(dtype=float) / TIME_SCALE
            xpos = x + offsets[cfg_idx]

            ax_energy.bar(
                xpos,
                energy_vals_kj,
                width=width,
                color=CONFIG_COLORS[config],
                edgecolor="black",
                linewidth=0.9,
                label=CONFIG_DISPLAY[config],
            )
            ax_time.bar(
                xpos,
                time_vals_min,
                width=width,
                color=CONFIG_COLORS[config],
                edgecolor="black",
                linewidth=0.9,
                label=CONFIG_DISPLAY[config],
            )

        ax_energy.set_title(POLICY_DISPLAY[policy], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        for ax in (ax_energy, ax_time):
            ax.set_xticks(x, [PHASE_DISPLAY[phase] for phase in PHASE_ORDER], rotation=0)
            ax.grid(axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=TICK_LABEL_SIZE)

        if col_idx == 0:
            ax_energy.set_ylabel("Energy (kJ)", fontsize=AXIS_LABEL_SIZE)
            ax_time.set_ylabel("Time (min)", fontsize=AXIS_LABEL_SIZE)

        ax_energy.set_ylim(0, global_energy_max_kj * 1.18)
        ax_time.set_ylim(0, global_time_max_min * 1.18)

    config_handles = [
        Patch(facecolor=CONFIG_COLORS[config], edgecolor="black", label=CONFIG_DISPLAY[config]) for config in CONFIG_ORDER
    ]
    fig.suptitle("Total Phase Energy and Time by Policy and Configuration", y=0.985, fontweight="bold", fontsize=FIGURE_TITLE_SIZE)
    fig.legend(
        handles=config_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="phase_energy_time_total",
        run_ids=selected_run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
