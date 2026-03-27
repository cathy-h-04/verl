"""Per-iteration phase-time trajectories for GRPO llama scaling runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/grpo_phase_time_by_iteration.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
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
CONFIG_MARKERS = {
    "2xA100": "*",
    "2xH200": "o",
    "4xA100": "s",
    "4xH200": "^",
}

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12


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


def _select_grpo_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower()
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[(selected["policy_norm"] == "grpo") & selected["config"].isin(CONFIG_ORDER)].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        checkpoint_flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(checkpoint_flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    if selected.empty:
        raise ValueError("No GRPO llama_scaling runs selected.")
    return selected[["run_id", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_grpo_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    pf, _ = load_view("phase_fact_view")
    needed = ["run_id", "global_step_canonical", "phase_name", "phase_time_s"]
    optional = [
        "analysis_ok",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    use_cols = [c for c in needed + optional if c in pf.columns]
    df = pf[pf["run_id"].astype(str).isin(run_ids)][use_cols].copy()
    if df.empty:
        raise ValueError("No GRPO phase rows found before filtering.")

    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")
    df = df[df["phase_name"].astype(str).isin(PHASE_ORDER)].copy()
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["phase_time_s"] = pd.to_numeric(df["phase_time_s"], errors="coerce")
    df = df.dropna(subset=["global_step_canonical", "phase_time_s"]).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    print("phase-time summary by (run, phase):")
    print(
        df.groupby(["run_id", "phase_name"], dropna=False)["phase_time_s"]
        .agg(["mean", "median", "max"])
        .reset_index()
        .sort_values(["run_id", "phase_name"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(len(PHASE_ORDER), 1, figsize=(13.8, 10.4), sharex=True)

    for ax, phase in zip(axes, PHASE_ORDER):
        phase_df = df[df["phase_name"].astype(str) == phase].copy()
        for config in CONFIG_ORDER:
            sub = phase_df[phase_df["config"] == config].sort_values("global_step_canonical")
            if sub.empty:
                continue
            ax.plot(
                sub["global_step_canonical"],
                sub["phase_time_s"],
                color=CONFIG_COLORS[config],
                linewidth=2.3,
                marker=CONFIG_MARKERS[config],
                markersize=7.5,
                alpha=0.96,
                label=CONFIG_DISPLAY[config],
            )
        ax.set_title(PHASE_DISPLAY[phase], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_ylabel("Time (s)", fontsize=AXIS_LABEL_SIZE)
        ax.grid(axis="both", alpha=0.24, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    axes[-1].set_xlabel("Iteration ID", fontsize=AXIS_LABEL_SIZE)

    handles = [
        Line2D(
            [0],
            [0],
            color=CONFIG_COLORS[config],
            marker=CONFIG_MARKERS[config],
            linewidth=2.3,
            markersize=7.5,
            label=CONFIG_DISPLAY[config],
        )
        for config in CONFIG_ORDER
    ]
    fig.suptitle("GRPO Phase Time by Iteration and Configuration", y=0.985, fontweight="bold", fontsize=FIGURE_TITLE_SIZE)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="grpo_phase_time_by_iteration",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
