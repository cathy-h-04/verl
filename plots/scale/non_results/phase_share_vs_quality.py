"""Phase time-share versus validation quality for llama scaling runs."""

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
from plots.plotting.style import savefig_paper, scatter_paper


OUTPATH = Path("plots/out/scale/phase_share_vs_quality.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
POLICY_COLORS = {"ppo": "#5B2A86", "remax": "#FF5C7A", "grpo": "#0097A7"}

CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
CONFIG_MARKERS = {"2xA100": "*", "2xH200": "o", "4xA100": "s", "4xH200": "^"}

PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout Share",
    "rl_policy": "Preparation Share",
    "training": "Training Share",
}

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 14
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


def _select_clean_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")

    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(
        summary_df[
            [
                "run_id",
                "policy",
                "join_coverage_rate",
                "phase_boundary_integrity_rate",
                "best_validation_metric",
                "is_checkpoint_continuation",
            ]
        ],
        on="run_id",
        how="inner",
        validate="one_to_one",
    )
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[
        selected["policy_norm"].isin(POLICY_ORDER)
        & selected["config"].isin(CONFIG_ORDER)
        & (pd.to_numeric(selected["join_coverage_rate"], errors="coerce") == 1.0)
        & (pd.to_numeric(selected["phase_boundary_integrity_rate"], errors="coerce") == 1.0)
        & (~selected["is_checkpoint_continuation"].fillna(False).astype(bool))
        & selected["best_validation_metric"].notna()
    ].copy()
    if selected.empty:
        raise ValueError("No clean llama_scaling runs selected.")
    return selected[["run_id", "policy_norm", "config", "best_validation_metric"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_clean_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_fact, _ = load_view("phase_fact_view")
    needed = [
        "run_id",
        "phase_name",
        "phase_time_s",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    phase_df = phase_fact[phase_fact["run_id"].astype(str).isin(run_ids)][[c for c in needed if c in phase_fact.columns]].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"filtering={explain_filtering(before, phase_df)}")
    phase_df["phase_name"] = phase_df["phase_name"].astype(str).str.lower()
    phase_df = phase_df[phase_df["phase_name"].isin(PHASE_ORDER)].copy()
    phase_df["phase_time_s"] = pd.to_numeric(phase_df["phase_time_s"], errors="coerce")
    phase_df = phase_df.dropna(subset=["phase_time_s"]).copy()

    run_phase = (
        phase_df.groupby(["run_id", "phase_name"], dropna=False)["phase_time_s"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for phase in PHASE_ORDER:
        if phase not in run_phase.columns:
            run_phase[phase] = 0.0
    run_phase["total_time_s"] = run_phase[list(PHASE_ORDER)].sum(axis=1)
    for phase in PHASE_ORDER:
        run_phase[f"{phase}_share"] = run_phase[phase] / run_phase["total_time_s"]

    plot_df = selected_runs.merge(run_phase[["run_id"] + [f"{phase}_share" for phase in PHASE_ORDER]], on="run_id", how="inner")
    print("run-level shares:")
    print(
        plot_df[
            ["run_id", "policy_norm", "config", "best_validation_metric"] + [f"{phase}_share" for phase in PHASE_ORDER]
        ]
        .sort_values(["policy_norm", "config", "run_id"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(1, len(PHASE_ORDER), figsize=(15.4, 5.8), sharey=True)
    for ax, phase in zip(axes, PHASE_ORDER):
        xcol = f"{phase}_share"
        for policy in POLICY_ORDER:
            for config in CONFIG_ORDER:
                sub = plot_df[(plot_df["policy_norm"] == policy) & (plot_df["config"] == config)]
                if sub.empty:
                    continue
                ax.scatter(
                    sub[xcol],
                    sub["best_validation_metric"],
                    s=180,
                    marker=CONFIG_MARKERS[config],
                    c=POLICY_COLORS[policy],
                    edgecolors="black",
                    linewidths=1.0,
                    alpha=0.97,
                    zorder=3,
                )

        scatter_paper(ax)
        ax.set_title(PHASE_DISPLAY[phase], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Time share", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    axes[0].set_ylabel("Best validation score", fontsize=AXIS_LABEL_SIZE)

    policy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=POLICY_DISPLAY[policy],
            markerfacecolor=POLICY_COLORS[policy],
            markeredgecolor="black",
            markersize=10,
        )
        for policy in POLICY_ORDER
    ]
    config_handles = [
        Line2D(
            [0],
            [0],
            marker=CONFIG_MARKERS[config],
            color="black",
            linestyle="None",
            label=CONFIG_DISPLAY[config],
            markersize=10 if config != "2xA100" else 13,
        )
        for config in CONFIG_ORDER
    ]

    fig.suptitle("Phase Time Share vs Validation Quality", y=0.99, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    legend1 = fig.legend(
        handles=policy_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.34, 0.94),
        fontsize=LEGEND_FONT_SIZE,
        title="Policy",
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles=config_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.77, 0.94),
        fontsize=LEGEND_FONT_SIZE,
        title="Config",
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="phase_share_vs_quality",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["runs", "run_summary_view", "phase_fact_view"],
            "selection": "clean scaling runs with join_coverage_rate=1 and phase_boundary_integrity_rate=1",
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
