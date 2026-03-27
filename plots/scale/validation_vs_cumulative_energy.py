"""Cumulative energy required to first reach validation score 0.80."""

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
from plots.plotting.style import bar_paper, savefig_paper


OUTPATH = Path("plots/out/scale/validation_vs_cumulative_energy.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")
GSM8K_METRIC_KEY = "val-core/openai/gsm8k/reward/mean@1"
ENERGY_TO_MJ = 1e6
TARGET_SCORE = 0.80

POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}

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

FIGURE_TITLE_SIZE = 18
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
    return selected[["run_id", "policy_norm", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_clean_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    steps = step_fact[step_fact["run_id"].astype(str).isin(run_ids)][["run_id", "global_step_canonical", "step_total_energy_j"]].copy()
    steps["global_step_canonical"] = pd.to_numeric(steps["global_step_canonical"], errors="coerce")
    steps["step_total_energy_j"] = pd.to_numeric(steps["step_total_energy_j"], errors="coerce")
    steps = steps.dropna(subset=["global_step_canonical", "step_total_energy_j"]).copy()
    steps["global_step_canonical"] = steps["global_step_canonical"].astype(int)
    steps = steps.sort_values(["run_id", "global_step_canonical"]).drop_duplicates(
        ["run_id", "global_step_canonical"], keep="last"
    )
    steps["cumulative_energy_mj"] = steps.groupby("run_id")["step_total_energy_j"].cumsum() / ENERGY_TO_MJ

    step_metrics_long, _ = load_view("step_metrics_long")
    sml = step_metrics_long[step_metrics_long["run_id"].astype(str).isin(run_ids)].copy()
    sml["global_step_canonical"] = pd.to_numeric(sml["global_step_canonical"], errors="coerce")
    sml["metric_value_float"] = pd.to_numeric(sml["metric_value_float"], errors="coerce")
    sml = sml.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    sml["global_step_canonical"] = sml["global_step_canonical"].astype(int)

    val = sml[sml["metric_key"] == GSM8K_METRIC_KEY][["run_id", "global_step_canonical", "metric_value_float"]].copy()
    val = val.rename(columns={"metric_value_float": "validation_score"})
    val = val.sort_values(["run_id", "global_step_canonical"]).drop_duplicates(
        ["run_id", "global_step_canonical"], keep="last"
    )

    plot_df = val.merge(
        steps[["run_id", "global_step_canonical", "cumulative_energy_mj"]],
        on=["run_id", "global_step_canonical"],
        how="inner",
        validate="one_to_one",
    ).merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    if plot_df.empty:
        raise ValueError("No validation rows after joining cumulative energy.")

    reached = (
        plot_df[plot_df["validation_score"] >= TARGET_SCORE]
        .sort_values(["run_id", "global_step_canonical"])
        .groupby(["run_id", "policy_norm", "config"], as_index=False)
        .first()[["run_id", "policy_norm", "config", "global_step_canonical", "validation_score", "cumulative_energy_mj"]]
        .rename(
            columns={
                "global_step_canonical": "step_reached",
                "validation_score": "score_reached",
                "cumulative_energy_mj": "energy_to_target_mj",
            }
        )
    )
    print("energy to reach target:")
    print(reached.sort_values(["policy_norm", "config", "run_id"]).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    x = np.arange(len(POLICY_ORDER), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONFIG_ORDER))

    ymax = float(reached["energy_to_target_mj"].max()) if not reached.empty else 0.0
    for idx, config in enumerate(CONFIG_ORDER):
        xpos = x + offsets[idx]
        heights = []
        for policy in POLICY_ORDER:
            row = reached[(reached["policy_norm"] == policy) & (reached["config"] == config)]
            heights.append(float(row["energy_to_target_mj"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            xpos,
            heights,
            width=width,
            color=CONFIG_COLORS[config],
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
            label=CONFIG_DISPLAY[config],
        )

    bar_paper(ax)
    ax.set_xticks(x, [POLICY_DISPLAY[p] for p in POLICY_ORDER])
    ax.set_ylabel("Energy to reach score 0.80 (MJ)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Policy", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_ylim(0, max(ymax * 1.18, 1.0))
    ax.grid(axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Cumulative Energy to Reach Validation Score 0.80", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", pad=12)

    handles = [Patch(facecolor=CONFIG_COLORS[c], edgecolor="black", label=CONFIG_DISPLAY[c]) for c in CONFIG_ORDER]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=LEGEND_FONT_SIZE, title="Config", title_fontsize=LEGEND_FONT_SIZE)

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="validation_vs_cumulative_energy",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["runs", "run_summary_view", "step_fact_view", "step_metrics_long"],
            "selection": "clean scaling runs with join_coverage_rate=1 and phase_boundary_integrity_rate=1",
            "target_score": TARGET_SCORE,
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
