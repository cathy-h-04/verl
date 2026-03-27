"""Best validation score for each clean llama scaling run."""

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


OUTPATH = Path("plots/out/scale/best_validation_score_by_run.png")
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

FIGURE_TITLE_SIZE = 18
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 11
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


def _short_label(run_id: str) -> str:
    cfg = _config_from_run_id(run_id)
    return cfg


def main() -> None:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")

    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(
        summary_df[
            [
                "run_id",
                "policy",
                "best_validation_metric",
                "join_coverage_rate",
                "phase_boundary_integrity_rate",
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
        raise ValueError("No clean llama_scaling runs with best_validation_metric found.")

    selected["best_validation_metric"] = pd.to_numeric(selected["best_validation_metric"], errors="coerce")
    selected = selected.dropna(subset=["best_validation_metric"]).copy()
    selected["label"] = selected["run_id"].map(_short_label)
    selected["policy_norm"] = pd.Categorical(selected["policy_norm"], categories=POLICY_ORDER, ordered=True)
    selected["config"] = pd.Categorical(selected["config"], categories=CONFIG_ORDER, ordered=True)
    selected = selected.sort_values(["policy_norm", "config", "run_id"]).reset_index(drop=True)

    print("best validation by run:")
    print(selected[["run_id", "policy_norm", "config", "best_validation_metric"]].to_string(index=False))

    x = np.arange(len(selected), dtype=float)
    colors = [POLICY_COLORS[p] for p in selected["policy_norm"].astype(str)]

    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.bar(
        x,
        selected["best_validation_metric"],
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
    )

    bar_paper(ax)
    ax.set_title("Best Validation Score by Run", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", pad=12)
    ax.set_ylabel("Best validation score", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Run", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x, selected["label"].tolist(), rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ymin = max(0.5, float(selected["best_validation_metric"].min()) - 0.05)
    ymax = min(1.0, float(selected["best_validation_metric"].max()) + 0.03)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)

    for idx in range(1, len(selected)):
        if selected.loc[idx, "policy_norm"] != selected.loc[idx - 1, "policy_norm"]:
            ax.axvline(idx - 0.5, color="black", linewidth=0.8, alpha=0.25)

    handles = [Patch(facecolor=POLICY_COLORS[p], edgecolor="black", label=POLICY_DISPLAY[p]) for p in POLICY_ORDER]
    ax.legend(handles=handles, loc="upper right", frameon=False, title="Policy", fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="best_validation_score_by_run",
        run_ids=selected["run_id"].astype(str).tolist(),
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["runs", "run_summary_view"],
            "selection": "clean scaling runs with best_validation_metric",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
