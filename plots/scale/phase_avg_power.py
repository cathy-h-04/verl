"""Phase average power with phase-segmented grouped bars."""

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


OUTPATH = Path("plots/out/scale/phase_avg_power.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

PHASE_ORDER = ("rollout", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "training": "Training",
}
CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
PHASE_COLORS = {
    "rollout": "#1D4E89",
    "training": "#C73E1D",
}

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 11

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
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[selected["policy_norm"].isin(POLICY_ORDER) & selected["config"].isin(CONFIG_ORDER)].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    if selected.empty:
        raise ValueError("No llama_scaling runs selected.")
    return selected[["run_id", "config", "policy_norm"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_df, _ = load_view("phase_fact_view")
    phase_df = phase_df[phase_df["run_id"].astype(str).isin(run_ids)].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"phase_filtering={explain_filtering(before, phase_df)}")

    phase_df = phase_df[["run_id", "phase_name", "avg_power_w"]].copy()
    phase_df["phase_name"] = phase_df["phase_name"].astype(str).str.lower()
    phase_df = phase_df[phase_df["phase_name"].isin(PHASE_ORDER)].copy()
    phase_df["avg_power_w"] = pd.to_numeric(phase_df["avg_power_w"], errors="coerce")
    phase_df = phase_df.dropna(subset=["avg_power_w"]).copy()
    phase_df = phase_df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    summary = (
        phase_df.groupby(["config", "policy_norm", "phase_name"], dropna=False)["avg_power_w"]
        .mean()
        .reset_index(name="mean_avg_power_w")
        .sort_values(["phase_name", "policy_norm", "config"])
    )
    print("phase power summary:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(16.8, 5.8), sharey=True)
    x = np.arange(len(CONFIG_ORDER), dtype=float)
    width = 0.22
    offsets = np.array([-width / 2, width / 2], dtype=float)

    for col_idx, policy in enumerate(POLICY_ORDER):
        ax = axes[col_idx]
        sub = phase_df[phase_df["policy_norm"] == policy].copy()
        ymax = float(sub["avg_power_w"].max()) if not sub.empty else 0.0
        for phase_idx, phase in enumerate(PHASE_ORDER):
            vals = []
            for config_idx, config in enumerate(CONFIG_ORDER):
                phase_vals = sub[(sub["config"] == config) & (sub["phase_name"] == phase)]["avg_power_w"].dropna()
                vals.append(float(phase_vals.mean()) if not phase_vals.empty else np.nan)
            xpos = x + offsets[phase_idx]
            ax.bar(
                xpos,
                vals,
                width=width * 0.92,
                color=PHASE_COLORS[phase],
                edgecolor="black",
                linewidth=0.9,
                alpha=0.78,
            )
        ax.set_title(POLICY_DISPLAY[policy], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_DISPLAY[c] for c in CONFIG_ORDER], rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(ymax * 1.16, 1.0))
        if col_idx == 0:
            ax.set_ylabel("Average power (W)", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.tick_params(labelleft=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_COLORS[phase], edgecolor="black", alpha=0.58)
        for phase in PHASE_ORDER
    ]
    labels = [PHASE_DISPLAY[phase] for phase in PHASE_ORDER]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Rollout and Training Average Power", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="phase_avg_power",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
            "phases": list(PHASE_ORDER),
            "metric": "avg_power_w",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
