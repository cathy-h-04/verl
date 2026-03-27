"""Throttle frequency by configuration for llama scaling runs."""

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


OUTPATH = Path("plots/out/scale/non_results/throttle_frequency.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

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
POLICY_COLORS = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}
THROTTLE_SPECS = (
    ("throttle_sw_power_cap_rate", "SW Power Cap"),
)

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
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
    return selected[["run_id", "policy_norm", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_fact, _ = load_view("phase_fact_view")
    cols = [
        "run_id",
        "phase_name",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ] + [metric for metric, _ in THROTTLE_SPECS if metric in phase_fact.columns]
    cols = [c for c in cols if c in phase_fact.columns]
    df_before = phase_fact[phase_fact["run_id"].astype(str).isin(run_ids)][cols].copy()
    df = apply_analysis_ok(df_before)
    print(f"filtering={explain_filtering(df_before, df)}")
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(["rollout", "rl_policy", "training"])].copy()
    for metric, _ in THROTTLE_SPECS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    run_summary = (
        df.groupby(["run_id", "policy_norm", "config"], dropna=False)[[metric for metric, _ in THROTTLE_SPECS]]
        .mean()
        .reset_index()
        .sort_values(["config", "policy_norm", "run_id"])
    )
    print("run-level throttle summary:")
    print(run_summary.to_string(index=False))

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 5.6), sharey=False)
    x = np.arange(len(CONFIG_ORDER), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(POLICY_ORDER))

    metric, title = THROTTLE_SPECS[0]
    ymax = float(run_summary[metric].max()) if metric in run_summary.columns and not run_summary.empty else 0.0
    for idx, policy in enumerate(POLICY_ORDER):
        xpos = x + offsets[idx]
        heights = []
        for config in CONFIG_ORDER:
            row = run_summary[(run_summary["config"] == config) & (run_summary["policy_norm"] == policy)]
            heights.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            xpos,
            heights,
            width=width,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.8,
            label=POLICY_DISPLAY[policy],
            zorder=2,
        )

    ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
    ax.set_xticks(x, [CONFIG_DISPLAY[cfg] for cfg in CONFIG_ORDER], rotation=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, max(ymax * 1.22, 0.02))
    ax.grid(axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Throttle frequency", fontsize=AXIS_LABEL_SIZE)

    handles = [Patch(facecolor=POLICY_COLORS[policy], edgecolor="black", label=POLICY_DISPLAY[policy]) for policy in POLICY_ORDER]
    fig.suptitle("Throttle Frequency Across Llama Scaling Configurations", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="throttle_frequency_by_configuration",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "run_summary_view", "runs"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
