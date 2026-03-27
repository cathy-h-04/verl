"""Host DRAM power by configuration across scaling runs."""

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


OUTPATH = Path("plots/out/scale/dram_power_by_configuration.png")
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

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 11


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
    selected = selected[selected["config"].isin(CONFIG_ORDER)].copy()
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

    pf, _ = load_view("phase_fact_view")
    pf = pf[pf["run_id"].astype(str).isin(run_ids)].copy()
    if pf.empty:
        raise ValueError("Selected llama_scaling runs missing from phase_fact_view.")

    pf_before = pf.copy()
    pf = apply_analysis_ok(pf)
    print(f"phase_filtering={explain_filtering(pf_before, pf)}")

    pf = pf[["run_id", "phase_name", "phase_time_s", "dram_energy_j"]].copy()
    pf["phase_name"] = pf["phase_name"].astype(str).str.lower()
    pf = pf[pf["phase_name"].isin(["rollout", "rl_policy", "training"])].copy()
    pf["phase_time_s"] = pd.to_numeric(pf["phase_time_s"], errors="coerce")
    pf["dram_energy_j"] = pd.to_numeric(pf["dram_energy_j"], errors="coerce")
    pf = pf.dropna(subset=["phase_time_s", "dram_energy_j"]).copy()
    pf = pf[pf["phase_time_s"] > 0].copy()
    pf["dram_power_w"] = pf["dram_energy_j"] / pf["phase_time_s"]
    pf = pf.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    summary = (
        pf.groupby(["config", "policy_norm"], dropna=False)
        .agg(
            n_rows=("dram_power_w", "size"),
            mean_dram_power_w=("dram_power_w", "mean"),
            median_dram_power_w=("dram_power_w", "median"),
        )
        .reset_index()
        .sort_values(["config", "policy_norm"])
    )
    print("config x policy summary:")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(CONFIG_ORDER), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(POLICY_ORDER))
    ymax = float(summary["median_dram_power_w"].max()) if not summary.empty else 0.0

    for idx, policy in enumerate(POLICY_ORDER):
        xpos = x + offsets[idx]
        heights = []
        for config in CONFIG_ORDER:
            row = summary[(summary["config"] == config) & (summary["policy_norm"] == policy)]
            heights.append(float(row["median_dram_power_w"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            xpos,
            heights,
            width=width,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_DISPLAY[c] for c in CONFIG_ORDER])
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(ymax * 1.18, 1.0))
    ax.set_ylabel("DRAM power (W)", fontsize=AXIS_LABEL_SIZE)
    fig.suptitle("Host DRAM Power", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", y=0.98)
    handles = [Patch(facecolor=POLICY_COLORS[p], edgecolor="black", label=POLICY_DISPLAY[p]) for p in POLICY_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="dram_power_by_configuration",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
            "metric": "dram_power_w = dram_energy_j / phase_time_s, aggregated across rollout/preparation/training",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
