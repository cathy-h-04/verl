"""Host DRAM share versus GPU utilization across phases."""

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


OUTPATH = Path("plots/out/scale/dram_share_vs_gpu_util.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
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
CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
CONFIG_MARKERS = {
    "2xA100": "*",
    "2xH200": "o",
    "4xA100": "s",
    "4xH200": "^",
}

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 10


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

    keep_cols = ["run_id", "phase_name", "dram_energy_j", "total_energy_j", "gpu_util_mean"]
    pf = pf[keep_cols].copy()
    pf["phase_name"] = pf["phase_name"].astype(str).str.lower()
    pf = pf[pf["phase_name"].isin(PHASE_ORDER)].copy()
    for col in ["dram_energy_j", "total_energy_j", "gpu_util_mean"]:
        pf[col] = pd.to_numeric(pf[col], errors="coerce")
    pf = pf.dropna(subset=["dram_energy_j", "total_energy_j", "gpu_util_mean"]).copy()
    pf = pf[pf["total_energy_j"] > 0].copy()
    pf["dram_share_pct"] = 100.0 * pf["dram_energy_j"] / pf["total_energy_j"]
    pf = pf.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    run_phase = (
        pf.groupby(["run_id", "config", "policy_norm", "phase_name"], dropna=False)[["gpu_util_mean", "dram_share_pct"]]
        .mean()
        .reset_index()
    )
    print("run-phase summary:")
    print(run_phase.sort_values(["phase_name", "policy_norm", "config"]).to_string(index=False))

    fig, axes = plt.subplots(1, len(PHASE_ORDER), figsize=(16.4, 5.4), sharey=True)
    for ax, phase in zip(axes, PHASE_ORDER):
        sub = run_phase[run_phase["phase_name"] == phase].copy()
        for _, row in sub.iterrows():
            ax.scatter(
                row["gpu_util_mean"],
                row["dram_share_pct"],
                s=120,
                marker=CONFIG_MARKERS[row["config"]],
                color=POLICY_COLORS[row["policy_norm"]],
                edgecolor="black",
                linewidth=0.9,
                alpha=0.9,
                zorder=3,
            )
        ax.set_title(PHASE_DISPLAY[phase], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.set_axisbelow(True)
        ax.set_xlabel("GPU utilization (%)", fontsize=AXIS_LABEL_SIZE)

    axes[0].set_ylabel("DRAM share of total energy (%)", fontsize=AXIS_LABEL_SIZE)
    fig.suptitle("Host DRAM Share vs GPU Utilization", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", y=0.98)

    policy_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POLICY_COLORS[p], markeredgecolor="black", markersize=9, label=POLICY_DISPLAY[p])
        for p in POLICY_ORDER
    ]
    config_handles = [
        Line2D([0], [0], marker=CONFIG_MARKERS[c], color="black", linestyle="None", markersize=9, label=CONFIG_DISPLAY[c])
        for c in CONFIG_ORDER
    ]
    fig.legend(handles=policy_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.36, 0.92), fontsize=LEGEND_FONT_SIZE)
    fig.legend(handles=config_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.79, 0.92), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="dram_share_vs_gpu_util",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
            "metric": "dram_share_pct = 100 * dram_energy_j / total_energy_j",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
