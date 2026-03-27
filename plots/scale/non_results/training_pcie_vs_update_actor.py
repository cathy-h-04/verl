"""Training PCIe throughput versus actor update time per token."""

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


OUTPATH = Path("plots/out/scale/training_pcie_vs_update_actor.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

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
    phase_before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"phase_filtering={explain_filtering(phase_before, phase_df)}")
    phase_df = phase_df[["run_id", "phase_name", "pcie_total_bytes_s_mean"]].copy()
    phase_df["phase_name"] = phase_df["phase_name"].astype(str).str.lower()
    phase_df = phase_df[phase_df["phase_name"] == "training"].copy()
    phase_df["pcie_total_bytes_s_mean"] = pd.to_numeric(phase_df["pcie_total_bytes_s_mean"], errors="coerce")
    phase_df = phase_df.dropna(subset=["pcie_total_bytes_s_mean"]).copy()
    phase_df = (
        phase_df.groupby("run_id", dropna=False)["pcie_total_bytes_s_mean"]
        .mean()
        .reset_index(name="training_pcie_bytes_s")
    )

    sm, _ = load_view("step_metrics_long")
    sm = sm[sm["run_id"].astype(str).isin(run_ids)].copy()
    sm_before = sm.copy()
    sm = apply_analysis_ok(sm)
    print(f"metric_filtering={explain_filtering(sm_before, sm)}")
    sm = sm[sm["metric_key"] == "timing_per_token_ms/update_actor"].copy()
    sm["metric_value_float"] = pd.to_numeric(sm["metric_value_float"], errors="coerce")
    sm = sm.dropna(subset=["metric_value_float"]).copy()
    update_actor = (
        sm.groupby("run_id", dropna=False)["metric_value_float"]
        .mean()
        .reset_index(name="update_actor_ms_per_token")
    )

    df = selected_runs.merge(phase_df, on="run_id", how="inner").merge(update_actor, on="run_id", how="inner")
    df["training_pcie_gb_s"] = df["training_pcie_bytes_s"] / 1e9
    print("run summary:")
    print(df.sort_values(["policy_norm", "config"]).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for _, row in df.sort_values(["policy_norm", "config"]).iterrows():
        ax.scatter(
            row["training_pcie_gb_s"],
            row["update_actor_ms_per_token"],
            s=140,
            marker=CONFIG_MARKERS[row["config"]],
            color=POLICY_COLORS[row["policy_norm"]],
            edgecolors="black",
            linewidths=0.9,
            alpha=0.95,
            zorder=3,
        )

    scatter_paper(ax)
    ax.set_xlabel("Training PCIe throughput (GB/s)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Update actor time per token (ms)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("Training PCIe vs Update Actor Time", fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_axisbelow(True)

    policy_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POLICY_COLORS[p], markeredgecolor="black", markersize=8, label=POLICY_DISPLAY[p])
        for p in POLICY_ORDER
    ]
    config_handles = [
        Line2D([0], [0], marker=CONFIG_MARKERS[c], color="black", linestyle="None", markersize=8, label=CONFIG_DISPLAY[c])
        for c in CONFIG_ORDER
    ]
    fig.legend(handles=policy_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.37, 0.92), fontsize=LEGEND_FONT_SIZE)
    fig.legend(handles=config_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.79, 0.92), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="training_pcie_vs_update_actor",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "step_metrics_long", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
            "metrics": {
                "x": "training pcie_total_bytes_s_mean",
                "y": "timing_per_token_ms/update_actor",
            },
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
