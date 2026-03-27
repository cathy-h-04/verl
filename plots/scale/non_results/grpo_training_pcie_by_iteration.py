"""Training-phase PCIe diagnostics over iteration for GRPO scaling runs."""

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


OUTPATH = Path("plots/out/scale/grpo_training_pcie_by_iteration.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

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
METRICS = (
    ("pcie_rx_gbps", "PCIe RX (GB/s)", "pcie_rx_bytes_s"),
    ("pcie_tx_gbps", "PCIe TX (GB/s)", "pcie_tx_bytes_s"),
    ("pcie_total_gbps", "PCIe Total (GB/s)", "pcie_total_bytes_s"),
)

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
GB = 1e9


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
    if selected.empty:
        raise ValueError("No GRPO llama_scaling runs selected.")
    return selected[["run_id", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_grpo_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    dts, _ = load_view("device_timeseries_view")
    needed = [
        "run_id",
        "global_step_canonical",
        "phase_name",
        "device_kind",
        "pcie_rx_bytes_s",
        "pcie_tx_bytes_s",
        "pcie_total_bytes_s",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    needed = [c for c in needed if c in dts.columns]
    df = dts[dts["run_id"].astype(str).isin(run_ids)][needed].copy()
    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")
    df = df[df["phase_name"].astype(str).str.lower() == "training"].copy()
    if "device_kind" in df.columns:
        df = df[df["device_kind"].astype(str).str.lower() == "gpu"].copy()
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    for _, _, raw_col in METRICS:
        df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
    df = df.dropna(subset=["global_step_canonical"]).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    by_iter = (
        df.groupby(["run_id", "config", "global_step_canonical"], dropna=False)[[raw_col for _, _, raw_col in METRICS]]
        .mean()
        .reset_index()
        .sort_values(["config", "global_step_canonical"])
    )
    by_iter["pcie_rx_gbps"] = by_iter["pcie_rx_bytes_s"] / GB
    by_iter["pcie_tx_gbps"] = by_iter["pcie_tx_bytes_s"] / GB
    by_iter["pcie_total_gbps"] = by_iter["pcie_total_bytes_s"] / GB
    print("training PCIe summary by run (GB/s):")
    print(by_iter.groupby(["config", "run_id"], dropna=False)[["pcie_rx_gbps", "pcie_tx_gbps", "pcie_total_gbps"]].mean().reset_index().to_string(index=False))

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(13.8, 10.6), sharex=True)
    for ax, (metric, label, _) in zip(axes, METRICS):
        for config in CONFIG_ORDER:
            sub = by_iter[by_iter["config"] == config].sort_values("global_step_canonical")
            if sub.empty:
                continue
            ax.plot(
                sub["global_step_canonical"],
                sub[metric],
                color=CONFIG_COLORS[config],
                linewidth=2.3,
                marker=CONFIG_MARKERS[config],
                markersize=7.0,
                alpha=0.96,
            )
        ax.set_title(label, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_ylabel(label, fontsize=AXIS_LABEL_SIZE)
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
            markersize=7.0,
            label=CONFIG_DISPLAY[config],
        )
        for config in CONFIG_ORDER
    ]
    fig.suptitle("GRPO Training PCIe Diagnostics by Iteration", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="grpo_training_pcie_by_iteration",
        run_ids=run_ids,
        data_sources={"root": "results/monitoring_val/llama_scaling", "views": ["device_timeseries_view", "runs", "run_summary_view"], "filter": "apply_analysis_ok"},
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
