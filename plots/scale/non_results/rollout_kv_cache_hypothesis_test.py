"""Test rollout KV-cache / offload hypothesis across scaling runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/rollout_kv_cache_hypothesis_test.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
POLICY_MARKERS = {"ppo": "o", "remax": "s", "grpo": "^"}
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
FAMILY_COLORS = {"A100": "#B2472F", "H200": "#1D7A6F"}
X_SPECS = (
    ("rollout_mem_util_mean", "Rollout mem_util_pct"),
    ("rollout_pcie_tx_gbps", "Rollout PCIe TX (GB/s)"),
    ("cpu_memory_used_gb", "perf/cpu_memory_used_gb"),
)

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 11
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


def _family_from_config(config: str) -> str:
    if "A100" in str(config):
        return "A100"
    if "H200" in str(config):
        return "H200"
    return "Other"


def _select_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected["family"] = selected["config"].map(_family_from_config)
    selected = selected[selected["policy_norm"].isin(POLICY_ORDER) & selected["config"].isin(CONFIG_ORDER)].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    if selected.empty:
        raise ValueError("No llama_scaling runs selected.")
    return selected[["run_id", "policy_norm", "config", "family"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    filter_cols = [
        "run_id",
        "global_step_canonical",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    filter_cols = [c for c in filter_cols if c in step_fact.columns]
    filt = step_fact[step_fact["run_id"].astype(str).isin(run_ids)][filter_cols].copy()
    before = filt.copy()
    filt = apply_analysis_ok(filt)
    print(f"step_filtering={explain_filtering(before, filt)}")
    keys = filt[["run_id", "global_step_canonical"]].drop_duplicates()

    long_df, _ = load_view("step_metrics_long")
    metric_df = long_df[long_df["run_id"].astype(str).isin(run_ids)][["run_id", "global_step_canonical", "metric_key", "metric_value_float"]].copy()
    metric_df["global_step_canonical"] = pd.to_numeric(metric_df["global_step_canonical"], errors="coerce")
    metric_df["metric_value_float"] = pd.to_numeric(metric_df["metric_value_float"], errors="coerce")
    metric_df = metric_df.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    metric_df["global_step_canonical"] = metric_df["global_step_canonical"].astype(int)
    metric_df = metric_df[metric_df["metric_key"].isin(["timing_per_token_ms/gen"])].copy()
    metric_df = metric_df.merge(keys, on=["run_id", "global_step_canonical"], how="inner")
    decode = (
        metric_df.pivot_table(
            index=["run_id", "global_step_canonical"],
            columns="metric_key",
            values="metric_value_float",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns={"timing_per_token_ms/gen": "timing_per_token_ms_gen"})
    )

    wide, _ = load_view("step_metrics_wide_curated")
    cpu = wide[wide["run_id"].astype(str).isin(run_ids)][["run_id", "global_step_canonical", "metric_perf_cpu_memory_used_gb"]].copy()
    cpu["global_step_canonical"] = pd.to_numeric(cpu["global_step_canonical"], errors="coerce")
    cpu["metric_perf_cpu_memory_used_gb"] = pd.to_numeric(cpu["metric_perf_cpu_memory_used_gb"], errors="coerce")
    cpu = cpu.dropna(subset=["global_step_canonical"]).copy()
    cpu["global_step_canonical"] = cpu["global_step_canonical"].astype(int)
    cpu = cpu.merge(keys, on=["run_id", "global_step_canonical"], how="inner")
    cpu = cpu.rename(columns={"metric_perf_cpu_memory_used_gb": "cpu_memory_used_gb"})

    phase_fact, _ = load_view("phase_fact_view")
    pf_cols = ["run_id", "global_step_canonical", "phase_name", "mem_util_mean"]
    pf = phase_fact[phase_fact["run_id"].astype(str).isin(run_ids)][pf_cols].copy()
    pf["phase_name"] = pf["phase_name"].astype(str).str.lower()
    pf = pf[pf["phase_name"] == "rollout"].copy()
    pf["global_step_canonical"] = pd.to_numeric(pf["global_step_canonical"], errors="coerce")
    pf["mem_util_mean"] = pd.to_numeric(pf["mem_util_mean"], errors="coerce")
    pf = pf.dropna(subset=["global_step_canonical"]).copy()
    pf["global_step_canonical"] = pf["global_step_canonical"].astype(int)
    pf = pf.merge(keys, on=["run_id", "global_step_canonical"], how="inner")
    pf = pf.rename(columns={"mem_util_mean": "rollout_mem_util_mean"})

    dts, _ = load_view("device_timeseries_view")
    dcols = ["run_id", "global_step_canonical", "phase_name", "device_kind", "pcie_tx_bytes_s"]
    ddf = dts[dts["run_id"].astype(str).isin(run_ids)][dcols].copy()
    d_before = ddf.copy()
    ddf = apply_analysis_ok(ddf)
    print(f"dts_filtering={explain_filtering(d_before, ddf)}")
    ddf["phase_name"] = ddf["phase_name"].astype(str).str.lower()
    ddf = ddf[ddf["phase_name"] == "rollout"].copy()
    if "device_kind" in ddf.columns:
        ddf = ddf[ddf["device_kind"].astype(str).str.lower() == "gpu"].copy()
    ddf["global_step_canonical"] = pd.to_numeric(ddf["global_step_canonical"], errors="coerce")
    ddf["pcie_tx_bytes_s"] = pd.to_numeric(ddf["pcie_tx_bytes_s"], errors="coerce")
    ddf = ddf.dropna(subset=["global_step_canonical", "pcie_tx_bytes_s"]).copy()
    ddf["global_step_canonical"] = ddf["global_step_canonical"].astype(int)
    pcie = (
        ddf.groupby(["run_id", "global_step_canonical"], dropna=False)["pcie_tx_bytes_s"]
        .mean()
        .reset_index(name="rollout_pcie_tx_bytes_s")
    )
    pcie["rollout_pcie_tx_gbps"] = pcie["rollout_pcie_tx_bytes_s"] / GB

    df = decode.merge(cpu, on=["run_id", "global_step_canonical"], how="left")
    df = df.merge(pf[["run_id", "global_step_canonical", "rollout_mem_util_mean"]], on=["run_id", "global_step_canonical"], how="left")
    df = df.merge(pcie[["run_id", "global_step_canonical", "rollout_pcie_tx_gbps"]], on=["run_id", "global_step_canonical"], how="left")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    run_means = (
        df.groupby(["run_id", "policy_norm", "config", "family"], dropna=False)[["timing_per_token_ms_gen", "cpu_memory_used_gb", "rollout_mem_util_mean", "rollout_pcie_tx_gbps"]]
        .mean()
        .reset_index()
        .sort_values(["family", "config", "policy_norm"])
    )
    print("run-level rollout hypothesis summary:")
    print(run_means.to_string(index=False))

    fig, axes = plt.subplots(1, len(X_SPECS), figsize=(16.2, 5.6), sharey=True)
    for ax, (x_col, x_label) in zip(axes, X_SPECS):
        for config in CONFIG_ORDER:
            sub = run_means[run_means["config"] == config].dropna(subset=[x_col, "timing_per_token_ms_gen"]).copy()
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub["timing_per_token_ms_gen"],
                s=90,
                color=CONFIG_COLORS[config],
                marker="o",
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                zorder=3,
            )
            if len(sub) >= 2:
                x = sub[x_col].to_numpy(dtype=float)
                y = sub["timing_per_token_ms_gen"].to_numpy(dtype=float)
                coef = np.polyfit(x, y, deg=1)
                x_line = np.linspace(float(x.min()), float(x.max()), 100)
                y_line = coef[0] * x_line + coef[1]
                ax.plot(x_line, y_line, color=CONFIG_COLORS[config], linewidth=1.8, alpha=0.9, zorder=2)

        for family in ("A100", "H200"):
            fam = run_means[run_means["family"] == family].dropna(subset=[x_col, "timing_per_token_ms_gen"]).copy()
            if len(fam) >= 2:
                x = fam[x_col].to_numpy(dtype=float)
                y = fam["timing_per_token_ms_gen"].to_numpy(dtype=float)
                coef = np.polyfit(x, y, deg=1)
                x_line = np.linspace(float(x.min()), float(x.max()), 100)
                y_line = coef[0] * x_line + coef[1]
                ax.plot(x_line, y_line, color=FAMILY_COLORS[family], linewidth=2.8, linestyle="--", alpha=0.8, zorder=1)

        ax.set_title(x_label, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
        ax.grid(axis="both", alpha=0.24, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    axes[0].set_ylabel("timing_per_token_ms/gen", fontsize=AXIS_LABEL_SIZE)

    config_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor=CONFIG_COLORS[cfg], markeredgecolor="black", label=CONFIG_DISPLAY[cfg])
        for cfg in CONFIG_ORDER
    ]
    family_handles = [
        Line2D([0], [0], color=FAMILY_COLORS[fam], linestyle="--", linewidth=2.5, label=f"{fam} family fit")
        for fam in ("A100", "H200")
    ]
    fig.suptitle("Rollout KV-Cache Hypothesis Test", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=config_handles + family_handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="rollout_kv_cache_hypothesis_test",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["step_fact_view", "step_metrics_long", "step_metrics_wide_curated", "phase_fact_view", "device_timeseries_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
