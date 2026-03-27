"""GRPO subphase time versus GPU utilization and SM clock."""

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


OUTPATH = Path("plots/out/scale/grpo_subphase_time_vs_gpu_util_sm_clock.png")
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
SUBPHASE_SPECS = (
    ("training", "update_actor", "Training: Actor Update"),
    ("rl_policy", "old_log_prob", "Preparation: Old Log Prob"),
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

    phase_fact, _ = load_view("phase_fact_view")
    phase_cols = [
        "run_id",
        "global_step_canonical",
        "phase_name",
        "gpu_util_mean",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    phase_cols = [c for c in phase_cols if c in phase_fact.columns]
    pf = phase_fact[phase_fact["run_id"].astype(str).isin(run_ids)][phase_cols].copy()
    before = pf.copy()
    pf = apply_analysis_ok(pf)
    print(f"phase_filtering={explain_filtering(before, pf)}")
    pf["phase_name"] = pf["phase_name"].astype(str).str.lower()
    pf = pf[pf["phase_name"].isin([phase for phase, _, _ in SUBPHASE_SPECS])].copy()
    pf["global_step_canonical"] = pd.to_numeric(pf["global_step_canonical"], errors="coerce")
    pf["gpu_util_mean"] = pd.to_numeric(pf["gpu_util_mean"], errors="coerce")
    pf = pf.dropna(subset=["global_step_canonical"]).copy()
    pf["global_step_canonical"] = pf["global_step_canonical"].astype(int)

    dts, _ = load_view("device_timeseries_view")
    dts_cols = [
        "run_id",
        "global_step_canonical",
        "phase_name",
        "device_kind",
        "sm_clock_mhz",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    dts_cols = [c for c in dts_cols if c in dts.columns]
    ddf = dts[dts["run_id"].astype(str).isin(run_ids)][dts_cols].copy()
    d_before = ddf.copy()
    ddf = apply_analysis_ok(ddf)
    print(f"dts_filtering={explain_filtering(d_before, ddf)}")
    ddf["phase_name"] = ddf["phase_name"].astype(str).str.lower()
    ddf = ddf[ddf["phase_name"].isin([phase for phase, _, _ in SUBPHASE_SPECS])].copy()
    if "device_kind" in ddf.columns:
        ddf = ddf[ddf["device_kind"].astype(str).str.lower() == "gpu"].copy()
    ddf["global_step_canonical"] = pd.to_numeric(ddf["global_step_canonical"], errors="coerce")
    ddf["sm_clock_mhz"] = pd.to_numeric(ddf["sm_clock_mhz"], errors="coerce")
    ddf = ddf.dropna(subset=["global_step_canonical"]).copy()
    ddf["global_step_canonical"] = ddf["global_step_canonical"].astype(int)
    clocks = (
        ddf.groupby(["run_id", "global_step_canonical", "phase_name"], dropna=False)["sm_clock_mhz"]
        .mean()
        .reset_index()
    )

    phase_timings, _ = load_view("phase_timings_long")
    pt = phase_timings[phase_timings["run_id"].astype(str).isin(run_ids)].copy()
    pt["phase_name"] = pt["phase_name"].astype(str).str.lower()
    pt = pt[pt["metric_unit"].astype(str) == "s"].copy()
    keep_pairs = {(phase, subphase) for phase, subphase, _ in SUBPHASE_SPECS}
    pt = pt[pt.apply(lambda r: (str(r["phase_name"]).lower(), str(r["subphase_name"])) in keep_pairs, axis=1)].copy()
    pt["global_step_canonical"] = pd.to_numeric(pt["global_step_canonical"], errors="coerce")
    pt["value"] = pd.to_numeric(pt["value"], errors="coerce")
    pt = pt.dropna(subset=["global_step_canonical", "value"]).copy()
    pt["global_step_canonical"] = pt["global_step_canonical"].astype(int)
    subphase = (
        pt.groupby(["run_id", "global_step_canonical", "phase_name", "subphase_name"], dropna=False)["value"]
        .sum(min_count=1)
        .reset_index(name="subphase_time_s")
    )

    df = subphase.merge(
        pf[["run_id", "global_step_canonical", "phase_name", "gpu_util_mean"]],
        on=["run_id", "global_step_canonical", "phase_name"],
        how="inner",
    )
    df = df.merge(clocks, on=["run_id", "global_step_canonical", "phase_name"], how="left")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    print("subphase versus hardware summary by run:")
    print(
        df.groupby(["config", "phase_name", "subphase_name"], dropna=False)[["subphase_time_s", "gpu_util_mean", "sm_clock_mhz"]]
        .mean()
        .reset_index()
        .sort_values(["config", "phase_name"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.0), sharey="row")
    for r, (phase_name, subphase_name, panel_title) in enumerate(SUBPHASE_SPECS):
        sub = df[(df["phase_name"] == phase_name) & (df["subphase_name"].astype(str) == subphase_name)].copy()
        for c, (x_col, x_label) in enumerate((("gpu_util_mean", "GPU Utilization (%)"), ("sm_clock_mhz", "SM Clock (MHz)"))):
            ax = axes[r, c]
            for config in CONFIG_ORDER:
                g = sub[sub["config"] == config].copy()
                g = g.dropna(subset=[x_col, "subphase_time_s"])
                if g.empty:
                    continue
                ax.scatter(
                    g[x_col],
                    g["subphase_time_s"],
                    s=70,
                    color=CONFIG_COLORS[config],
                    marker=CONFIG_MARKERS[config],
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.9,
                )
            if r == 0:
                ax.set_title(x_label, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{panel_title}\nTime (s)", fontsize=AXIS_LABEL_SIZE)
            ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
            ax.grid(axis="both", alpha=0.24, linestyle="--", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=TICK_LABEL_SIZE)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            color=CONFIG_COLORS[config],
            marker=CONFIG_MARKERS[config],
            markersize=8,
            markeredgecolor="black",
            label=CONFIG_DISPLAY[config],
        )
        for config in CONFIG_ORDER
    ]
    fig.suptitle("GRPO Subphase Time vs GPU Utilization and SM Clock", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.94), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="grpo_subphase_time_vs_gpu_util_sm_clock",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "phase_timings_long", "device_timeseries_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
