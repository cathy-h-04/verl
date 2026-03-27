"""GRPO actor-update efficiency versus SM clock, split by hardware family."""

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


OUTPATH = Path("plots/out/scale/grpo_update_actor_ms_per_train_token_vs_sm_clock.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

CONFIG_ORDER = ("2xA100", "4xA100", "2xH200", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "4xA100": "4x A100",
    "2xH200": "2x H200",
    "4xH200": "4x H200",
}
CONFIG_COLORS = {
    "2xA100": "#E76F51",
    "4xA100": "#E9C46A",
    "2xH200": "#2A9D8F",
    "4xH200": "#457B9D",
}
CONFIG_MARKERS = {
    "2xA100": "*",
    "4xA100": "s",
    "2xH200": "o",
    "4xH200": "^",
}
FAMILY_ORDER = ("A100", "H200")
FAMILY_CONFIGS = {
    "A100": ("2xA100", "4xA100"),
    "H200": ("2xH200", "4xH200"),
}

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
MS_PER_S = 1000.0


def _config_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_a100" in rid:
        return "2xA100"
    if "4gpu_a100" in rid:
        return "4xA100"
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_h200" in rid:
        return "4xH200"
    return "Unknown"


def _family_from_config(config: str) -> str:
    if "A100" in str(config):
        return "A100"
    if "H200" in str(config):
        return "H200"
    return "Other"


def _select_grpo_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower()
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected["family"] = selected["config"].map(_family_from_config)
    selected = selected[(selected["policy_norm"] == "grpo") & selected["config"].isin(CONFIG_ORDER)].copy()
    if selected.empty:
        raise ValueError("No GRPO llama_scaling runs selected.")
    return selected[["run_id", "config", "family"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_grpo_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    step_cols = [
        "run_id",
        "global_step_canonical",
        "step_train_tokens_est",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    step_cols = [c for c in step_cols if c in step_fact.columns]
    sf = step_fact[step_fact["run_id"].astype(str).isin(run_ids)][step_cols].copy()
    sf_before = sf.copy()
    sf = apply_analysis_ok(sf)
    print(f"step_filtering={explain_filtering(sf_before, sf)}")
    sf["global_step_canonical"] = pd.to_numeric(sf["global_step_canonical"], errors="coerce")
    sf["step_train_tokens_est"] = pd.to_numeric(sf["step_train_tokens_est"], errors="coerce")
    sf = sf.dropna(subset=["global_step_canonical", "step_train_tokens_est"]).copy()
    sf["global_step_canonical"] = sf["global_step_canonical"].astype(int)

    pt, _ = load_view("phase_timings_long")
    pt = pt[pt["run_id"].astype(str).isin(run_ids)].copy()
    pt["phase_name"] = pt["phase_name"].astype(str).str.lower()
    pt = pt[
        (pt["phase_name"] == "training")
        & (pt["subphase_name"].astype(str) == "update_actor")
        & (pt["metric_unit"].astype(str) == "s")
    ].copy()
    pt["global_step_canonical"] = pd.to_numeric(pt["global_step_canonical"], errors="coerce")
    pt["value"] = pd.to_numeric(pt["value"], errors="coerce")
    pt = pt.dropna(subset=["global_step_canonical", "value"]).copy()
    pt["global_step_canonical"] = pt["global_step_canonical"].astype(int)
    pt = (
        pt.groupby(["run_id", "global_step_canonical"], dropna=False)["value"]
        .sum(min_count=1)
        .reset_index(name="update_actor_time_s")
    )

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
    ddf_before = ddf.copy()
    ddf = apply_analysis_ok(ddf)
    print(f"dts_filtering={explain_filtering(ddf_before, ddf)}")
    ddf["phase_name"] = ddf["phase_name"].astype(str).str.lower()
    ddf = ddf[ddf["phase_name"] == "training"].copy()
    if "device_kind" in ddf.columns:
        ddf = ddf[ddf["device_kind"].astype(str).str.lower() == "gpu"].copy()
    ddf["global_step_canonical"] = pd.to_numeric(ddf["global_step_canonical"], errors="coerce")
    ddf["sm_clock_mhz"] = pd.to_numeric(ddf["sm_clock_mhz"], errors="coerce")
    ddf = ddf.dropna(subset=["global_step_canonical", "sm_clock_mhz"]).copy()
    ddf["global_step_canonical"] = ddf["global_step_canonical"].astype(int)
    clocks = (
        ddf.groupby(["run_id", "global_step_canonical"], dropna=False)["sm_clock_mhz"]
        .mean()
        .reset_index()
    )

    df = sf.merge(pt, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(clocks, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    df = df[df["step_train_tokens_est"] > 0].copy()
    df["update_actor_ms_per_train_token"] = MS_PER_S * df["update_actor_time_s"] / df["step_train_tokens_est"]

    print("run-level summary:")
    print(
        df.groupby(["family", "config", "run_id"], dropna=False)[["update_actor_ms_per_train_token", "sm_clock_mhz"]]
        .mean()
        .reset_index()
        .sort_values(["family", "config"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(13.8, 5.8), sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER):
        panel = df[df["family"] == family].copy()
        for config in FAMILY_CONFIGS[family]:
            sub = panel[panel["config"] == config].copy()
            if sub.empty:
                continue
            ax.scatter(
                sub["sm_clock_mhz"],
                sub["update_actor_ms_per_train_token"],
                s=78,
                color=CONFIG_COLORS[config],
                marker=CONFIG_MARKERS[config],
                edgecolors="black",
                linewidths=0.6,
                alpha=0.92,
            )
        ax.set_title(family, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Training SM Clock (MHz)", fontsize=AXIS_LABEL_SIZE)
        ax.grid(axis="both", alpha=0.24, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    axes[0].set_ylabel("Update Actor Time (ms/train token)", fontsize=AXIS_LABEL_SIZE)

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
    fig.suptitle("GRPO Actor-Update Efficiency vs SM Clock", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="grpo_update_actor_ms_per_train_token_vs_sm_clock",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["step_fact_view", "phase_timings_long", "device_timeseries_view", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
