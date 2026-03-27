"""GRPO preparation/training subphase timing decomposition over iteration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/grpo_subphase_time_by_iteration.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
PHASE_ORDER = ("rl_policy", "training")
PHASE_DISPLAY = {
    "rl_policy": "Preparation",
    "training": "Training",
}
PHASE_SUBPHASES = {
    "rl_policy": ("reward", "old_log_prob", "adv", "residual"),
    "training": ("update_actor", "residual"),
}
SUBPHASE_DISPLAY = {
    "reward": "Reward",
    "old_log_prob": "Old Log Prob",
    "adv": "Advantage",
    "update_actor": "Actor Update",
    "residual": "Unattributed",
}
SUBPHASE_COLORS = {
    "reward": "#4C956C",
    "old_log_prob": "#2E8B57",
    "adv": "#B6992D",
    "update_actor": "#F58518",
    "residual": "#8D99AE",
}

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

    pf, _ = load_view("phase_fact_view")
    phase_cols = [
        "run_id",
        "global_step_canonical",
        "phase_name",
        "phase_time_s",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    phase_cols = [c for c in phase_cols if c in pf.columns]
    phase_df = pf[pf["run_id"].astype(str).isin(run_ids)][phase_cols].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"phase_filtering={explain_filtering(before, phase_df)}")
    phase_df["phase_name"] = phase_df["phase_name"].astype(str).str.lower()
    phase_df = phase_df[phase_df["phase_name"].isin(PHASE_ORDER)].copy()
    phase_df["global_step_canonical"] = pd.to_numeric(phase_df["global_step_canonical"], errors="coerce")
    phase_df["phase_time_s"] = pd.to_numeric(phase_df["phase_time_s"], errors="coerce")
    phase_df = phase_df.dropna(subset=["global_step_canonical", "phase_time_s"]).copy()
    phase_df["global_step_canonical"] = phase_df["global_step_canonical"].astype(int)
    phase_df = phase_df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    timings, _ = load_view("phase_timings_long")
    needed = ["run_id", "global_step_canonical", "phase_name", "subphase_name", "metric_unit", "value"]
    timing_df = timings[timings["run_id"].astype(str).isin(run_ids)][needed].copy()
    timing_df["phase_name"] = timing_df["phase_name"].astype(str).str.lower()
    timing_df = timing_df[timing_df["phase_name"].isin(PHASE_ORDER) & (timing_df["metric_unit"].astype(str) == "s")].copy()
    timing_df["global_step_canonical"] = pd.to_numeric(timing_df["global_step_canonical"], errors="coerce")
    timing_df["value"] = pd.to_numeric(timing_df["value"], errors="coerce")
    timing_df = timing_df.dropna(subset=["global_step_canonical", "value"]).copy()
    timing_df["global_step_canonical"] = timing_df["global_step_canonical"].astype(int)
    valid_keys = phase_df[["run_id", "global_step_canonical"]].drop_duplicates()
    timing_df = timing_df.merge(valid_keys, on=["run_id", "global_step_canonical"], how="inner")

    subphase_df = (
        timing_df.groupby(["run_id", "global_step_canonical", "phase_name", "subphase_name"], dropna=False)["value"]
        .sum(min_count=1)
        .reset_index(name="subphase_time_s")
    )
    totals = (
        subphase_df.groupby(["run_id", "global_step_canonical", "phase_name"], dropna=False)["subphase_time_s"]
        .sum(min_count=1)
        .reset_index(name="measured_subphase_time_s")
    )
    merged = phase_df.merge(totals, on=["run_id", "global_step_canonical", "phase_name"], how="left")
    merged["measured_subphase_time_s"] = pd.to_numeric(merged["measured_subphase_time_s"], errors="coerce").fillna(0.0)
    merged["residual_time_s"] = (merged["phase_time_s"] - merged["measured_subphase_time_s"]).clip(lower=0.0)
    residual_rows = merged[["run_id", "config", "global_step_canonical", "phase_name", "residual_time_s"]].rename(columns={"residual_time_s": "subphase_time_s"})
    residual_rows = residual_rows.assign(subphase_name="residual")

    subphase_df = subphase_df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    subphase_df = subphase_df[subphase_df.apply(lambda r: r["subphase_name"] in PHASE_SUBPHASES.get(r["phase_name"], ()), axis=1)].copy()
    stacked = pd.concat(
        [
            subphase_df[["run_id", "config", "global_step_canonical", "phase_name", "subphase_name", "subphase_time_s"]],
            residual_rows[["run_id", "config", "global_step_canonical", "phase_name", "subphase_name", "subphase_time_s"]],
        ],
        ignore_index=True,
    )

    print("subphase summary by run:")
    print(stacked.groupby(["config", "phase_name", "subphase_name"], dropna=False)["subphase_time_s"].mean().reset_index().sort_values(["config", "phase_name", "subphase_name"]).to_string(index=False))

    fig, axes = plt.subplots(len(PHASE_ORDER), len(CONFIG_ORDER), figsize=(18.0, 8.8), sharex="col", sharey="row")
    for r, phase in enumerate(PHASE_ORDER):
        subphases = PHASE_SUBPHASES[phase]
        for c, config in enumerate(CONFIG_ORDER):
            ax = axes[r, c]
            panel = stacked[(stacked["phase_name"] == phase) & (stacked["config"] == config)].copy()
            if panel.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue
            panel = panel.pivot_table(
                index="global_step_canonical",
                columns="subphase_name",
                values="subphase_time_s",
                aggfunc="sum",
                fill_value=0.0,
            ).reindex(columns=subphases, fill_value=0.0).sort_index()
            x = panel.index.to_numpy(dtype=float)
            bottom = pd.Series(0.0, index=panel.index)
            for subphase in subphases:
                vals = panel[subphase]
                ax.bar(
                    x,
                    vals.to_numpy(dtype=float),
                    bottom=bottom.to_numpy(dtype=float),
                    width=0.85,
                    color=SUBPHASE_COLORS[subphase],
                    edgecolor="none",
                    alpha=0.92,
                )
                bottom = bottom + vals
            if r == 0:
                ax.set_title(CONFIG_DISPLAY[config], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(f"{PHASE_DISPLAY[phase]}\nTime (s)", fontsize=AXIS_LABEL_SIZE)
            ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=TICK_LABEL_SIZE)

    for ax in axes[-1]:
        ax.set_xlabel("Iteration ID", fontsize=AXIS_LABEL_SIZE)

    handles = [Patch(facecolor=SUBPHASE_COLORS[subphase], edgecolor="none", label=SUBPHASE_DISPLAY[subphase]) for phase in PHASE_ORDER for subphase in PHASE_SUBPHASES[phase]]
    dedup_handles = []
    seen = set()
    for handle in handles:
        label = handle.get_label()
        if label in seen:
            continue
        seen.add(label)
        dedup_handles.append(handle)
    fig.suptitle("GRPO Preparation and Training Subphase Time by Iteration", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.legend(handles=dedup_handles, loc="upper center", ncol=len(dedup_handles), frameon=False, bbox_to_anchor=(0.5, 0.94), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="grpo_subphase_time_by_iteration",
        run_ids=run_ids,
        data_sources={"root": "results/monitoring_val/llama_scaling", "views": ["phase_fact_view", "phase_timings_long", "runs", "run_summary_view"], "filter": "apply_analysis_ok"},
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
