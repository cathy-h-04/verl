"""PPO subphase decomposition by configuration."""

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


OUTPATH = Path("plots/out/scale/subphase_decomp.png")
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
METRIC_ORDER = (
    "timing_per_token_ms/gen",
    "timing_per_token_ms/update_critic",
    "timing_per_token_ms/update_actor",
)
METRIC_DISPLAY = {
    "timing_per_token_ms/gen": "Generation",
    "timing_per_token_ms/update_critic": "Critic Update",
    "timing_per_token_ms/update_actor": "Actor Update",
}

FIGURE_TITLE_SIZE = 18
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 11
SUBPHASE_COLORS = {
    "timing_per_token_ms/gen": "#4C78A8",
    "timing_per_token_ms/update_critic": "#ECA82C",
    "timing_per_token_ms/update_actor": "#F58518",
}
SUBPHASE_PHASE = {
    "timing_per_token_ms/gen": "rollout",
    "timing_per_token_ms/update_critic": "training",
    "timing_per_token_ms/update_actor": "training",
}


def _annotation_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "black" if luminance > 155 else "white"


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


def _select_ppo_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[(selected["policy_norm"] == "ppo") & selected["config"].isin(CONFIG_ORDER)].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    if selected.empty:
        raise ValueError("No PPO llama_scaling runs selected.")
    return selected[["run_id", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_ppo_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    sm, _ = load_view("step_metrics_long")
    sm = sm[sm["run_id"].astype(str).isin(run_ids)].copy()
    before = sm.copy()
    sm = apply_analysis_ok(sm)
    print(f"metric_filtering={explain_filtering(before, sm)}")

    sm = sm[sm["metric_key"].isin(METRIC_ORDER)].copy()
    sm["metric_value_float"] = pd.to_numeric(sm["metric_value_float"], errors="coerce")
    sm = sm.dropna(subset=["metric_value_float"]).copy()
    sm = sm.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    summary = (
        sm.groupby(["config", "metric_key"], dropna=False)
        .agg(
            n_rows=("metric_value_float", "size"),
            mean_ms_per_token=("metric_value_float", "mean"),
            median_ms_per_token=("metric_value_float", "median"),
        )
        .reset_index()
        .sort_values(["metric_key", "config"])
    )
    print("ppo timing summary:")
    print(summary.to_string(index=False))

    phase_fact, _ = load_view("phase_fact_view")
    pf = phase_fact[phase_fact["run_id"].astype(str).isin(run_ids)].copy()
    pf_before = pf.copy()
    pf = apply_analysis_ok(pf)
    print(f"phase_filtering={explain_filtering(pf_before, pf)}")
    pf = pf[["run_id", "phase_name", "avg_power_w"]].copy()
    pf["phase_name"] = pf["phase_name"].astype(str).str.lower()
    pf = pf[pf["phase_name"].isin(["rollout", "rl_policy", "training"])].copy()
    pf["avg_power_w"] = pd.to_numeric(pf["avg_power_w"], errors="coerce")
    pf = pf.dropna(subset=["avg_power_w"]).copy()
    pf = pf.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    power_summary = (
        pf.groupby(["config", "phase_name"], dropna=False)["avg_power_w"]
        .mean()
        .reset_index()
    )

    piv = (
        summary.pivot_table(index="config", columns="metric_key", values="mean_ms_per_token", aggfunc="last")
        .reindex(CONFIG_ORDER)
        .reindex(columns=METRIC_ORDER)
    )
    totals = piv.sum(axis=1, min_count=1)

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    x = range(len(CONFIG_ORDER))
    bottom = pd.Series(0.0, index=piv.index, dtype=float)
    for metric in METRIC_ORDER:
        heights = piv[metric].fillna(0.0)
        bars = ax.bar(
            x,
            heights.values,
            bottom=bottom.values,
            color=SUBPHASE_COLORS[metric],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            label=METRIC_DISPLAY[metric],
        )
        phase_name = SUBPHASE_PHASE[metric]
        for idx, (bar, height) in enumerate(zip(bars, heights.values)):
            if height <= 0.08:
                continue
            config = CONFIG_ORDER[idx]
            match = power_summary[
                (power_summary["config"] == config) & (power_summary["phase_name"] == phase_name)
            ]["avg_power_w"]
            if match.empty:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bottom.iloc[idx] + (height / 2.0),
                f"{float(match.iloc[0]):.0f} W",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=_annotation_color(SUBPHASE_COLORS[metric]),
                zorder=4,
            )
        bottom = bottom + heights

    ax.scatter(
        list(x),
        totals.values,
        marker="D",
        s=62,
        color="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=4,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([CONFIG_DISPLAY[c] for c in CONFIG_ORDER])
    ax.set_ylabel("Time per token (ms)", fontsize=AXIS_LABEL_SIZE)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(float(totals.max()) * 1.16, 0.1))

    fig.suptitle("PPO Subphase Decomposition", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", y=0.98)
    handles = [Patch(facecolor=SUBPHASE_COLORS[m], edgecolor="black", label=METRIC_DISPLAY[m]) for m in METRIC_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="subphase_decomp",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["step_metrics_long", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
            "metrics": list(METRIC_ORDER),
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
