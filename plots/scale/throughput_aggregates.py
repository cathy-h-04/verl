"""Phase-native throughput aggregates by configuration across scaling runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/throughput_aggregates.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
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


def _load_remax_gen_ratio(run_ids: list[str]) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame(columns=["run_id", "global_step_canonical", "genmax_over_gen"])

    pt, _ = load_view("phase_timings_long")
    df = pt[
        pt["run_id"].astype(str).isin(run_ids)
        & pt["subphase_name"].isin(["gen", "gen_max"])
        & pt["metric_unit"].astype(str).eq("s")
    ][["run_id", "global_step_canonical", "subphase_name", "value"]].copy()
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["global_step_canonical", "value"]).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)

    piv = (
        df.pivot_table(index=["run_id", "global_step_canonical"], columns="subphase_name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    piv["genmax_over_gen"] = piv["gen_max"] / piv["gen"]
    piv.loc[~np.isfinite(piv["genmax_over_gen"]), "genmax_over_gen"] = np.nan
    return piv[["run_id", "global_step_canonical", "genmax_over_gen"]]


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    sf = step_fact[step_fact["run_id"].astype(str).isin(run_ids)].copy()
    sf_before = sf.copy()
    sf = apply_analysis_ok(sf)
    print(f"step_filtering={explain_filtering(sf_before, sf)}")
    keep_step_cols = [
        "run_id",
        "global_step_canonical",
        "step_rollout_output_tokens",
        "step_rollout_total_tokens",
        "step_train_tokens_est",
        "policy",
    ]
    sf = sf[keep_step_cols].copy()
    for col in ["global_step_canonical", "step_rollout_output_tokens", "step_rollout_total_tokens", "step_train_tokens_est"]:
        sf[col] = pd.to_numeric(sf[col], errors="coerce")
    sf = sf.dropna(subset=["global_step_canonical", "step_rollout_output_tokens", "step_rollout_total_tokens"]).copy()
    sf["global_step_canonical"] = sf["global_step_canonical"].astype(int)
    sf["policy_norm"] = sf["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    sf["step_rollout_prompt_tokens"] = sf["step_rollout_total_tokens"] - sf["step_rollout_output_tokens"]

    remax_ids = sorted(sf.loc[sf["policy_norm"] == "remax", "run_id"].astype(str).unique().tolist())
    sf = sf.merge(_load_remax_gen_ratio(remax_ids), on=["run_id", "global_step_canonical"], how="left")
    sf["genmax_over_gen"] = pd.to_numeric(sf["genmax_over_gen"], errors="coerce")
    run_medians = sf.groupby("run_id")["genmax_over_gen"].median().to_dict()
    sf["genmax_over_gen"] = sf.apply(
        lambda row: run_medians.get(row["run_id"], np.nan) if pd.isna(row["genmax_over_gen"]) else row["genmax_over_gen"],
        axis=1,
    )

    sf["corrected_rollout_tokens"] = sf["step_rollout_total_tokens"]
    remax_mask = sf["policy_norm"] == "remax"
    sf.loc[remax_mask, "corrected_rollout_tokens"] = (
        sf.loc[remax_mask, "step_rollout_total_tokens"]
        + sf.loc[remax_mask, "step_rollout_prompt_tokens"]
        + sf.loc[remax_mask, "step_rollout_output_tokens"] * sf.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )

    phase_fact, _ = load_view("phase_fact_view")
    pf = phase_fact[phase_fact["run_id"].astype(str).isin(run_ids)].copy()
    pf_before = pf.copy()
    pf = apply_analysis_ok(pf)
    print(f"phase_filtering={explain_filtering(pf_before, pf)}")
    keep_phase_cols = ["run_id", "global_step_canonical", "phase_name", "phase_time_s"]
    pf = pf[keep_phase_cols].copy()
    pf["phase_name"] = pf["phase_name"].astype(str).str.lower()
    pf = pf[pf["phase_name"].isin(PHASE_ORDER)].copy()
    pf["global_step_canonical"] = pd.to_numeric(pf["global_step_canonical"], errors="coerce")
    pf["phase_time_s"] = pd.to_numeric(pf["phase_time_s"], errors="coerce")
    pf = pf.dropna(subset=["global_step_canonical", "phase_time_s"]).copy()
    pf["global_step_canonical"] = pf["global_step_canonical"].astype(int)
    pf = pf[pf["phase_time_s"] > 0].copy()

    df = pf.merge(
        sf[
            [
                "run_id",
                "global_step_canonical",
                "corrected_rollout_tokens",
                "step_train_tokens_est",
                "policy_norm",
            ]
        ],
        on=["run_id", "global_step_canonical"],
        how="inner",
    )
    df = df.merge(selected_runs[["run_id", "config"]], on="run_id", how="inner", validate="many_to_one")

    df["normalization_tokens"] = np.nan
    phase_rollout_mask = df["phase_name"].isin(["rollout", "rl_policy"])
    phase_train_mask = df["phase_name"] == "training"
    df.loc[phase_rollout_mask, "normalization_tokens"] = df.loc[phase_rollout_mask, "corrected_rollout_tokens"]
    df.loc[phase_train_mask, "normalization_tokens"] = df.loc[phase_train_mask, "step_train_tokens_est"]
    df = df[df["normalization_tokens"] > 0].copy()
    df["corrected_throughput_tokens_s"] = df["normalization_tokens"] / df["phase_time_s"]

    print("run-phase mean throughput:")
    print(
        df.groupby(["run_id", "policy_norm", "config", "phase_name"], dropna=False)["corrected_throughput_tokens_s"]
        .mean()
        .reset_index()
        .sort_values(["policy_norm", "config", "phase_name"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(1, len(PHASE_ORDER), figsize=(16.2, 5.8), sharey=False)
    config_to_x = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    box_width = 0.46

    for ax, phase in zip(axes, PHASE_ORDER):
        sub = df[df["phase_name"] == phase].copy()
        mean_x = []
        mean_y = []
        for config in CONFIG_ORDER:
            vals = sub.loc[sub["config"] == config, "corrected_throughput_tokens_s"].dropna()
            if vals.empty:
                continue
            pos = config_to_x[config]
            bp = ax.boxplot(
                [vals.to_numpy()],
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"color": "black", "linewidth": 1.0},
                capprops={"color": "black", "linewidth": 1.0},
                boxprops={"edgecolor": "black", "linewidth": 0.9},
                zorder=2,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(CONFIG_COLORS[config])
                patch.set_alpha(0.50)
            mean_x.append(pos)
            mean_y.append(float(vals.mean()))

        if mean_x:
            ax.scatter(
                mean_x,
                mean_y,
                s=60,
                color=[CONFIG_COLORS[CONFIG_ORDER[int(x)]] for x in mean_x],
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )

        ax.set_title(PHASE_DISPLAY[phase], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.set_xticks(range(len(CONFIG_ORDER)))
        ax.set_xticklabels([CONFIG_DISPLAY[c] for c in CONFIG_ORDER], rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.set_axisbelow(True)
        ax.set_xlabel("Configuration", fontsize=AXIS_LABEL_SIZE)

    axes[0].set_ylabel("Throughput (tokens/s)", fontsize=AXIS_LABEL_SIZE)
    fig.suptitle("Throughput by Configuration and Phase", fontsize=FIGURE_TITLE_SIZE, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="throughput_aggregates",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "step_fact_view", "phase_timings_long", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
            "normalization": {
                "rollout": "corrected_rollout_tokens / phase_time_s",
                "rl_policy": "corrected_rollout_tokens / phase_time_s",
                "training": "step_train_tokens_est / phase_time_s",
                "remax_correction": "rollout_total + rollout_prompt + (gen_max/gen)*rollout_output",
            },
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
