"""Phase-segmented communication tax for llama scaling runs."""

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


OUTPATH = Path("plots/out/scale/communication_tax_comm_fraction_by_phase_policy.png")
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
CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}
CONFIG_COLORS = {
    "2xA100": "#C8553D",
    "2xH200": "#4C956C",
    "4xA100": "#D4A373",
    "4xH200": "#577590",
}

COMM_PHASE_MAP = {
    "comm_s/gen": "rollout",
    "comm_s/gen_max": "rollout",
    "comm_s/old_log_prob": "rl_policy",
    "comm_s/values": "rl_policy",
    "comm_s/update_actor": "training",
    "comm_s/update_critic": "training",
}

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
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
    return selected[["run_id", "policy_norm", "config"]].drop_duplicates()


def main() -> None:
    selected_runs = _select_scaling_runs()
    run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_fact, _ = load_view("phase_fact_view")
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
    phase_cols = [c for c in phase_cols if c in phase_fact.columns]
    pf = phase_fact[phase_fact["run_id"].astype(str).isin(run_ids)][phase_cols].copy()
    pf_before = pf.copy()
    pf = apply_analysis_ok(pf)
    print(f"phase_filtering={explain_filtering(pf_before, pf)}")
    pf["phase_name"] = pf["phase_name"].astype(str).str.lower()
    pf = pf[pf["phase_name"].isin(PHASE_ORDER)].copy()
    pf["global_step_canonical"] = pd.to_numeric(pf["global_step_canonical"], errors="coerce")
    pf["phase_time_s"] = pd.to_numeric(pf["phase_time_s"], errors="coerce")
    pf = pf.dropna(subset=["global_step_canonical", "phase_time_s"]).copy()
    pf["global_step_canonical"] = pf["global_step_canonical"].astype(int)
    if pf.empty:
        raise ValueError("No phase rows remain after filtering for selected scaling runs.")

    valid_steps = pf[["run_id", "global_step_canonical"]].drop_duplicates().copy()

    step_long, _ = load_view("step_metrics_long")
    needed_long = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    missing_long = [c for c in needed_long if c not in step_long.columns]
    if missing_long:
        raise ValueError(f"step_metrics_long missing required columns: {missing_long}")
    comm = step_long[step_long["run_id"].astype(str).isin(run_ids)][needed_long].copy()
    comm = comm[comm["metric_key"].isin(COMM_PHASE_MAP)].copy()
    comm["global_step_canonical"] = pd.to_numeric(comm["global_step_canonical"], errors="coerce")
    comm["metric_value_float"] = pd.to_numeric(comm["metric_value_float"], errors="coerce")
    comm = comm.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    comm["global_step_canonical"] = comm["global_step_canonical"].astype(int)
    comm = comm.merge(valid_steps, on=["run_id", "global_step_canonical"], how="inner")
    comm["phase_name"] = comm["metric_key"].map(COMM_PHASE_MAP)

    comm_by_phase = (
        comm.groupby(["run_id", "global_step_canonical", "phase_name"], dropna=False)["metric_value_float"]
        .sum(min_count=1)
        .reset_index(name="comm_s_phase")
    )

    df = pf.merge(comm_by_phase, on=["run_id", "global_step_canonical", "phase_name"], how="left")
    df["comm_s_phase"] = pd.to_numeric(df["comm_s_phase"], errors="coerce").fillna(0.0)
    df = df[df["phase_time_s"] > 0].copy()
    df["comm_fraction_phase"] = df["comm_s_phase"] / df["phase_time_s"]
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    df = df[df["policy_norm"].isin(POLICY_ORDER) & df["config"].isin(CONFIG_ORDER)].copy()
    df = df.dropna(subset=["comm_fraction_phase"]).copy()

    if df.empty:
        raise ValueError("No valid comm-fraction rows after joins/filters.")

    summary = (
        df.groupby(["phase_name", "policy_norm", "config"], dropna=False)["comm_fraction_phase"]
        .agg(n_phases="size", mean="mean", median="median", p90=lambda s: s.quantile(0.90))
        .reset_index()
        .sort_values(["phase_name", "policy_norm", "config"])
    )
    run_means = (
        df.groupby(["phase_name", "policy_norm", "config", "run_id"], dropna=False)["comm_fraction_phase"]
        .mean()
        .reset_index(name="run_mean_comm_fraction")
        .sort_values(["phase_name", "policy_norm", "config", "run_id"])
    )

    print("phase x policy x config summary:")
    print(summary.to_string(index=False))
    print("phase x policy x config x run means:")
    print(run_means.to_string(index=False))

    rng = np.random.default_rng(17)
    global_top = float(df["comm_fraction_phase"].quantile(0.995))
    y_max = global_top * 1.18 if global_top > 0 else 1.0
    x_map = {config: i for i, config in enumerate(CONFIG_ORDER)}

    fig, axes = plt.subplots(len(PHASE_ORDER), len(POLICY_ORDER), figsize=(17.0, 11.2), sharey="row")
    for r, phase in enumerate(PHASE_ORDER):
        for c, policy in enumerate(POLICY_ORDER):
            ax = axes[r, c]
            panel = df[(df["phase_name"] == phase) & (df["policy_norm"] == policy)].copy()
            if panel.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            box_data: list[np.ndarray] = []
            box_pos: list[int] = []
            for config in CONFIG_ORDER:
                sub = panel[panel["config"] == config].copy()
                if sub.empty:
                    continue
                x0 = x_map[config]
                jitter = rng.uniform(-0.16, 0.16, size=len(sub))
                ax.scatter(
                    x0 + jitter,
                    sub["comm_fraction_phase"],
                    s=14,
                    alpha=0.18,
                    color=CONFIG_COLORS[config],
                    edgecolors="none",
                    zorder=1,
                )
                box_data.append(sub["comm_fraction_phase"].to_numpy())
                box_pos.append(x0)

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    positions=box_pos,
                    widths=0.34,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={"edgecolor": "black", "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 1.2},
                    whiskerprops={"color": "black", "linewidth": 0.8},
                    capprops={"color": "black", "linewidth": 0.8},
                )
                for patch, pos in zip(bp["boxes"], box_pos):
                    patch.set_facecolor(CONFIG_COLORS[CONFIG_ORDER[pos]])
                    patch.set_alpha(0.35)

            rm = run_means[(run_means["phase_name"] == phase) & (run_means["policy_norm"] == policy)].copy()
            for config in CONFIG_ORDER:
                sub = rm[rm["config"] == config].copy()
                if sub.empty:
                    continue
                x0 = x_map[config]
                jitter = np.linspace(-0.08, 0.08, num=len(sub)) if len(sub) > 1 else np.array([0.0])
                ax.scatter(
                    x0 + jitter,
                    sub["run_mean_comm_fraction"],
                    s=78,
                    marker="D",
                    color=CONFIG_COLORS[config],
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.96,
                    zorder=3,
                )

            if r == 0:
                ax.set_title(POLICY_DISPLAY[policy], fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold", pad=10)
            if c == 0:
                ax.set_ylabel(f"{PHASE_DISPLAY[phase]}\ncomm_fraction/phase", fontsize=AXIS_LABEL_SIZE)

            ax.set_xticks(range(len(CONFIG_ORDER)))
            ax.set_xticklabels([CONFIG_DISPLAY[cfg] for cfg in CONFIG_ORDER], rotation=0)
            ax.tick_params(labelsize=TICK_LABEL_SIZE)
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.24)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.5, len(CONFIG_ORDER) - 0.5)
            ax.set_ylim(0, y_max)

    config_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=9,
            markerfacecolor=CONFIG_COLORS[config],
            markeredgecolor="black",
            label=CONFIG_DISPLAY[config],
        )
        for config in CONFIG_ORDER
    ]
    point_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5,
            markerfacecolor="#888888",
            markeredgecolor="none",
            alpha=0.35,
            label="Phase-level points",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=7,
            markerfacecolor="#888888",
            markeredgecolor="black",
            label="Run mean",
        ),
    ]
    leg1 = fig.legend(
        handles=config_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.37, 0.952),
        fontsize=LEGEND_FONT_SIZE,
        title="Configuration",
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=point_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.82, 0.952),
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.suptitle("Phase-Segmented Communication Tax Across Llama Scaling Runs", y=0.985, fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="communication_tax_comm_fraction_by_phase_policy",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "step_metrics_long", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
            "comm_phase_map": COMM_PHASE_MAP,
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
