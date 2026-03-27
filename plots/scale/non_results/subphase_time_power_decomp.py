"""Major subphase time and average power decomposition for llama scaling runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/subphase_time_power_decomp.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")
INCLUDE_VALIDATION = False

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
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
SUBPHASE_ORDER = (
    "gen",
    "old_log_prob",
    "update_actor",
)
SUBPHASE_PARENT = {
    "gen": "rollout",
    "gen_max": "rollout",
    "old_log_prob": "rl_policy",
    "values": "rl_policy",
    "update_critic": "training",
    "update_actor": "training",
}
SUBPHASE_DISPLAY = {
    "gen": "Generation",
    "old_log_prob": "Old Log Prob",
    "update_actor": "Actor Update",
}
SUBPHASE_COLORS = {
    "gen": "#4C78A8",
    "gen_max": "#9C755F",
    "old_log_prob": "#2E8B57",
    "values": "#86BC86",
    "update_critic": "#ECA82C",
    "update_actor": "#F58518",
}
CONFIG_COLORS = {
    "2xA100": "#E76F51",
    "2xH200": "#2A9D8F",
    "4xA100": "#E9C46A",
    "4xH200": "#457B9D",
}
RECONSTRUCTION_SUBPHASE_ORDER = {
    "rollout": ("gen", "gen_max"),
    "rl_policy": ("reward", "old_log_prob", "values", "adv"),
    "training": ("update_critic", "update_actor"),
}

FIGURE_TITLE_SIZE = 18
PANEL_TITLE_SIZE = 11
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9
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


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _annotation_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "black" if luminance > 155 else "white"


def _select_runs() -> pd.DataFrame:
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


def _load_phase_fact_for_plot() -> pd.DataFrame:
    df, _ = load_view("phase_fact_view")
    required_cols = [
        "run_id",
        "phase_name",
        "phase_id",
        "phase_instance_id",
        "avg_power_w",
        "global_step_canonical",
        "phase_start_ts",
        "phase_end_ts",
    ]
    filter_cols_optional = [
        "global_step",
        "analysis_ok",
        "boundary_integrity_ok",
        "join_integrity_ok",
        "is_warmup_idle",
        "is_validation_step",
        "is_incomplete_phase",
        "is_outlier_sample",
    ]
    needed = [col for col in required_cols + filter_cols_optional if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"phase_fact_view missing required columns: {missing_required}")
    return df[needed].copy()


def _load_phase_timings() -> pd.DataFrame:
    df, _ = load_view("phase_timings_long")
    required = ["run_id", "global_step_canonical", "phase_name", "phase_id", "subphase_name", "metric_unit", "value"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"phase_timings_long missing required columns: {missing}")
    return df[required].copy()


def _load_hardware_periodic() -> pd.DataFrame:
    df, _ = load_view("hardware_periodic")
    required = ["run_id", "phase_instance_id", "device_id", "device_kind", "ts_monotonic_ns", "gpu_power_mW"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")
    return df[required].copy()


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    phase_timings = _load_phase_timings()
    selected_runs = _select_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    if phase_df.empty:
        raise ValueError(f"No phase_fact_view rows found for selected runs: {selected_run_ids}")

    phase_before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"filtering={explain_filtering(phase_before, phase_df)}")

    if not INCLUDE_VALIDATION:
        phase_df = phase_df[phase_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
    phase_df["avg_power_w"] = pd.to_numeric(phase_df["avg_power_w"], errors="coerce")
    phase_df["phase_start_ts"] = pd.to_numeric(phase_df["phase_start_ts"], errors="coerce")
    phase_df["phase_end_ts"] = pd.to_numeric(phase_df["phase_end_ts"], errors="coerce")
    phase_df = phase_df[phase_df["phase_bucket"].isin(PHASE_ORDER)].dropna(subset=["avg_power_w"]).copy()
    phase_df = phase_df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    retained_phase_keys = phase_df[
        [
            "run_id",
            "global_step_canonical",
            "phase_name",
            "phase_id",
            "phase_instance_id",
            "phase_bucket",
            "policy_norm",
            "config",
            "phase_start_ts",
            "phase_end_ts",
        ]
    ].drop_duplicates()

    phase_timings = phase_timings[
        phase_timings["run_id"].astype(str).isin(selected_run_ids) & (phase_timings["metric_unit"].astype(str) == "s")
    ].copy()
    phase_timings["value"] = pd.to_numeric(phase_timings["value"], errors="coerce")
    phase_timings = phase_timings.dropna(subset=["value"]).copy()
    phase_timings = phase_timings.merge(
        retained_phase_keys,
        on=["run_id", "global_step_canonical", "phase_name", "phase_id"],
        how="inner",
        validate="many_to_one",
    )
    if phase_timings.empty:
        raise ValueError("No retained subphase timing rows remained after filtering and join.")

    display_phase_timings = phase_timings[phase_timings["subphase_name"].isin(SUBPHASE_ORDER)].copy()

    phase_counts = (
        retained_phase_keys.groupby(["config", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_phase_instances")
        .reset_index()
    )
    subphase_sums = (
        display_phase_timings.groupby(["config", "policy_norm", "phase_bucket", "subphase_name"], dropna=False)["value"]
        .sum()
        .rename("subphase_time_s_total")
        .reset_index()
    )
    subphase_means = subphase_sums.merge(
        phase_counts,
        on=["config", "policy_norm", "phase_bucket"],
        how="left",
        validate="many_to_one",
    )
    subphase_means["mean_subphase_time_s"] = subphase_means["subphase_time_s_total"] / subphase_means["n_phase_instances"].clip(lower=1)

    hardware_periodic = _load_hardware_periodic()
    hardware_periodic = hardware_periodic[
        hardware_periodic["run_id"].astype(str).isin(selected_run_ids)
        & (hardware_periodic["device_kind"].astype(str).str.lower() == "gpu")
        & hardware_periodic["phase_instance_id"].isin(retained_phase_keys["phase_instance_id"])
    ].copy()
    hardware_periodic["ts_monotonic_ns"] = pd.to_numeric(hardware_periodic["ts_monotonic_ns"], errors="coerce")
    hardware_periodic["gpu_power_mW"] = pd.to_numeric(hardware_periodic["gpu_power_mW"], errors="coerce")
    hardware_periodic = hardware_periodic.dropna(subset=["ts_monotonic_ns", "gpu_power_mW", "device_id"]).copy()

    subphase_duration_lookup = phase_timings.groupby(["phase_instance_id", "subphase_name"], dropna=False)["value"].sum().to_dict()
    reconstructed_windows: list[dict[str, object]] = []
    for row in retained_phase_keys.itertuples(index=False):
        phase_start_ns = float(row.phase_start_ts)
        phase_end_ns = float(row.phase_end_ts)
        if not np.isfinite(phase_start_ns) or not np.isfinite(phase_end_ns) or phase_end_ns <= phase_start_ns:
            continue
        cursor_ns = phase_start_ns
        for subphase_name in RECONSTRUCTION_SUBPHASE_ORDER.get(row.phase_bucket, ()):
            duration_s = float(subphase_duration_lookup.get((row.phase_instance_id, subphase_name), 0.0) or 0.0)
            if duration_s <= 0:
                continue
            start_ns = cursor_ns
            end_ns = min(cursor_ns + (duration_s * 1e9), phase_end_ns)
            if end_ns <= start_ns:
                continue
            reconstructed_windows.append(
                {
                    "run_id": row.run_id,
                    "phase_instance_id": row.phase_instance_id,
                    "phase_bucket": row.phase_bucket,
                    "policy_norm": row.policy_norm,
                    "config": row.config,
                    "subphase_name": subphase_name,
                    "subphase_start_ts": start_ns,
                    "subphase_end_ts": end_ns,
                }
            )
            cursor_ns = end_ns

    windows_df = pd.DataFrame(reconstructed_windows)
    if windows_df.empty:
        subphase_power = pd.DataFrame(columns=["config", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"])
    else:
        hardware_periodic = hardware_periodic.sort_values(["run_id", "device_id", "ts_monotonic_ns"]).copy()
        device_groups = hardware_periodic.groupby(["run_id", "device_id"], dropna=False)
        hardware_periodic["_next_ts"] = device_groups["ts_monotonic_ns"].shift(-1)
        hardware_periodic["sample_weight_ns"] = pd.to_numeric(hardware_periodic["_next_ts"] - hardware_periodic["ts_monotonic_ns"], errors="coerce")
        fallback_weight = device_groups["sample_weight_ns"].transform(lambda s: s[s > 0].median())
        hardware_periodic["sample_weight_ns"] = hardware_periodic["sample_weight_ns"].fillna(fallback_weight).fillna(1.0).clip(lower=1.0)
        hardware_periodic["interval_end_ts"] = pd.to_numeric(hardware_periodic["ts_monotonic_ns"], errors="coerce") + hardware_periodic["sample_weight_ns"]

        periodic_by_phase = {phase_instance_id: grp.copy() for phase_instance_id, grp in hardware_periodic.groupby("phase_instance_id", dropna=False)}
        estimated_power_rows: list[dict[str, object]] = []
        for row in windows_df.itertuples(index=False):
            periodic_slice = periodic_by_phase.get(row.phase_instance_id)
            if periodic_slice is None or periodic_slice.empty:
                continue
            device_power_estimates: list[float] = []
            for _, device_slice in periodic_slice.groupby("device_id", dropna=False):
                overlap_ns = (
                    np.minimum(device_slice["interval_end_ts"].to_numpy(dtype=float), float(row.subphase_end_ts))
                    - np.maximum(device_slice["ts_monotonic_ns"].to_numpy(dtype=float), float(row.subphase_start_ts))
                )
                overlap_mask = overlap_ns > 0
                if not np.any(overlap_mask):
                    continue
                weighted_power_mw = np.average(
                    device_slice.loc[overlap_mask, "gpu_power_mW"].to_numpy(dtype=float),
                    weights=overlap_ns[overlap_mask],
                )
                device_power_estimates.append(weighted_power_mw / 1000.0)
            if device_power_estimates:
                estimated_power_rows.append(
                    {
                        "config": row.config,
                        "policy_norm": row.policy_norm,
                        "phase_bucket": row.phase_bucket,
                        "subphase_name": row.subphase_name,
                        "phase_instance_id": row.phase_instance_id,
                        "estimated_subphase_power_w": float(np.sum(device_power_estimates)),
                    }
                )

        subphase_power = pd.DataFrame(estimated_power_rows)
        if subphase_power.empty:
            subphase_power = pd.DataFrame(columns=["config", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"])
        else:
            subphase_power = (
                subphase_power[subphase_power["subphase_name"].isin(SUBPHASE_ORDER)]
                .groupby(["config", "policy_norm", "phase_bucket", "subphase_name"], dropna=False)["estimated_subphase_power_w"]
                .mean()
                .rename("mean_subphase_power_w")
                .reset_index()
            )

    print("retained phase instances by (config, policy, phase):")
    print(phase_counts.sort_values(["config", "policy_norm", "phase_bucket"]).to_string(index=False))
    print("mean subphase duration by (config, policy, phase, subphase):")
    print(
        subphase_means[["config", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_time_s"]]
        .sort_values(["config", "policy_norm", "phase_bucket", "subphase_name"])
        .to_string(index=False)
    )
    if not subphase_power.empty:
        print("estimated mean subphase power by (config, policy, phase, subphase):")
        print(
            subphase_power[["config", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]]
            .sort_values(["config", "policy_norm", "phase_bucket", "subphase_name"])
            .to_string(index=False)
        )

    plot_df = subphase_means.merge(
        subphase_power,
        on=["config", "policy_norm", "phase_bucket", "subphase_name"],
        how="left",
        validate="one_to_one",
    )
    global_time_max = max(float(plot_df["mean_subphase_time_s"].max()), 1.0)

    fig, axes = plt.subplots(1, len(SUBPHASE_ORDER), figsize=(16.2, 4.8), sharey=False)
    axes_flat = np.atleast_1d(axes).flatten()
    x = np.arange(len(POLICY_ORDER), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONFIG_ORDER))

    for ax, subphase in zip(axes_flat, SUBPHASE_ORDER):
        sub_df = plot_df[plot_df["subphase_name"] == subphase].copy()
        if sub_df.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        for cfg_idx, config in enumerate(CONFIG_ORDER):
            xpos = x + offsets[cfg_idx]
            heights = []
            for policy in POLICY_ORDER:
                row = sub_df[(sub_df["config"] == config) & (sub_df["policy_norm"] == policy)]
                if row.empty:
                    heights.append(0.0)
                else:
                    heights.append(float(row["mean_subphase_time_s"].iloc[0]))
            ax.bar(
                xpos,
                heights,
                width=width,
                color=CONFIG_COLORS[config],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.88,
            )

        ax.set_title(SUBPHASE_DISPLAY[subphase], fontsize=PANEL_TITLE_SIZE, fontweight="bold")
        ax.set_xticks(x, [POLICY_DISPLAY[p] for p in POLICY_ORDER])
        ax.set_ylim(0, global_time_max * 1.18)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

        parent_phase = SUBPHASE_PARENT[subphase]
        ax.text(
            0.98,
            0.96,
            PHASE_DISPLAY[parent_phase],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.8, "pad": 0.2},
        )

    for idx, ax in enumerate(axes_flat):
        if idx == 0 and ax.axison:
            ax.set_ylabel("Mean subphase time (s)", fontsize=AXIS_LABEL_SIZE)

    config_handles = [
        Patch(facecolor=CONFIG_COLORS[config], edgecolor="black", alpha=0.88, label=CONFIG_DISPLAY[config])
        for config in CONFIG_ORDER
    ]
    fig.suptitle("Major Subphase Time and Average Power Decomposition", y=0.992, fontweight="bold", fontsize=FIGURE_TITLE_SIZE)
    fig.legend(handles=config_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955), fontsize=LEGEND_FONT_SIZE, title="Configuration")
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="subphase_time_power_decomp",
        run_ids=selected_run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["phase_fact_view", "phase_timings_long", "hardware_periodic", "runs", "run_summary_view"],
            "filter": "apply_analysis_ok",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
