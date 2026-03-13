"""Faithful major subphase time decomposition for baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/baselines/subphase_time_power_decomp.png")
INCLUDE_VALIDATION = False
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
SUBPHASE_ORDER = (
    "gen",
    "gen_max",
    "old_log_prob",
    "values",
    "update_critic",
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
    "gen_max": "Baseline Generation",
    "old_log_prob": "Old Log Prob",
    "values": "Value Inference",
    "update_critic": "Critic Update",
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
RECONSTRUCTION_SUBPHASE_ORDER = {
    "rollout": ("gen", "gen_max"),
    "rl_policy": ("reward", "old_log_prob", "values", "adv"),
    "training": ("update_critic", "update_actor"),
}


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = [
        "run_id",
        "phase_name",
        "phase_id",
        "phase_instance_id",
        "avg_power_w",
        "policy",
        "model",
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
    df, _ = load_view("phase_fact_view")
    needed = [col for col in required_cols + filter_cols_optional if col in df.columns]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            "phase_fact_view is missing required columns "
            f"{missing_required}. Available columns: {list(df.columns)}"
        )
    return df[needed].copy()


def _load_phase_timings() -> pd.DataFrame:
    df, _ = load_view("phase_timings_long")
    required = ["run_id", "global_step_canonical", "phase_name", "phase_id", "subphase_name", "metric_unit", "value"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "phase_timings_long is missing required columns "
            f"{missing}. Available columns: {list(df.columns)}"
        )
    return df[required].copy()


def _load_hardware_periodic() -> pd.DataFrame:
    df, _ = load_view("hardware_periodic")
    required = ["run_id", "phase_instance_id", "device_id", "device_kind", "ts_monotonic_ns", "gpu_power_mW"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "hardware_periodic is missing required columns "
            f"{missing}. Available columns: {list(df.columns)}"
        )
    return df[required].copy()


def _annotation_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "black" if luminance > 155 else "white"


def _load_run_summary_for_selection() -> pd.DataFrame:
    df_runs, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "model", "logical_run_group"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(
            "run_summary_view is missing required selection columns "
            f"{missing}. Available columns: {list(df_runs.columns)}"
        )
    return df_runs.copy()


def _load_runs_with_slurm_metadata() -> pd.DataFrame:
    df_runs, _ = load_view("runs")
    required = ["run_id", "slurm_job_name"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(
            "runs is missing required slurm metadata columns "
            f"{missing}. Available columns: {list(df_runs.columns)}"
        )
    return df_runs[required].copy()


def _select_runs() -> pd.DataFrame:
    runs_df = _load_run_summary_for_selection()
    runs_meta_df = _load_runs_with_slurm_metadata()
    runs_df = runs_df.merge(runs_meta_df, on="run_id", how="left", validate="one_to_one")

    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    expected_slurm_job_name = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm_job_name
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected_runs = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & checkpoint_mask & slurm_job_mask
    ].copy()
    if selected_runs.empty:
        raise ValueError("No baseline runs selected.")
    return selected_runs


def main() -> None:
    phase_df = _load_phase_fact_for_plot()
    phase_timings = _load_phase_timings()
    selected_runs = _select_runs()

    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    phase_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    if phase_df.empty:
        raise ValueError(f"No phase_fact_view rows found for selected runs: {selected_run_ids}")

    phase_df_before_filter = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    filtering = explain_filtering(phase_df_before_filter, phase_df)
    print(f"filtering={filtering}")

    if not INCLUDE_VALIDATION:
        phase_df = phase_df[phase_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
    phase_df["model_facet"] = phase_df["model"].map(_model_facet)
    phase_df["policy_norm"] = phase_df["policy"].astype(str).str.lower()
    phase_df["avg_power_w"] = pd.to_numeric(phase_df["avg_power_w"], errors="coerce")
    phase_df["phase_start_ts"] = pd.to_numeric(phase_df["phase_start_ts"], errors="coerce")
    phase_df["phase_end_ts"] = pd.to_numeric(phase_df["phase_end_ts"], errors="coerce")
    phase_df = phase_df[phase_df["phase_bucket"].isin(PHASE_ORDER)].dropna(subset=["avg_power_w"]).copy()

    retained_phase_keys = phase_df[
        [
            "run_id",
            "global_step_canonical",
            "phase_name",
            "phase_id",
            "phase_instance_id",
            "phase_bucket",
            "model_facet",
            "policy_norm",
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
        retained_phase_keys.groupby(["model_facet", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_phase_instances")
        .reset_index()
    )
    subphase_sums = (
        display_phase_timings.groupby(["model_facet", "policy_norm", "phase_bucket", "subphase_name"], dropna=False)["value"]
        .sum()
        .rename("subphase_time_s_total")
        .reset_index()
    )
    subphase_means = subphase_sums.merge(
        phase_counts,
        on=["model_facet", "policy_norm", "phase_bucket"],
        how="left",
        validate="many_to_one",
    )
    subphase_means["mean_subphase_time_s"] = (
        subphase_means["subphase_time_s_total"] / subphase_means["n_phase_instances"].clip(lower=1)
    )

    # Reconstruct major subphase windows in trainer execution order, then estimate mean subphase power
    # from periodic GPU samples by time-weighted overlap. This is approximate because subphase boundaries
    # are not logged directly.
    hardware_periodic = _load_hardware_periodic()
    hardware_periodic = hardware_periodic[
        hardware_periodic["run_id"].astype(str).isin(selected_run_ids)
        & (hardware_periodic["device_kind"].astype(str).str.lower() == "gpu")
        & hardware_periodic["phase_instance_id"].isin(retained_phase_keys["phase_instance_id"])
    ].copy()
    hardware_periodic["ts_monotonic_ns"] = pd.to_numeric(hardware_periodic["ts_monotonic_ns"], errors="coerce")
    hardware_periodic["gpu_power_mW"] = pd.to_numeric(hardware_periodic["gpu_power_mW"], errors="coerce")
    hardware_periodic = hardware_periodic.dropna(subset=["ts_monotonic_ns", "gpu_power_mW", "device_id"]).copy()

    subphase_duration_lookup = (
        phase_timings.groupby(["phase_instance_id", "subphase_name"], dropna=False)["value"].sum().to_dict()
    )
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
                    "model_facet": row.model_facet,
                    "policy_norm": row.policy_norm,
                    "subphase_name": subphase_name,
                    "subphase_start_ts": start_ns,
                    "subphase_end_ts": end_ns,
                }
            )
            cursor_ns = end_ns

    windows_df = pd.DataFrame(reconstructed_windows)
    if windows_df.empty:
        subphase_power = pd.DataFrame(
            columns=["model_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
        )
    else:
        hardware_periodic = hardware_periodic.sort_values(["run_id", "device_id", "ts_monotonic_ns"]).copy()
        device_groups = hardware_periodic.groupby(["run_id", "device_id"], dropna=False)
        hardware_periodic["_next_ts"] = device_groups["ts_monotonic_ns"].shift(-1)
        hardware_periodic["sample_weight_ns"] = pd.to_numeric(
            hardware_periodic["_next_ts"] - hardware_periodic["ts_monotonic_ns"],
            errors="coerce",
        )
        fallback_weight = device_groups["sample_weight_ns"].transform(lambda s: s[s > 0].median())
        hardware_periodic["sample_weight_ns"] = (
            hardware_periodic["sample_weight_ns"].fillna(fallback_weight).fillna(1.0).clip(lower=1.0)
        )
        hardware_periodic["interval_end_ts"] = (
            pd.to_numeric(hardware_periodic["ts_monotonic_ns"], errors="coerce") + hardware_periodic["sample_weight_ns"]
        )

        periodic_by_phase = {
            phase_instance_id: grp.copy()
            for phase_instance_id, grp in hardware_periodic.groupby("phase_instance_id", dropna=False)
        }
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
                        "model_facet": row.model_facet,
                        "policy_norm": row.policy_norm,
                        "phase_bucket": row.phase_bucket,
                        "subphase_name": row.subphase_name,
                        "phase_instance_id": row.phase_instance_id,
                        "estimated_subphase_power_w": float(np.sum(device_power_estimates)),
                    }
                )

        subphase_power = pd.DataFrame(estimated_power_rows)
        if subphase_power.empty:
            subphase_power = pd.DataFrame(
                columns=["model_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
            )
        else:
            subphase_power = (
                subphase_power[subphase_power["subphase_name"].isin(SUBPHASE_ORDER)]
                .groupby(["model_facet", "policy_norm", "phase_bucket", "subphase_name"], dropna=False)[
                    "estimated_subphase_power_w"
                ]
                .mean()
                .rename("mean_subphase_power_w")
                .reset_index()
            )

    print("retained phase instances by (model, policy, phase):")
    print(phase_counts.sort_values(["model_facet", "policy_norm", "phase_bucket"]).to_string(index=False))
    print("mean subphase duration by (model, policy, phase, subphase):")
    print(
        subphase_means[
            ["model_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_time_s"]
        ].sort_values(["model_facet", "policy_norm", "phase_bucket", "subphase_name"]).to_string(index=False)
    )
    if not subphase_power.empty:
        print("estimated mean subphase power by (model, policy, phase, subphase):")
        print(
            subphase_power[
                ["model_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
            ].sort_values(["model_facet", "policy_norm", "phase_bucket", "subphase_name"]).to_string(index=False)
        )

    global_time_max = max(
        float(
            subphase_means.groupby(["model_facet", "policy_norm", "phase_bucket"], dropna=False)["mean_subphase_time_s"]
            .sum()
            .max()
        ),
        1.0,
    )
    fig, axes = plt.subplots(len(TARGET_MODEL_FACETS), len(POLICY_ORDER), figsize=(15.5, 8.8))
    x = np.arange(len(PHASE_ORDER), dtype=float)

    for row_idx, facet in enumerate(TARGET_MODEL_FACETS):
        for col_idx, policy in enumerate(POLICY_ORDER):
            ax = axes[row_idx][col_idx]

            combo_sub = subphase_means[
                (subphase_means["model_facet"] == facet) & (subphase_means["policy_norm"] == policy)
            ].copy()
            combo_subpower = subphase_power[
                (subphase_power["model_facet"] == facet) & (subphase_power["policy_norm"] == policy)
            ].copy()

            bottom = np.zeros(len(PHASE_ORDER), dtype=float)
            for subphase in SUBPHASE_ORDER:
                heights = []
                for phase in PHASE_ORDER:
                    parent = SUBPHASE_PARENT[subphase]
                    if parent != phase:
                        heights.append(0.0)
                        continue
                    match = combo_sub[
                        (combo_sub["phase_bucket"] == phase) & (combo_sub["subphase_name"] == subphase)
                    ]["mean_subphase_time_s"]
                    heights.append(float(match.iloc[0]) if not match.empty else 0.0)
                heights = np.array(heights, dtype=float)
                ax.bar(
                    x,
                    heights,
                    bottom=bottom,
                    width=0.62,
                    color=SUBPHASE_COLORS[subphase],
                    edgecolor="black",
                    linewidth=0.7,
                    label=subphase,
                )
                for phase_idx, height in enumerate(heights):
                    if height <= 0.55:
                        continue
                    power_match = combo_subpower[
                        (combo_subpower["phase_bucket"] == PHASE_ORDER[phase_idx])
                        & (combo_subpower["subphase_name"] == subphase)
                    ]["mean_subphase_power_w"]
                    if power_match.empty:
                        continue
                    ax.text(
                        x[phase_idx],
                        bottom[phase_idx] + (height / 2.0),
                        f"{float(power_match.iloc[0]):.0f} W",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=_annotation_color(SUBPHASE_COLORS[subphase]),
                    )
                bottom += heights

            ax.set_title(f"{MODEL_DISPLAY[facet]} | {POLICY_DISPLAY[policy]}", fontsize=10, fontweight="bold")
            ax.set_xticks(x, [PHASE_DISPLAY[phase] for phase in PHASE_ORDER])
            ax.set_ylim(0, global_time_max * 1.14)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)

            if col_idx == 0:
                ax.set_ylabel(f"{facet}\nMean subphase time (s)")

    subphase_handles = [
        Patch(facecolor=SUBPHASE_COLORS[subphase], edgecolor="black", label=SUBPHASE_DISPLAY[subphase])
        for subphase in SUBPHASE_ORDER
    ]

    fig.suptitle(
        "Major Subphase Time and Average Power Decomposition",
        y=0.992,
        fontweight="bold",
    )
    fig.legend(
        handles=subphase_handles,
        title="Subphase",
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
