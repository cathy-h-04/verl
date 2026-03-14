"""Faithful all-subphase time decomposition for reward-model runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/reward_model/subphase_time_power_decomp.png")
INCLUDE_VALIDATION = False
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama Reward Function": "llama_new_baseline",
    "Llama Reward Model": "llama_rm_gsm8k",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
EXPERIMENT_COLORS = {
    "Llama Reward Function": "#CC4118",
    "Llama Reward Model": "#295894",
}
TARGET_EXPERIMENT_FACETS = ("Llama Reward Function", "Llama Reward Model")
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
LOGICAL_GROUP_PREFIXES_BY_FACET = {
    "Llama Reward Function": ("stage1_llama8b_",),
    "Llama Reward Model": ("llama8b_",),
}
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
BASE_SUBPHASE_COLORS = {
    "gen": "#4C78A8",
    "generate_sequences": "#72B7B2",
    "gen_max": "#9C755F",
    "reward": "#2E8B57",
    "old_log_prob": "#54A24B",
    "values": "#86BC86",
    "adv": "#B6992D",
    "update_critic": "#ECA82C",
    "update_actor": "#F58518",
    "testing": "#B279A2",
}
RECONSTRUCTION_SUBPHASE_ORDER = {
    "rollout": ("gen", "gen_max"),
    "rl_policy": ("reward", "old_log_prob", "values", "adv"),
    "training": ("update_critic", "update_actor"),
}
TARGET_PHASE = "rl_policy"
TARGET_SUBPHASE = "reward"


def _experiment_facet(slurm_job_name: str, logical_run_group: str) -> str:
    slurm_text = str(slurm_job_name).strip().lower()
    logical_text = str(logical_run_group).strip().lower()
    for facet in TARGET_EXPERIMENT_FACETS:
        expected_slurm = TARGET_SLURM_JOB_NAME_BY_FACET[facet]
        logical_prefixes = LOGICAL_GROUP_PREFIXES_BY_FACET[facet]
        if slurm_text == expected_slurm and logical_text.startswith(logical_prefixes):
            return facet
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


def _subphase_display(name: str) -> str:
    return str(name).replace("_", " ")


def _build_subphase_color_map(subphase_order: tuple[str, ...]) -> dict[str, str]:
    color_map = {name: color for name, color in BASE_SUBPHASE_COLORS.items() if name in subphase_order}
    fallback_names = [name for name in subphase_order if name not in color_map]
    if fallback_names:
        cmap = plt.get_cmap("tab20", max(len(fallback_names), 1))
        for idx, name in enumerate(fallback_names):
            color_map[name] = mcolors.to_hex(cmap(idx))
    return color_map


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
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()
    runs_df["experiment_facet"] = [
        _experiment_facet(slurm_job_name=slurm_job_name, logical_run_group=logical_run_group)
        for slurm_job_name, logical_run_group in zip(runs_df["slurm_job_name"], runs_df["logical_run_group"])
    ]

    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["experiment_facet"].isin(
        TARGET_EXPERIMENT_FACETS
    )
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected_runs = runs_df[non_rollout_knob_mask & target_pair_mask & checkpoint_mask].copy()
    if selected_runs.empty:
        raise ValueError("No reward-model runs selected.")
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

    phase_df = phase_df.merge(
        selected_runs[["run_id", "experiment_facet"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
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
            "experiment_facet",
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

    display_phase_timings = phase_timings[phase_timings["phase_bucket"] == TARGET_PHASE].copy()
    subphase_totals = (
        display_phase_timings.groupby(["phase_bucket", "subphase_name"], dropna=False)["value"]
        .sum()
        .rename("total_subphase_time_s")
        .reset_index()
    )
    phase_rank_map = {name: idx for idx, name in enumerate(PHASE_ORDER)}
    subphase_totals["phase_rank"] = subphase_totals["phase_bucket"].map(phase_rank_map)
    subphase_totals = subphase_totals.sort_values(
        ["phase_rank", "total_subphase_time_s", "subphase_name"],
        ascending=[True, False, True],
    )
    subphase_order = tuple(subphase_totals["subphase_name"].tolist())
    subphase_parent = {row.subphase_name: row.phase_bucket for row in subphase_totals.itertuples(index=False)}
    subphase_colors = _build_subphase_color_map(subphase_order)

    phase_counts = (
        retained_phase_keys[retained_phase_keys["phase_bucket"] == TARGET_PHASE]
        .groupby(["experiment_facet", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_phase_instances")
        .reset_index()
    )
    subphase_sums = (
        display_phase_timings.groupby(["experiment_facet", "policy_norm", "phase_bucket", "subphase_name"], dropna=False)[
            "value"
        ]
        .sum()
        .rename("subphase_time_s_total")
        .reset_index()
    )
    subphase_means = subphase_sums.merge(
        phase_counts,
        on=["experiment_facet", "policy_norm", "phase_bucket"],
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
                    "experiment_facet": row.experiment_facet,
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
            columns=["experiment_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
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
                        "experiment_facet": row.experiment_facet,
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
                columns=["experiment_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
            )
        else:
            subphase_power = (
                subphase_power.groupby(["experiment_facet", "policy_norm", "phase_bucket", "subphase_name"], dropna=False)[
                    "estimated_subphase_power_w"
                ]
                .mean()
                .rename("mean_subphase_power_w")
                .reset_index()
            )

    print("retained preparation phase instances by (experiment, policy):")
    print(phase_counts.sort_values(["experiment_facet", "policy_norm"]).to_string(index=False))
    print(f"mean preparation subphase duration for '{TARGET_SUBPHASE}' by (experiment, policy):")
    print(
        subphase_means[subphase_means["subphase_name"] == TARGET_SUBPHASE][
            ["experiment_facet", "policy_norm", "subphase_name", "mean_subphase_time_s"]
        ].sort_values(["experiment_facet", "policy_norm"]).to_string(index=False)
    )
    if not subphase_power.empty:
        print(f"estimated mean preparation subphase power for '{TARGET_SUBPHASE}' by (experiment, policy):")
        print(
            subphase_power[subphase_power["subphase_name"] == TARGET_SUBPHASE][
                ["experiment_facet", "policy_norm", "subphase_name", "mean_subphase_power_w"]
            ].sort_values(["experiment_facet", "policy_norm"]).to_string(index=False)
        )

    prep_subphase_order = tuple(
        name for name in subphase_order if subphase_parent.get(name) == TARGET_PHASE and name == TARGET_SUBPHASE
    )
    prep_power = subphase_power[subphase_power["phase_bucket"] == TARGET_PHASE].copy()
    global_power_max = max(float(prep_power["mean_subphase_power_w"].max()) if not prep_power.empty else 0.0, 1.0)

    policy_x = np.arange(len(POLICY_ORDER), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.8))
    bar_width = 0.34
    offsets = np.array([-bar_width / 2, bar_width / 2])

    for offset, facet in zip(offsets, TARGET_EXPERIMENT_FACETS):
        facet_power = prep_power[prep_power["experiment_facet"] == facet].copy()
        heights = []
        for policy in POLICY_ORDER:
            match = facet_power[
                (facet_power["policy_norm"] == policy) & (facet_power["subphase_name"] == TARGET_SUBPHASE)
            ]["mean_subphase_power_w"]
            heights.append(float(match.iloc[0]) if not match.empty else 0.0)
        heights = np.array(heights, dtype=float)
        bars = ax.bar(
            policy_x + offset,
            heights,
            width=bar_width,
            color=EXPERIMENT_COLORS[facet],
            edgecolor="black",
            linewidth=0.9,
            label=EXPERIMENT_DISPLAY[facet],
        )
        for bar, val in zip(bars, heights):
            if val <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + global_power_max * 0.018,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
            )

    ax.set_xticks(policy_x, [POLICY_DISPLAY[policy] for policy in POLICY_ORDER])
    ax.set_ylim(0, global_power_max * 1.16)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_ylabel("Mean Reward Subphase Power (W)")
    ax.set_xlabel("Policy")

    fig.suptitle(
        "Reward Subphase Power Decomposition by Reward Mechanism",
        y=0.992,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor=EXPERIMENT_COLORS[facet], edgecolor="black", label=EXPERIMENT_DISPLAY[facet])
            for facet in TARGET_EXPERIMENT_FACETS
        ],
        title="Experiment",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
