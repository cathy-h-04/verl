"""Estimated CPU package power by major subphase."""

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


OUTPATH = Path("plots/out/baselines/non_results/subphase_cpu_power_only.png")
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
SUBPHASE_OFFSETS = {
    "gen": -0.16,
    "gen_max": 0.16,
    "old_log_prob": -0.16,
    "values": 0.16,
    "update_critic": -0.16,
    "update_actor": 0.16,
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


def _annotation_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "black" if luminance > 155 else "white"


def _load_phase_fact_for_plot() -> pd.DataFrame:
    required_cols = [
        "run_id",
        "phase_name",
        "phase_id",
        "phase_instance_id",
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
    required = [
        "run_id",
        "phase_instance_id",
        "device_id",
        "device_kind",
        "ts_monotonic_ns",
        "cpu_energy_uJ",
        "max_energy_range_uJ",
        "rapl_domain",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")
    return df[required].copy()


def _load_run_summary_for_selection() -> pd.DataFrame:
    df_runs, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "model", "logical_run_group"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    return df_runs.copy()


def _load_runs_with_slurm_metadata() -> pd.DataFrame:
    df_runs, _ = load_view("runs")
    required = ["run_id", "slurm_job_name"]
    missing = [col for col in required if col not in df_runs.columns]
    if missing:
        raise ValueError(f"runs missing required columns: {missing}")
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


def _estimate_subphase_power() -> pd.DataFrame:
    phase_df = _load_phase_fact_for_plot()
    phase_timings = _load_phase_timings()
    selected_runs = _select_runs()

    selected_run_ids = selected_runs["run_id"].astype(str).tolist()
    phase_df = phase_df[phase_df["run_id"].astype(str).isin(selected_run_ids)].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"filtering={explain_filtering(before, phase_df)}")

    if not INCLUDE_VALIDATION:
        phase_df = phase_df[phase_df["phase_name"].astype(str).str.lower() != "validation"].copy()

    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
    phase_df["model_facet"] = phase_df["model"].map(_model_facet)
    phase_df["policy_norm"] = phase_df["policy"].astype(str).str.lower()
    phase_df["phase_start_ts"] = pd.to_numeric(phase_df["phase_start_ts"], errors="coerce")
    phase_df["phase_end_ts"] = pd.to_numeric(phase_df["phase_end_ts"], errors="coerce")
    phase_df = phase_df[phase_df["phase_bucket"].isin(PHASE_ORDER)].copy()

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

    hardware_periodic = _load_hardware_periodic()
    hardware_periodic = hardware_periodic[
        hardware_periodic["run_id"].astype(str).isin(selected_run_ids)
        & (hardware_periodic["device_kind"].astype(str).str.lower() == "rapl")
        & hardware_periodic["phase_instance_id"].isin(retained_phase_keys["phase_instance_id"])
    ].copy()
    hardware_periodic["rapl_domain_lc"] = hardware_periodic["rapl_domain"].astype(str).str.lower()
    hardware_periodic = hardware_periodic[hardware_periodic["rapl_domain_lc"].str.startswith("package")].copy()
    hardware_periodic["ts_monotonic_ns"] = pd.to_numeric(hardware_periodic["ts_monotonic_ns"], errors="coerce")
    hardware_periodic["cpu_energy_uJ"] = pd.to_numeric(hardware_periodic["cpu_energy_uJ"], errors="coerce")
    hardware_periodic["max_energy_range_uJ"] = pd.to_numeric(hardware_periodic["max_energy_range_uJ"], errors="coerce")
    hardware_periodic = hardware_periodic.dropna(
        subset=["ts_monotonic_ns", "cpu_energy_uJ", "device_id", "max_energy_range_uJ"]
    ).copy()

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
            if subphase_name in SUBPHASE_ORDER:
                reconstructed_windows.append(
                    {
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
    if windows_df.empty or hardware_periodic.empty:
        return pd.DataFrame(
            columns=["model_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
        )

    hardware_periodic = hardware_periodic.sort_values(["run_id", "device_id", "ts_monotonic_ns"]).copy()
    device_groups = hardware_periodic.groupby(["run_id", "device_id"], dropna=False)
    hardware_periodic["_next_ts"] = device_groups["ts_monotonic_ns"].shift(-1)
    hardware_periodic["_next_energy_uj"] = device_groups["cpu_energy_uJ"].shift(-1)
    hardware_periodic["_next_max_range_uj"] = device_groups["max_energy_range_uJ"].shift(-1)
    hardware_periodic["sample_weight_ns"] = pd.to_numeric(
        hardware_periodic["_next_ts"] - hardware_periodic["ts_monotonic_ns"],
        errors="coerce",
    )
    energy_delta_uj = pd.to_numeric(hardware_periodic["_next_energy_uj"] - hardware_periodic["cpu_energy_uJ"], errors="coerce")
    wrap_mask = energy_delta_uj < 0
    energy_delta_uj = energy_delta_uj.where(~wrap_mask, energy_delta_uj + hardware_periodic["_next_max_range_uj"])
    hardware_periodic["cpu_power_w"] = (energy_delta_uj / 1e6) / (hardware_periodic["sample_weight_ns"] / 1e9)

    fallback_weight = device_groups["sample_weight_ns"].transform(lambda s: s[s > 0].median())
    hardware_periodic["sample_weight_ns"] = (
        hardware_periodic["sample_weight_ns"].fillna(fallback_weight).fillna(1.0).clip(lower=1.0)
    )
    hardware_periodic["interval_end_ts"] = hardware_periodic["ts_monotonic_ns"] + hardware_periodic["sample_weight_ns"]
    hardware_periodic = hardware_periodic.replace([np.inf, -np.inf], np.nan)
    hardware_periodic = hardware_periodic.dropna(subset=["cpu_power_w"]).copy()
    hardware_periodic = hardware_periodic[hardware_periodic["cpu_power_w"] >= 0].copy()

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
            weighted_power_w = np.average(
                device_slice.loc[overlap_mask, "cpu_power_w"].to_numpy(dtype=float),
                weights=overlap_ns[overlap_mask],
            )
            device_power_estimates.append(float(weighted_power_w))
        if device_power_estimates:
            estimated_power_rows.append(
                {
                    "model_facet": row.model_facet,
                    "policy_norm": row.policy_norm,
                    "phase_bucket": row.phase_bucket,
                    "subphase_name": row.subphase_name,
                    "estimated_subphase_power_w": float(np.sum(device_power_estimates)),
                }
            )

    power_df = pd.DataFrame(estimated_power_rows)
    if power_df.empty:
        return pd.DataFrame(
            columns=["model_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
        )
    return (
        power_df.groupby(["model_facet", "policy_norm", "phase_bucket", "subphase_name"], dropna=False)[
            "estimated_subphase_power_w"
        ]
        .mean()
        .rename("mean_subphase_power_w")
        .reset_index()
    )


def main() -> None:
    subphase_power = _estimate_subphase_power()
    print("estimated mean CPU package power by (model, policy, phase, subphase):")
    if subphase_power.empty:
        print("none")
        raise ValueError("No CPU package power estimates available.")
    print(
        subphase_power[
            ["model_facet", "policy_norm", "phase_bucket", "subphase_name", "mean_subphase_power_w"]
        ].sort_values(["model_facet", "policy_norm", "phase_bucket", "subphase_name"]).to_string(index=False)
    )

    global_power_max = max(float(subphase_power["mean_subphase_power_w"].max()), 1.0)
    fig, axes = plt.subplots(len(TARGET_MODEL_FACETS), len(POLICY_ORDER), figsize=(15.5, 8.2))
    x = np.arange(len(PHASE_ORDER), dtype=float)
    bar_width = 0.28

    for row_idx, facet in enumerate(TARGET_MODEL_FACETS):
        for col_idx, policy in enumerate(POLICY_ORDER):
            ax = axes[row_idx][col_idx]
            combo = subphase_power[
                (subphase_power["model_facet"] == facet) & (subphase_power["policy_norm"] == policy)
            ].copy()

            for subphase in SUBPHASE_ORDER:
                phase_bucket = SUBPHASE_PARENT[subphase]
                values = []
                positions = []
                for phase_idx, phase in enumerate(PHASE_ORDER):
                    if phase != phase_bucket:
                        continue
                    row = combo[(combo["phase_bucket"] == phase) & (combo["subphase_name"] == subphase)]
                    value = float(row["mean_subphase_power_w"].iloc[0]) if not row.empty else 0.0
                    values.append(value)
                    positions.append(x[phase_idx] + SUBPHASE_OFFSETS[subphase])
                if not values:
                    continue
                ax.bar(
                    positions,
                    values,
                    width=bar_width,
                    color=SUBPHASE_COLORS[subphase],
                    edgecolor="black",
                    linewidth=0.7,
                    label=SUBPHASE_DISPLAY[subphase],
                    zorder=3,
                )
                for xpos, value in zip(positions, values):
                    if value <= 0:
                        continue
                    ax.text(
                        xpos,
                        value * 0.5,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=_annotation_color(SUBPHASE_COLORS[subphase]),
                    )

            ax.set_title(f"{MODEL_DISPLAY[facet]} | {POLICY_DISPLAY[policy]}", fontsize=10, fontweight="bold")
            ax.set_xticks(x, [PHASE_DISPLAY[phase] for phase in PHASE_ORDER])
            ax.set_ylim(0, global_power_max * 1.12)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_ylabel(f"{facet}\nEstimated mean CPU package power (W)")

    subphase_handles = [
        Patch(facecolor=SUBPHASE_COLORS[subphase], edgecolor="black", label=SUBPHASE_DISPLAY[subphase])
        for subphase in SUBPHASE_ORDER
    ]
    fig.suptitle("Estimated CPU Package Power by Major Subphase", y=0.992, fontweight="bold")
    fig.text(
        0.5,
        0.962,
        "Bars and in-bar labels show estimated mean CPU package power from reconstructed windows and periodic RAPL samples",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.legend(
        handles=subphase_handles,
        title="Subphase",
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.922),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.865))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
