"""Combined boundary-aligned GPU temperature transition plots for baseline runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/baselines/phase_transition_temperature.png")
BIN_WIDTH_S = 0.5
WINDOW_LEFT_S = -20.0
WINDOW_RIGHT_S = 40.0

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_PANEL_TITLE = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
POLICY_COLOR = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
CAP_BG_COLOR = "#D62728"
PRE_SHADE = "#ECECEC"
TRANSITIONS = (
    {
        "key": "rollout_to_rlpolicy",
        "anchor_phase": "rl_policy",
        "window_phases": ("rollout", "rl_policy"),
        "title": "rollout -> rl_policy",
    },
    {
        "key": "rlpolicy_to_training",
        "anchor_phase": "training",
        "window_phases": ("rl_policy", "training"),
        "title": "rl_policy -> training",
    },
    {
        "key": "training_to_rollout",
        "anchor_phase": "rollout",
        "window_phases": ("training", "rollout"),
        "cross_step": True,
        "title": "training -> rollout",
    },
)


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _select_baseline_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")

    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    ).copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].eq("Llama")
    expected_slurm = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & slurm_job_mask & checkpoint_mask
    ][["run_id", "model_facet", "policy_norm"]].drop_duplicates()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def _build_transition_samples(
    anchor_phase: str,
    window_phases: tuple[str, str],
    selected_runs: pd.DataFrame,
    cross_step: bool = False,
) -> pd.DataFrame:
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    phase_fact, _ = load_view("phase_fact_view")
    phase_fact = phase_fact[phase_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    phase_fact = apply_analysis_ok(phase_fact)
    phase_fact["phase_name"] = phase_fact["phase_name"].astype(str).str.lower()
    phase_fact = phase_fact[phase_fact["phase_name"] == anchor_phase].copy()

    anchors = phase_fact[["run_id", "global_step_canonical", "phase_start_ts"]].copy()
    anchors["phase_start_ts"] = pd.to_numeric(anchors["phase_start_ts"], errors="coerce")
    anchors = anchors.dropna(subset=["phase_start_ts"]).copy()
    anchors["phase_start_ts"] = anchors["phase_start_ts"].astype(np.int64)
    if anchors.empty:
        raise ValueError(f"No analysis-valid {anchor_phase} phase starts found.")

    periodic, _ = load_view("hardware_periodic")
    needed = [
        "run_id",
        "global_step_canonical",
        "phase_name",
        "ts_monotonic_ns",
        "temp_gpu_C",
        "thr_sw_power_cap",
        "record_type",
        "source",
    ]
    missing = [c for c in needed if c not in periodic.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")

    df = periodic[periodic["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    df = df[df["record_type"].astype(str).str.upper() == "PERIODIC"].copy()
    df = df[df["source"].astype(str).str.lower() == "nvml"].copy()
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(window_phases)].copy()
    df["ts_monotonic_ns"] = pd.to_numeric(df["ts_monotonic_ns"], errors="coerce")
    df["temp_gpu_C"] = pd.to_numeric(df["temp_gpu_C"], errors="coerce")
    df["thr_sw_power_cap"] = df["thr_sw_power_cap"].fillna(False).astype(bool)
    df = df.dropna(subset=["ts_monotonic_ns", "temp_gpu_C"]).copy()
    df["ts_monotonic_ns"] = df["ts_monotonic_ns"].astype(np.int64)

    if cross_step:
        df = df.merge(anchors[["run_id", "phase_start_ts"]], on="run_id", how="inner")
        df = df.merge(selected_runs, on="run_id", how="inner")
        df["seconds_from_anchor"] = (df["ts_monotonic_ns"] - df["phase_start_ts"]) / 1e9
        df = df[
            (df["seconds_from_anchor"] >= WINDOW_LEFT_S)
            & (df["seconds_from_anchor"] <= WINDOW_RIGHT_S)
        ].copy()
        df = df[
            ((df["phase_name"] == window_phases[0]) & (df["seconds_from_anchor"] < 0.0))
            | ((df["phase_name"] == window_phases[1]) & (df["seconds_from_anchor"] >= 0.0))
        ].copy()
    else:
        df = df.merge(anchors, on=["run_id", "global_step_canonical"], how="inner")
        df = df.merge(selected_runs, on="run_id", how="inner")
        df["seconds_from_anchor"] = (df["ts_monotonic_ns"] - df["phase_start_ts"]) / 1e9
        df = df[
            (df["seconds_from_anchor"] >= WINDOW_LEFT_S)
            & (df["seconds_from_anchor"] <= WINDOW_RIGHT_S)
        ].copy()

    if df.empty:
        raise ValueError(f"No periodic samples in the requested window for {anchor_phase}.")

    df["bin_center_s"] = (
        np.floor((df["seconds_from_anchor"] - WINDOW_LEFT_S) / BIN_WIDTH_S) * BIN_WIDTH_S
        + WINDOW_LEFT_S
        + (BIN_WIDTH_S / 2.0)
    )
    return df


def _aggregate_bins(df: pd.DataFrame, transition_key: str) -> pd.DataFrame:
    rows = []
    for (policy_norm, bin_center_s), g in df.groupby(["policy_norm", "bin_center_s"], dropna=False):
        temp = pd.to_numeric(g["temp_gpu_C"], errors="coerce").dropna().to_numpy(dtype=float)
        if temp.size == 0:
            continue
        cap = g["thr_sw_power_cap"].astype(bool)
        rows.append(
            {
                "transition_key": transition_key,
                "policy_norm": policy_norm,
                "bin_center_s": float(bin_center_s),
                "temp_median": float(np.median(temp)),
                "temp_q25": float(np.percentile(temp, 25)),
                "temp_q75": float(np.percentile(temp, 75)),
                "cap_frac": float(cap.mean()),
                "n_samples": int(len(g)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No binned rows available for plotting: {transition_key}")
    return out.sort_values(["transition_key", "policy_norm", "bin_center_s"])


def main() -> None:
    selected_runs = _select_baseline_runs()

    sample_frames = []
    binned_frames = []
    for transition in TRANSITIONS:
        samples = _build_transition_samples(
            anchor_phase=transition["anchor_phase"],
            window_phases=transition["window_phases"],
            selected_runs=selected_runs,
            cross_step=bool(transition.get("cross_step", False)),
        )
        samples["transition_key"] = transition["key"]
        sample_frames.append(samples)
        binned_frames.append(_aggregate_bins(samples, transition_key=transition["key"]))

    all_samples = pd.concat(sample_frames, ignore_index=True)
    all_binned = pd.concat(binned_frames, ignore_index=True)

    print("windowed sample counts by transition, policy, and phase:")
    print(
        all_samples.groupby(["transition_key", "policy_norm", "phase_name"], dropna=False)
        .size()
        .rename("n_samples")
        .reset_index()
        .sort_values(["transition_key", "policy_norm", "phase_name"])
        .to_string(index=False)
    )

    y_max = float(all_binned["temp_q75"].max())
    y_min = float(all_binned["temp_q25"].min())
    y_pad = max(1.0, 0.06 * (y_max - y_min))
    band_height = 0.025
    band_gap = 0.008

    n_rows = len(TRANSITIONS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.6, 3.1 * n_rows + 1.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for row_idx, transition in enumerate(TRANSITIONS):
        ax = axes[row_idx]
        sub = all_binned[all_binned["transition_key"] == transition["key"]].copy()

        ax_right = ax.twinx()
        ax_right.set_ylim(0.0, 1.0)
        ax_right.set_yticks([0.0, 0.5, 1.0])
        ax_right.set_ylabel("fraction of samples", color="black")
        ax_right.tick_params(axis="y", colors="black")
        ax_right.spines["right"].set_color("black")
        ax_right.grid(False)

        if not sub.empty:
            for policy_idx, policy_norm in enumerate(TARGET_POLICIES):
                pol = sub[sub["policy_norm"] == policy_norm].copy()
                if pol.empty:
                    continue
                x = pol["bin_center_s"].to_numpy(dtype=float)
                med = pol["temp_median"].to_numpy(dtype=float)
                color = POLICY_COLOR[policy_norm]

                cap_profile = (
                    pol.groupby("bin_center_s", dropna=False)["cap_frac"]
                    .mean()
                    .reset_index()
                    .sort_values("bin_center_s")
                )
                band_bottom = 0.01 + policy_idx * (band_height + band_gap)
                cap_top = (cap_profile["cap_frac"].to_numpy(dtype=float) * band_height) + band_bottom
                ax.fill_between(
                    cap_profile["bin_center_s"].to_numpy(dtype=float),
                    band_bottom,
                    cap_top,
                    color=color,
                    alpha=0.20,
                    transform=ax.get_xaxis_transform(),
                    linewidth=0.0,
                    zorder=1,
                )

                ax.plot(
                    x,
                    med,
                    color=color,
                    linewidth=2.2,
                    zorder=3,
                    label=POLICY_PANEL_TITLE[policy_norm] if row_idx == 0 else None,
                )

        ax.axvline(0.0, color=CAP_BG_COLOR, linewidth=1.2, alpha=0.95, zorder=4)
        ax.set_ylabel(f"{transition['title']}\ntemp_gpu_C")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("(s) from phase boundary start")
    for ax in axes:
        ax.set_xlim(WINDOW_LEFT_S, WINDOW_RIGHT_S)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.suptitle("GPU Temperature Recovery Across Phase Transitions, Llama-3.1-8B-Inst", y=0.99, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=CAP_BG_COLOR, linewidth=1.2))
    labels.append("phase boundary")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.8)

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
