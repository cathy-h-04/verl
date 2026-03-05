"""Per-phase utilization-power coupling signature space.

Computes coupling per phase instance from periodic GPU telemetry, then aggregates by
(model, policy, phase_bucket) for baseline configs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier1/util_power_coupling_signature_per_phase.png")
MIN_SAMPLES = 3
INCLUDE_ROLLOUT_OVERLAY = False

MODEL_ORDER = ("Llama", "Qwen")
POLICY_ORDER = ("ppo", "remax", "grpo")
PHASE_COLORS = {
    "rollout": "#1f77b4",
    "training": "#ff7f0e",
    "rl_policy": "#2ca02c",
    "validation": "#d62728",
    "other": "#7f7f7f",
}
POLICY_MARKERS = {"ppo": "o", "remax": "s", "grpo": "^"}


def _norm_policy(policy: str) -> str:
    return str(policy).strip().lower().replace("remx", "remax")


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _phase_bucket(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _fit_slope_corr(util: np.ndarray, power_w: np.ndarray) -> tuple[float, float]:
    if len(util) < 2:
        return np.nan, np.nan
    if np.allclose(util, util[0]) or np.allclose(power_w, power_w[0]):
        return np.nan, np.nan
    b = float(np.polyfit(util, power_w, 1)[0])
    corr = float(np.corrcoef(util, power_w)[0, 1])
    return b, corr


def main() -> None:
    # Strict run/config selection from run_summary_view.
    rs, _ = load_view("run_summary_view")
    required_rs = [
        "run_id",
        "model",
        "policy",
        "rollout_max_batched_tokens",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
    ]
    missing_rs = [c for c in required_rs if c not in rs.columns]
    if missing_rs:
        raise ValueError(f"run_summary_view missing required columns: {missing_rs}")

    rs = rs.copy()
    rs["run_id"] = rs["run_id"].astype(str)
    rs["model_facet"] = rs["model"].map(_model_facet)
    rs["policy_norm"] = rs["policy"].map(_norm_policy)
    rs["rollout_max_batched_tokens"] = pd.to_numeric(rs["rollout_max_batched_tokens"], errors="coerce")
    rs["is_checkpoint_continuation"] = rs["is_checkpoint_continuation"].fillna(False).astype(bool)
    rs["join_coverage_rate"] = pd.to_numeric(rs["join_coverage_rate"], errors="coerce")
    rs["phase_boundary_integrity_rate"] = pd.to_numeric(rs["phase_boundary_integrity_rate"], errors="coerce")

    mask = (
        rs["model_facet"].isin(MODEL_ORDER)
        & rs["policy_norm"].isin(POLICY_ORDER)
        & (~rs["is_checkpoint_continuation"])
        & (rs["join_coverage_rate"] == 1.0)
        & (rs["phase_boundary_integrity_rate"] == 1.0)
    )
    if INCLUDE_ROLLOUT_OVERLAY:
        mask &= rs["rollout_max_batched_tokens"].isin([8192.0, 16384.0])
    else:
        mask &= rs["rollout_max_batched_tokens"] == 8192.0

    selected_runs = rs.loc[mask, ["run_id", "model_facet", "policy_norm", "rollout_max_batched_tokens"]].copy()
    if selected_runs.empty:
        raise ValueError("No runs selected under baseline/integrity criteria.")

    if not INCLUDE_ROLLOUT_OVERLAY:
        expected = {(m, p) for m in MODEL_ORDER for p in POLICY_ORDER}
        got = set(zip(selected_runs["model_facet"], selected_runs["policy_norm"]))
        if expected - got:
            raise ValueError(f"Missing baseline model/policy combos: {sorted(expected-got)}")

    # Load phase windows and periodic telemetry.
    phase_instances = pd.read_parquet("DATASETS/phase_instances.parquet")
    periodic = pd.read_parquet("DATASETS/hardware_periodic.parquet")

    required_pi = [
        "run_id",
        "phase_instance_id",
        "phase_name",
        "phase_start_ts_monotonic_ns",
        "phase_end_ts_monotonic_ns",
    ]
    required_hp = [
        "run_id",
        "phase_instance_id",
        "ts_monotonic_ns",
        "gpu_util_pct",
        "gpu_power_mW",
        "gpu_index",
    ]
    miss_pi = [c for c in required_pi if c not in phase_instances.columns]
    miss_hp = [c for c in required_hp if c not in periodic.columns]
    if miss_pi or miss_hp:
        raise RuntimeError(
            "Cannot compute true per-phase coupling: missing periodic or phase-window fields. "
            f"phase_instances missing={miss_pi}, hardware_periodic missing={miss_hp}"
        )

    selected_run_ids = set(selected_runs["run_id"].tolist())
    phase_instances = phase_instances[phase_instances["run_id"].astype(str).isin(selected_run_ids)].copy()
    periodic = periodic[periodic["run_id"].astype(str).isin(selected_run_ids)].copy()

    phase_instances["run_id"] = phase_instances["run_id"].astype(str)
    periodic["run_id"] = periodic["run_id"].astype(str)
    phase_instances["phase_instance_id"] = phase_instances["phase_instance_id"].astype(str)
    periodic["phase_instance_id"] = periodic["phase_instance_id"].astype(str)
    phase_instances["phase_start_ts_monotonic_ns"] = pd.to_numeric(phase_instances["phase_start_ts_monotonic_ns"], errors="coerce")
    phase_instances["phase_end_ts_monotonic_ns"] = pd.to_numeric(phase_instances["phase_end_ts_monotonic_ns"], errors="coerce")

    periodic["ts_monotonic_ns"] = pd.to_numeric(periodic["ts_monotonic_ns"], errors="coerce")
    periodic["gpu_util_pct"] = pd.to_numeric(periodic["gpu_util_pct"], errors="coerce")
    periodic["gpu_power_mW"] = pd.to_numeric(periodic["gpu_power_mW"], errors="coerce")

    # Join periodic samples to phase metadata (phase_instance_id), then enforce timestamps in-window.
    merged = periodic.merge(
        phase_instances[
            ["run_id", "phase_instance_id", "phase_name", "phase_start_ts_monotonic_ns", "phase_end_ts_monotonic_ns"]
        ].rename(columns={"phase_name": "phase_name_window"}),
        on=["run_id", "phase_instance_id"],
        how="inner",
    )
    in_window = (
        (merged["ts_monotonic_ns"] >= merged["phase_start_ts_monotonic_ns"])
        & (merged["ts_monotonic_ns"] <= merged["phase_end_ts_monotonic_ns"])
    )
    merged = merged[in_window].copy()
    merged = merged.dropna(subset=["gpu_util_pct", "gpu_power_mW", "ts_monotonic_ns"]).copy()

    if merged.empty:
        raise RuntimeError("No periodic samples overlap phase windows for selected runs; cannot compute per-phase coupling.")

    # Aggregate across GPUs per timestamp: mean util, sum power.
    t_agg = (
        merged.groupby(["run_id", "phase_instance_id", "phase_name_window", "ts_monotonic_ns"], dropna=False)
        .agg(
            util_mean=("gpu_util_pct", "mean"),
            power_w_sum=("gpu_power_mW", lambda s: float(np.nansum(s) / 1000.0)),
        )
        .reset_index()
    )

    # Coupling per phase instance.
    records: list[dict[str, object]] = []
    for (run_id, pid, phase_name), grp in t_agg.groupby(
        ["run_id", "phase_instance_id", "phase_name_window"], dropna=False
    ):
        util = grp["util_mean"].to_numpy(dtype=float)
        power = grp["power_w_sum"].to_numpy(dtype=float)
        slope, corr = _fit_slope_corr(util, power)
        records.append(
            {
                "run_id": str(run_id),
                "phase_instance_id": str(pid),
                "phase_name": str(phase_name),
                "phase_bucket": _phase_bucket(str(phase_name)),
                "util_power_slope_b_phase": slope,
                "util_power_corr_phase": corr,
                "n_samples_for_coupling_phase": int(len(grp)),
            }
        )

    per_phase = pd.DataFrame(records)
    per_phase = per_phase.merge(selected_runs, on="run_id", how="left")

    # Reliability gating and cleanup.
    reasons: dict[str, int] = {}
    n0 = len(per_phase)
    mask_samples = per_phase["n_samples_for_coupling_phase"] >= MIN_SAMPLES
    reasons["excluded_low_samples"] = int((~mask_samples).sum())
    per_phase = per_phase[mask_samples].copy()

    mask_slope = per_phase["util_power_slope_b_phase"].notna()
    reasons["excluded_missing_slope"] = int((~mask_slope).sum())
    per_phase = per_phase[mask_slope].copy()

    mask_corr = per_phase["util_power_corr_phase"].notna()
    reasons["excluded_missing_corr"] = int((~mask_corr).sum())
    per_phase = per_phase[mask_corr].copy()

    if per_phase.empty:
        raise ValueError(
            f"No per-phase points left after reliability gating (MIN_SAMPLES={MIN_SAMPLES}). "
            f"drops={reasons}"
        )

    bad_corr = per_phase[(per_phase["util_power_corr_phase"] < -1.0) | (per_phase["util_power_corr_phase"] > 1.0)]
    if not bad_corr.empty:
        raise ValueError(
            "util_power_corr_phase out of [-1,1] for rows:\n"
            + bad_corr[
                ["run_id", "phase_instance_id", "phase_bucket", "util_power_corr_phase"]
            ].to_string(index=False)
        )

    neg_slope = per_phase[per_phase["util_power_slope_b_phase"] < 0]
    if not neg_slope.empty:
        print("warning: negative per-phase slopes detected (possible throttling/artifacts):")
        print(
            neg_slope[
                ["run_id", "model_facet", "policy_norm", "phase_bucket", "util_power_slope_b_phase", "util_power_corr_phase"]
            ].to_string(index=False)
        )

    print(f"per_phase_points_before={n0} per_phase_points_after={len(per_phase)}")
    print(f"drop_reasons={reasons}")

    # Aggregate to plot-level points.
    group_cols = ["model_facet", "policy_norm", "phase_bucket"]
    agg = (
        per_phase.groupby(group_cols, dropna=False)
        .agg(
            util_power_slope_b_phase=("util_power_slope_b_phase", "median"),
            util_power_corr_phase=("util_power_corr_phase", "median"),
            n_samples_for_coupling_phase=("n_samples_for_coupling_phase", "sum"),
            n_phase_instances=("phase_instance_id", "nunique"),
        )
        .reset_index()
    )
    agg = agg[agg["phase_bucket"] != "validation"].copy()

    print("plot_table:")
    print(
        agg[
            [
                "policy_norm",
                "model_facet",
                "phase_bucket",
                "util_power_slope_b_phase",
                "util_power_corr_phase",
                "n_samples_for_coupling_phase",
                "n_phase_instances",
            ]
        ]
        .sort_values(["model_facet", "policy_norm", "phase_bucket"])
        .to_string(index=False)
    )

    # Plot: simplify by faceting model x policy; color encodes phase only.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    for r, model in enumerate(MODEL_ORDER):
        for c, policy in enumerate(POLICY_ORDER):
            ax = axes[r, c]
            sub = agg[(agg["model_facet"] == model) & (agg["policy_norm"] == policy)]
            for _, row in sub.iterrows():
                phase_bucket = str(row["phase_bucket"])
                x = float(row["util_power_slope_b_phase"])
                y = float(row["util_power_corr_phase"])
                ax.scatter(
                    x,
                    y,
                    s=95,
                    marker="o",
                    color=PHASE_COLORS.get(phase_bucket, PHASE_COLORS["other"]),
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.92,
                    zorder=3,
                )

            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.2, zorder=1)
            ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.2, zorder=1)
            ax.set_title(f"{model} | {policy.upper()}")
            ax.grid(alpha=0.2, zorder=0)
            if r == len(MODEL_ORDER) - 1:
                ax.set_xlabel("util→power slope (b)")
            if c == 0:
                ax.set_ylabel("corr(util, power)")

            if sub.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=9)

    # Legend
    phase_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=PHASE_COLORS[p], markeredgecolor="black", markersize=8, label=p)
        for p in ["rollout", "training", "rl_policy", "other"]
        if p in set(agg["phase_bucket"].tolist())
    ]
    fig.legend(
        phase_handles,
        [h.get_label() for h in phase_handles],
        title="phase_bucket (color)",
        loc="upper center",
        ncol=max(1, len(phase_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
    )

    fig.suptitle("Utilization–Power Coupling Signature per Phase", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
