"""Heatmap of software power-cap throttling rate by RLHF phase.

Panels:
- Baseline runs (base_8192)
- Rollout knob variants (non-8192)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/throttle_sw_power_cap_rate_by_phase_heatmap.png")
MIN_SAMPLES = 3
INCLUDE_VALIDATION = False

MODEL_ORDER = ("Llama", "Qwen")
POLICY_ORDER = ("ppo", "remax", "grpo")
PHASE_ORDER = ["rollout", "training", "rl_policy", "other"]


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


def _variant_label(tokens: float | int | None) -> str:
    if tokens is None or pd.isna(tokens):
        return "unknown_variant"
    v = int(tokens)
    return "base_8192" if v == 8192 else f"knob_{v}"


def _build_matrix(df: pd.DataFrame, row_keys: list[tuple[str, str, str]], phase_cols: list[str]) -> pd.DataFrame:
    rows = []
    labels = []
    for model, policy, variant in row_keys:
        labels.append(f"{model} | {policy.upper()}" if variant == "base_8192" else f"{model} | {policy.upper()} | {variant}")
        sub = df[(df["model_facet"] == model) & (df["policy_norm"] == policy) & (df["variant_label"] == variant)]
        row = {phase: np.nan for phase in phase_cols}
        for phase in phase_cols:
            r = sub[sub["phase_bucket"] == phase]
            if not r.empty:
                row[phase] = float(r["throttle_mean"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows, index=labels, columns=phase_cols)


def _plot_heatmap(ax: plt.Axes, mat: pd.DataFrame, title: str, vmax: float) -> matplotlib.image.AxesImage:
    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                continue
            txt_color = "black" if val < (0.55 * vmax if vmax > 0 else 0.0) else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
    return im


def main() -> None:
    # Run selection metadata and integrity gates.
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
    miss_rs = [c for c in required_rs if c not in rs.columns]
    if miss_rs:
        raise ValueError(f"run_summary_view missing required columns: {miss_rs}")

    rs = rs.copy()
    rs["run_id"] = rs["run_id"].astype(str)
    rs["model_facet"] = rs["model"].map(_model_facet)
    rs["policy_norm"] = rs["policy"].map(_norm_policy)
    rs["rollout_max_batched_tokens"] = pd.to_numeric(rs["rollout_max_batched_tokens"], errors="coerce")
    rs["is_checkpoint_continuation"] = rs["is_checkpoint_continuation"].fillna(False).astype(bool)
    rs["join_coverage_rate"] = pd.to_numeric(rs["join_coverage_rate"], errors="coerce")
    rs["phase_boundary_integrity_rate"] = pd.to_numeric(rs["phase_boundary_integrity_rate"], errors="coerce")

    run_mask = (
        rs["model_facet"].isin(MODEL_ORDER)
        & rs["policy_norm"].isin(POLICY_ORDER)
        & (~rs["is_checkpoint_continuation"])
        & (rs["join_coverage_rate"] == 1.0)
        & (rs["phase_boundary_integrity_rate"] == 1.0)
    )
    selected_runs = rs.loc[run_mask, ["run_id", "model_facet", "policy_norm", "rollout_max_batched_tokens"]].copy()
    selected_runs["variant_label"] = selected_runs["rollout_max_batched_tokens"].map(_variant_label)
    if selected_runs.empty:
        raise ValueError("No runs selected under required baseline/knob integrity filters.")

    baseline_pairs = set(
        selected_runs[selected_runs["variant_label"] == "base_8192"][["model_facet", "policy_norm"]].itertuples(index=False, name=None)
    )
    expected_pairs = {(m, p) for m in MODEL_ORDER for p in POLICY_ORDER}
    if expected_pairs - baseline_pairs:
        available = (
            selected_runs[["model_facet", "policy_norm", "variant_label", "rollout_max_batched_tokens"]]
            .sort_values(["model_facet", "policy_norm", "variant_label"])
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Missing baseline (model,policy) combos: {sorted(expected_pairs-baseline_pairs)}. "
            f"Available selected runs: {available}"
        )

    # Phase data.
    pf, _ = load_view("phase_fact_view")
    required_pf = [
        "run_id",
        "phase_instance_id",
        "phase_name",
        "throttle_sw_power_cap_rate",
        "phase_time_s",
    ]
    miss_pf = [c for c in required_pf if c not in pf.columns]
    if miss_pf:
        raise ValueError(f"phase_fact_view missing required columns: {miss_pf}")

    pf = pf[pf["run_id"].astype(str).isin(selected_runs["run_id"])].copy()
    before = pf.copy()
    pf = apply_analysis_ok(pf)
    filtering = explain_filtering(before, pf)
    print(f"filtering={filtering}")

    pf["run_id"] = pf["run_id"].astype(str)
    pf["phase_instance_id"] = pf["phase_instance_id"].astype(str)
    pf["phase_bucket"] = pf["phase_name"].map(_phase_bucket)
    if not INCLUDE_VALIDATION:
        pf = pf[pf["phase_bucket"] != "validation"].copy()

    pf["throttle_sw_power_cap_rate"] = pd.to_numeric(pf["throttle_sw_power_cap_rate"], errors="coerce")
    pf["phase_time_s"] = pd.to_numeric(pf["phase_time_s"], errors="coerce")

    # Reliability counts from periodic telemetry.
    hp = pd.read_parquet("DATASETS/hardware_periodic.parquet", columns=["run_id", "phase_instance_id"])
    hp["run_id"] = hp["run_id"].astype(str)
    hp["phase_instance_id"] = hp["phase_instance_id"].astype(str)
    hp = hp[hp["run_id"].isin(selected_runs["run_id"])].copy()
    counts = (
        hp.groupby(["run_id", "phase_instance_id"], dropna=False)
        .size()
        .rename("n_periodic_samples_in_phase")
        .reset_index()
    )
    pf = pf.merge(counts, on=["run_id", "phase_instance_id"], how="left")

    drop_reasons: dict[str, int] = {}
    n0 = len(pf)
    miss_samples = pf["n_periodic_samples_in_phase"].isna()
    drop_reasons["missing_periodic_sample_count"] = int(miss_samples.sum())
    pf = pf[~miss_samples].copy()

    low_samples = pf["n_periodic_samples_in_phase"] < MIN_SAMPLES
    drop_reasons["below_min_samples"] = int(low_samples.sum())
    pf = pf[~low_samples].copy()

    miss_throttle = pf["throttle_sw_power_cap_rate"].isna()
    drop_reasons["missing_throttle_sw_power_cap_rate"] = int(miss_throttle.sum())
    pf = pf[~miss_throttle].copy()

    pf = pf.merge(
        selected_runs[["run_id", "model_facet", "policy_norm", "variant_label"]],
        on="run_id",
        how="inner",
    )
    print(f"phase_instances_before={n0} after_filters={len(pf)}")
    print(f"drop_reasons={drop_reasons}")
    if pf.empty:
        raise ValueError("No phase instances remain after reliability + integrity filtering.")

    # Aggregate to heatmap cells.
    group_cols = ["model_facet", "policy_norm", "variant_label", "phase_bucket"]
    has_time_weights = pf["phase_time_s"].notna().any() and (pf["phase_time_s"] > 0).any()
    if has_time_weights:
        pf["weighted_throttle"] = pf["throttle_sw_power_cap_rate"] * pf["phase_time_s"].fillna(0.0)
        agg = (
            pf.groupby(group_cols, dropna=False)
            .agg(
                weighted_sum=("weighted_throttle", "sum"),
                total_time_s=("phase_time_s", "sum"),
                n_phase_instances=("phase_instance_id", "nunique"),
            )
            .reset_index()
        )
        agg["throttle_mean"] = agg["weighted_sum"] / agg["total_time_s"]
    else:
        print("warning: phase_time_s unavailable/invalid; using simple mean for throttle aggregation.")
        agg = (
            pf.groupby(group_cols, dropna=False)
            .agg(
                throttle_mean=("throttle_sw_power_cap_rate", "mean"),
                n_phase_instances=("phase_instance_id", "nunique"),
            )
            .reset_index()
        )
        agg["total_time_s"] = np.nan

    bad = agg[(agg["throttle_mean"] < 0) | (agg["throttle_mean"] > 1)]
    if not bad.empty:
        raise ValueError(
            "Invalid throttle_mean outside [0,1] in aggregated cells:\n"
            + bad[
                ["model_facet", "policy_norm", "variant_label", "phase_bucket", "throttle_mean"]
            ].to_string(index=False)
        )

    print("aggregated_table:")
    print(
        agg[
            [
                "model_facet",
                "policy_norm",
                "variant_label",
                "phase_bucket",
                "throttle_mean",
                "n_phase_instances",
                "total_time_s",
            ]
        ]
        .sort_values(["variant_label", "model_facet", "policy_norm", "phase_bucket"])
        .to_string(index=False)
    )

    baseline = agg[agg["variant_label"] == "base_8192"].copy()
    knob = agg[agg["variant_label"] != "base_8192"].copy()

    # Build row ordering.
    baseline_rows = [(m, p, "base_8192") for m in MODEL_ORDER for p in POLICY_ORDER]
    knob_variants = sorted(knob["variant_label"].dropna().unique().tolist())
    knob_rows: list[tuple[str, str, str]] = []
    for v in knob_variants:
        for m in MODEL_ORDER:
            for p in POLICY_ORDER:
                if ((knob["variant_label"] == v) & (knob["model_facet"] == m) & (knob["policy_norm"] == p)).any():
                    knob_rows.append((m, p, v))

    phase_cols = [p for p in PHASE_ORDER if p in set(agg["phase_bucket"].tolist())]
    if not phase_cols:
        raise ValueError("No phase buckets left to plot after filtering.")

    mat_base = _build_matrix(baseline, baseline_rows, phase_cols)
    mat_knob = _build_matrix(knob, knob_rows, phase_cols) if knob_rows else pd.DataFrame(columns=phase_cols)

    vmax_candidates = [np.nanmax(mat_base.to_numpy(dtype=float)) if not mat_base.empty else np.nan]
    if not mat_knob.empty:
        vmax_candidates.append(np.nanmax(mat_knob.to_numpy(dtype=float)))
    vmax = np.nanmax(vmax_candidates)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), sharex=False, sharey=False)
    im0 = _plot_heatmap(axes[0], mat_base, "A) Baseline (base_8192)", vmax=vmax)
    if mat_knob.empty:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No rollout knob variants found", ha="center", va="center", transform=axes[1].transAxes)
    else:
        _plot_heatmap(axes[1], mat_knob, "B) Rollout knob variants", vmax=vmax)

    fig.subplots_adjust(top=0.90, bottom=0.20, wspace=0.18)
    cax = fig.add_axes([0.20, 0.08, 0.60, 0.03])
    cbar = fig.colorbar(im0, cax=cax, orientation="horizontal")
    cbar.set_label("throttle_sw_power_cap_rate (mean)")

    fig.suptitle("Throttle = sw_power_cap rate (fraction of periodic samples throttling)", y=0.995)
    fig.text(0.5, 0.955, f"Filtered by n_periodic_samples_in_phase >= {MIN_SAMPLES}", ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.935, "Baseline + rollout knob runs (checkpoint_continuation=false)", ha="center", va="center", fontsize=10)
    # keep manual layout so colorbar does not overlap with heatmaps

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
