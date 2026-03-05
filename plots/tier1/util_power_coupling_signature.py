"""Utilization-power coupling signature space.

Baseline-first plot with strict reliability gating for coupling metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier1/util_power_coupling_signature.png")
MIN_SAMPLES = 200
MIN_SAMPLES_PROXY_N_STEPS = 40
INCLUDE_ROLLOUT_OVERLAY = True

POLICY_ORDER = ("ppo", "remax", "grpo")
MODEL_ORDER = ("Llama", "Qwen")
POLICY_COLORS = {"ppo": "#1f77b4", "remax": "#ff7f0e", "grpo": "#2ca02c"}


def _norm_policy(policy: str) -> str:
    return str(policy).strip().lower().replace("remx", "remax")


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def main() -> None:
    df, _ = load_view("comparison_view")

    required = [
        "policy",
        "model",
        "experiment_variant",
        "rollout_max_batched_tokens",
        "is_checkpoint_continuation",
        "util_power_slope_b",
        "util_power_corr",
    ]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"comparison_view missing required columns: {missing_required}")

    # Reliability gate must be explicit for coupling metrics.
    reliability_col = None
    min_samples = MIN_SAMPLES
    using_proxy = False
    for candidate in ("n_samples_for_coupling",):
        if candidate in df.columns:
            reliability_col = candidate
            break
    if reliability_col is None:
        if "n_steps" in df.columns:
            reliability_col = "n_steps"
            min_samples = MIN_SAMPLES_PROXY_N_STEPS
            using_proxy = True
            print(
                "warning: 'n_samples_for_coupling' missing; using proxy reliability gate "
                f"'{reliability_col}>={min_samples}'. Please add n_samples_for_coupling upstream."
            )
        else:
            raise RuntimeError(
                "comparison_view is missing required reliability field 'n_samples_for_coupling' "
                "and fallback 'n_steps'. Cannot apply any reliability gate."
            )

    work = df.copy()
    work["policy_norm"] = work["policy"].map(_norm_policy)
    work["model_facet"] = work["model"].map(_model_facet)
    work["rollout_max_batched_tokens"] = pd.to_numeric(work["rollout_max_batched_tokens"], errors="coerce")
    work["util_power_slope_b"] = pd.to_numeric(work["util_power_slope_b"], errors="coerce")
    work["util_power_corr"] = pd.to_numeric(work["util_power_corr"], errors="coerce")
    work[reliability_col] = pd.to_numeric(work[reliability_col], errors="coerce")
    work["is_checkpoint_continuation"] = work["is_checkpoint_continuation"].fillna(False).astype(bool)

    reasons: dict[str, int] = {}
    n0 = len(work)

    # Baseline first, with optional rollout-scale overlay.
    if INCLUDE_ROLLOUT_OVERLAY:
        mask_baseline_or_knob = (
            ~work["is_checkpoint_continuation"]
            & work["rollout_max_batched_tokens"].isin([8192.0, 16384.0])
        )
        reasons["excluded_non_baseline_or_knob"] = int((~mask_baseline_or_knob).sum())
        work = work[mask_baseline_or_knob].copy()
    else:
        mask_baseline = (
            ~work["is_checkpoint_continuation"]
            & (work["rollout_max_batched_tokens"] == 8192.0)
        )
        reasons["excluded_non_baseline"] = int((~mask_baseline).sum())
        work = work[mask_baseline].copy()

    mask_model_policy = work["model_facet"].isin(MODEL_ORDER) & work["policy_norm"].isin(POLICY_ORDER)
    reasons["excluded_non_target_model_policy"] = int((~mask_model_policy).sum())
    work = work[mask_model_policy].copy()

    mask_samples = work[reliability_col] >= min_samples
    reasons["excluded_low_coupling_samples"] = int((~mask_samples).sum())
    work = work[mask_samples].copy()

    mask_slope = work["util_power_slope_b"].notna()
    reasons["excluded_missing_slope"] = int((~mask_slope).sum())
    work = work[mask_slope].copy()

    mask_corr = work["util_power_corr"].notna()
    reasons["excluded_missing_corr"] = int((~mask_corr).sum())
    work = work[mask_corr].copy()

    if "integrity_ok" in work.columns:
        integ = work["integrity_ok"].fillna(False).astype(bool)
        reasons["excluded_integrity_not_ok"] = int((~integ).sum())
        work = work[integ].copy()

    print(f"points_before={n0} points_after={len(work)}")
    print(f"drop_reasons={reasons}")

    if work.empty:
        raise ValueError("No points left after filtering and reliability gating.")

    corr_bad = work[(work["util_power_corr"] < -1.0) | (work["util_power_corr"] > 1.0)]
    if not corr_bad.empty:
        raise ValueError(
            "util_power_corr out of [-1,1] for rows:\n"
            + corr_bad[["policy_norm", "model_facet", "experiment_variant", "util_power_corr"]].to_string(index=False)
        )

    negative_slope = work[work["util_power_slope_b"] < 0]
    if not negative_slope.empty:
        print("warning: negative slope points detected:")
        print(
            negative_slope[
                ["policy_norm", "model_facet", "experiment_variant", "util_power_slope_b", "util_power_corr", reliability_col]
            ].to_string(index=False)
        )

    baseline_subset = work[work["rollout_max_batched_tokens"] == 8192.0].copy()
    expected_baseline = {(m, p) for m in MODEL_ORDER for p in POLICY_ORDER}
    got_baseline = set(zip(baseline_subset["model_facet"], baseline_subset["policy_norm"]))
    if expected_baseline - got_baseline:
        available = work[["model_facet", "policy_norm", "experiment_variant", "rollout_max_batched_tokens"]].to_dict(orient="records")
        raise ValueError(
            f"Expected 6 baseline points (2x3), missing baseline combos: {sorted(expected_baseline-got_baseline)}. "
            f"Available filtered points: {available}"
        )
    if len(baseline_subset) != 6:
        raise ValueError(f"Expected exactly 6 baseline points at rollout_max_batched_tokens=8192, got {len(baseline_subset)}")

    if not INCLUDE_ROLLOUT_OVERLAY:
        expected = {(m, p) for m in MODEL_ORDER for p in POLICY_ORDER}
        got = set(zip(work["model_facet"], work["policy_norm"]))
        if expected - got:
            available = work[["model_facet", "policy_norm", "experiment_variant", "rollout_max_batched_tokens"]].to_dict(orient="records")
            raise ValueError(
                f"Expected 6 baseline points (2x3), missing: {sorted(expected-got)}. "
                f"Available filtered points: {available}"
            )
        if len(work) != 6:
            raise ValueError(f"Baseline-only mode should have exactly 6 points, got {len(work)}")

    table_cols = [
        "policy_norm",
        "model_facet",
        "experiment_variant",
        "rollout_max_batched_tokens",
        "util_power_slope_b",
        "util_power_corr",
        reliability_col,
    ]
    print("plot_table:")
    print(work[table_cols].sort_values(["model_facet", "policy_norm", "rollout_max_batched_tokens"]).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    model_to_ax = dict(zip(MODEL_ORDER, axes))
    for model in MODEL_ORDER:
        ax = model_to_ax[model]
        sub = work[work["model_facet"] == model]
        for policy in POLICY_ORDER:
            psub = sub[sub["policy_norm"] == policy]
            if psub.empty:
                continue
            psub_base = psub[psub["rollout_max_batched_tokens"] == 8192.0]
            if not psub_base.empty:
                ax.scatter(
                    psub_base["util_power_slope_b"],
                    psub_base["util_power_corr"],
                    s=70,
                    marker="o",
                    color=POLICY_COLORS[policy],
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.9,
                    label=policy.upper(),
                )

            psub_knob = psub[psub["rollout_max_batched_tokens"] == 16384.0]
            if not psub_knob.empty:
                ax.scatter(
                    psub_knob["util_power_slope_b"],
                    psub_knob["util_power_corr"],
                    s=140,
                    marker="*",
                    color=POLICY_COLORS[policy],
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.95,
                    label=None,
                )

        ax.set_title(model)
        ax.set_xlabel("util→power slope (b)")
        ax.grid(alpha=0.2)
        if model == MODEL_ORDER[0]:
            ax.set_ylabel("corr(util, power)")

    # De-duplicate legend labels across facets.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        fig.legend(uniq.values(), uniq.keys(), title="policy", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.96))
    style_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markerfacecolor="white", markeredgecolor="black", label="baseline (8192)"),
        Line2D([0], [0], marker="*", color="black", linestyle="None", markerfacecolor="white", markeredgecolor="black", label="rollout scale (16384)"),
    ]
    fig.legend(style_handles, ["baseline (8192)", "rollout scale (16384)"], title="marker", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.88))

    fig.suptitle("Utilization–Power Coupling Signature Space", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.85))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
