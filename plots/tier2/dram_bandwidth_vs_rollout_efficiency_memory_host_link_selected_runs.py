"""DRAM Bandwidth vs Rollout Efficiency (Memory-Host Link).

Rollout-only phase correlation:
- X: timing_per_token_ms/gen
- Y: DRAM uJ per rollout output token

Grouping:
- Platform inferred from run_id:
  - 2gpu_h200 -> Sapphire Rapids
  - 4gpu_a100 -> Ice Lake
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


OUTPATH = Path("plots/out/figures/tier2/dram_bandwidth_vs_rollout_efficiency_memory_host_link_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PLATFORM_ORDER = ["Sapphire Rapids", "Ice Lake"]
PLATFORM_COLORS = {
    "Sapphire Rapids": "#4c78a8",
    "Ice Lake": "#f58518",
}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "Sapphire Rapids"
    if "4gpu_a100" in rid:
        return "Ice Lake"
    return "other"


def _fit_line(x: pd.Series, y: pd.Series) -> tuple[float, float, float, int]:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    m = xv.notna() & yv.notna()
    n = int(m.sum())
    if n < 2:
        return np.nan, np.nan, np.nan, n
    xx = xv[m].to_numpy(dtype=float)
    yy = yv[m].to_numpy(dtype=float)
    slope, intercept = np.polyfit(xx, yy, 1)
    r = float(pd.Series(xx).corr(pd.Series(yy)))
    return float(slope), float(intercept), r, n


def main() -> None:
    pf, _ = load_view("phase_fact_view")
    ml, _ = load_view("step_metrics_long")

    pf = pf[pf["run_id"].astype(str).isin(RUN_IDS)].copy()
    pf_before = pf.copy()
    pf = apply_analysis_ok(pf)
    print(f"phase_filtering={explain_filtering(pf_before, pf)}")

    # Rollout-only phase rows with DRAM and rollout tokens.
    phase = pf[
        ["run_id", "global_step_canonical", "phase_name", "dram_energy_j", "rollout_output_tokens_total"]
    ].copy()
    phase["phase_name"] = phase["phase_name"].astype(str).str.lower()
    phase = phase[phase["phase_name"] == "rollout"].copy()
    phase["dram_energy_j"] = pd.to_numeric(phase["dram_energy_j"], errors="coerce")
    phase["rollout_output_tokens_total"] = pd.to_numeric(phase["rollout_output_tokens_total"], errors="coerce")
    phase = phase[phase["rollout_output_tokens_total"] > 0].copy()
    phase["dram_uj_per_rollout_token"] = (phase["dram_energy_j"] * 1_000_000.0) / phase["rollout_output_tokens_total"]

    # Step metric x-axis.
    m = ml[ml["run_id"].astype(str).isin(RUN_IDS)].copy()
    m = m[m["metric_key"] == "timing_per_token_ms/gen"].copy()
    m["metric_value_float"] = pd.to_numeric(m["metric_value_float"], errors="coerce")
    x_step = (
        m.groupby(["run_id", "global_step_canonical"], dropna=False)["metric_value_float"]
        .mean()
        .reset_index(name="timing_per_token_ms_gen")
    )

    df = phase.merge(x_step, on=["run_id", "global_step_canonical"], how="inner")
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df = df[df["platform"].isin(PLATFORM_ORDER)].copy()
    df = df.dropna(subset=["timing_per_token_ms_gen", "dram_uj_per_rollout_token"]).copy()
    df = df[df["timing_per_token_ms_gen"] > 0].copy()

    if df.empty:
        raise ValueError("No valid rollout rows for DRAM-vs-efficiency correlation.")

    summary = (
        df.groupby("platform", dropna=False)
        .agg(
            n_points=("timing_per_token_ms_gen", "size"),
            x_mean=("timing_per_token_ms_gen", "mean"),
            y_mean=("dram_uj_per_rollout_token", "mean"),
            y_median=("dram_uj_per_rollout_token", "median"),
        )
        .reset_index()
        .sort_values("platform")
    )
    print("platform summary:")
    print(summary.to_string(index=False))

    fit_rows = []
    for platform in PLATFORM_ORDER:
        s = df[df["platform"] == platform]
        slope, intercept, r, n = _fit_line(s["timing_per_token_ms_gen"], s["dram_uj_per_rollout_token"])
        fit_rows.append(
            {
                "platform": platform,
                "n": n,
                "slope": slope,
                "intercept": intercept,
                "corr_r": r,
            }
        )
    fit_df = pd.DataFrame(fit_rows)
    print("fit stats:")
    print(fit_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for platform in PLATFORM_ORDER:
        s = df[df["platform"] == platform].copy()
        if s.empty:
            continue
        ax.scatter(
            s["timing_per_token_ms_gen"],
            s["dram_uj_per_rollout_token"],
            s=28,
            alpha=0.45,
            color=PLATFORM_COLORS[platform],
            edgecolors="none",
            label=f"{platform} samples",
            zorder=2,
        )

        slope, intercept, r, n = _fit_line(s["timing_per_token_ms_gen"], s["dram_uj_per_rollout_token"])
        if np.isfinite(slope) and np.isfinite(intercept):
            xs = np.linspace(
                float(s["timing_per_token_ms_gen"].min()),
                float(s["timing_per_token_ms_gen"].max()),
                80,
            )
            ys = slope * xs + intercept
            ax.plot(
                xs,
                ys,
                color=PLATFORM_COLORS[platform],
                linewidth=2.0,
                alpha=0.95,
                label=f"{platform} fit (r={r:.2f})",
                zorder=3,
            )
            # Annotate slope near the right side of each fit line.
            x_annot = float(np.quantile(xs, 0.82))
            y_annot = float(slope * x_annot + intercept)
            ax.text(
                x_annot,
                y_annot,
                f"slope={slope:.0f}",
                color=PLATFORM_COLORS[platform],
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
                zorder=4,
            )

    ax.set_xlabel("timing_per_token_ms/gen")
    ax.set_ylabel("DRAM energy per rollout output token (uJ/token)")
    ax.set_title("DRAM Bandwidth vs Rollout Efficiency (Rollout Phases)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.legend(frameon=False, loc="best")

    fig.text(
        0.5,
        0.01,
        "Grouping by platform mapping from run_id: 2gpu_h200 -> Sapphire Rapids, 4gpu_a100 -> Ice Lake",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
