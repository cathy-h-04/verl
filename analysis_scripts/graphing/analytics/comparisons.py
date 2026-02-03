#!/usr/bin/env python3
"""Quad-view comparison plots for multiple runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from ..core.base import ThemeConfig, apply_theme, EXCLUDED_OPERATIONS, PHASE_OPERATION_ORDER
from ..core.loaders import RunPaths, compute_sample_durations_seconds
from ..plotters.hardware import (
    GPUOverviewPlotter,
    PhaseComputeDensityPlotter,
    SmoothedTimeSeriesPlotter,
    ThermalSteadyStatePlotter,
)
from ..plotters.timing import (
    HierarchicalWaterfallPlotter,
    PhaseAggregatePlotter,
    PhaseBoxplotPlotter,
    PhaseCorrelationPlotter,
    PhaseEnergyTimeStackedPlotter,
    PhaseFocusRLPolicyPlotter,
    PhaseFocusRolloutPlotter,
    PhaseFocusTrainingPlotter,
    PhaseTimelinePlotter,
)
from ..plotters.efficiency import (
    BottleneckEvolutionPlotter,
    HardwareROIPlotter,
    LearningPricePlotter,
    MFUComparisonPlotter,
    OperationAggregatePlotter,
    OperationComparisonPlotter,
    SweepMetricsPlotter,
    ThroughputRewardFrontierPlotter,
    ThroughputVsLengthPlotter,
    TokenBottlenecksPlotter,
)


def get_clean_label(run_name: str) -> str:
    """Return a readable label like 'Small (1.5B) | PPO | 2 GPUs'."""
    parts = [p for p in run_name.lower().split("_") if p]

    def looks_like_date(token: str) -> bool:
        return len(token) == 8 and token.isdigit()

    def looks_like_time(token: str) -> bool:
        return len(token) == 6 and token.isdigit()

    while parts and (looks_like_date(parts[-1]) or looks_like_time(parts[-1])):
        parts = parts[:-1]

    ignored = {"sweep", "gsm8k"}
    parts = [p for p in parts if p not in ignored]

    algo_map = {
        "ppo": "PPO",
        "remax": "ReMax",
        "dpo": "DPO",
        "sft": "SFT",
    }
    model_map = {
        "small": "Small (1.5B)",
        "medium": "Medium",
        "large": "Large",
        "xl": "XL",
        "xxl": "XXL",
    }

    algo_label = None
    model_label = None
    gpu_count = None

    for token in parts:
        if algo_label is None and token in algo_map:
            algo_label = algo_map[token]
        if model_label is None and token in model_map:
            model_label = model_map[token]
        if gpu_count is None:
            if token.endswith("gpn") and token[:-3].isdigit():
                gpu_count = int(token[:-3])
            elif token.endswith("gpu") and token[:-3].isdigit():
                gpu_count = int(token[:-3])
            elif token.isdigit() and "gpu" in parts:
                gpu_count = int(token)

    if model_label is None:
        for token in parts:
            if token.endswith("b") and any(ch.isdigit() for ch in token):
                model_label = token.upper().replace("B", "B")
                break

    if model_label is None:
        model_label = "Unknown Model"
    if algo_label is None:
        algo_label = "Unknown Algo"
    gpu_label = f"{gpu_count} GPUs" if gpu_count is not None else "Unknown GPUs"

    return f"{model_label} | {algo_label} | {gpu_label}"


def _extract_algo_and_gpu(run_name: str) -> Tuple[str, Optional[int]]:
    parts = [p for p in run_name.lower().split("_") if p]

    def looks_like_date(token: str) -> bool:
        return len(token) == 8 and token.isdigit()

    def looks_like_time(token: str) -> bool:
        return len(token) == 6 and token.isdigit()

    while parts and (looks_like_date(parts[-1]) or looks_like_time(parts[-1])):
        parts = parts[:-1]

    ignored = {"sweep", "gsm8k"}
    parts = [p for p in parts if p not in ignored]

    algo_map = {
        "ppo": "PPO",
        "remax": "ReMax",
        "dpo": "DPO",
        "sft": "SFT",
    }

    algo_label = None
    gpu_count = None

    for token in parts:
        if algo_label is None and token in algo_map:
            algo_label = algo_map[token]
        if gpu_count is None:
            if token.endswith("gpn") and token[:-3].isdigit():
                gpu_count = int(token[:-3])
            elif token.endswith("gpu") and token[:-3].isdigit():
                gpu_count = int(token[:-3])
            elif token.isdigit() and "gpu" in parts:
                gpu_count = int(token)

    if algo_label is None:
        algo_label = "Unknown Algo"

    return algo_label, gpu_count


def _merge_bounds(
    bounds: Optional[Tuple[float, float]],
    new_bounds: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if new_bounds is None:
        return bounds
    if bounds is None:
        return new_bounds
    return (min(bounds[0], new_bounds[0]), max(bounds[1], new_bounds[1]))


def _finite_min_max(values: Iterable[float]) -> Optional[Tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return (float(arr.min()), float(arr.max()))


def _pad_bounds(
    bounds: Optional[Tuple[float, float]],
    pad_ratio: float = 0.05,
    min_floor: Optional[float] = None,
    max_ceiling: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    if bounds is None:
        return None
    min_v, max_v = bounds
    if not np.isfinite(min_v) or not np.isfinite(max_v):
        return None
    if min_v == max_v:
        pad = 1.0 if min_v == 0 else abs(min_v) * pad_ratio
    else:
        pad = (max_v - min_v) * pad_ratio
    min_v -= pad
    max_v += pad
    if min_floor is not None:
        min_v = max(min_v, min_floor)
    if max_ceiling is not None:
        max_v = min(max_v, max_ceiling)
    return (min_v, max_v)


def _bounds_for_column(
    runs: Sequence[RunPaths],
    df_attr: str,
    col: str,
) -> Optional[Tuple[float, float]]:
    bounds: Optional[Tuple[float, float]] = None
    for run in runs:
        df = getattr(run, df_attr, None)
        if df is None or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bounds = _merge_bounds(bounds, _finite_min_max(series))
    return _pad_bounds(bounds)


def _apply_limits(
    ax: plt.Axes,
    x_bounds: Optional[Tuple[float, float]],
    y_bounds: Optional[Tuple[float, float]],
) -> None:
    if x_bounds is not None:
        ax.set_xlim(*x_bounds)
    if y_bounds is not None:
        ax.set_ylim(*y_bounds)


def _create_quad_figure(base_size: Tuple[float, float]) -> Tuple[plt.Figure, GridSpec]:
    fig = plt.figure(figsize=(base_size[0] * 2.0, base_size[1] * 2.0))
    outer = GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.35)
    return fig, outer


def _make_inner_axes(
    fig: plt.Figure,
    outer: GridSpec,
    index: int,
    rows: int,
    cols: int,
    sharex: bool = False,
    sharey: bool = False,
) -> np.ndarray:
    row = index // 2
    col = index % 2
    inner = GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[row, col], wspace=0.3, hspace=0.35)
    axes = np.empty((rows, cols), dtype=object)
    sharex_ax = None
    sharey_ax = None
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(
                inner[r, c],
                sharex=sharex_ax if sharex else None,
                sharey=sharey_ax if sharey else None,
            )
            if sharex_ax is None:
                sharex_ax = ax
            if sharey_ax is None:
                sharey_ax = ax
            axes[r, c] = ax
    if rows == 1 or cols == 1:
        return np.ravel(axes)
    return axes


QUAD_LABEL_SIZE = 12
QUAD_TICK_SIZE = 10
QUAD_TITLE_SIZE = 12
QUAD_QUADRANT_TITLE_SIZE = 13


def _scale_axes_text(axes: np.ndarray) -> None:
    for ax in np.ravel(axes):
        if not hasattr(ax, "set_title"):
            continue
        ax.title.set_fontsize(QUAD_TITLE_SIZE)
        ax.xaxis.label.set_size(QUAD_LABEL_SIZE)
        ax.yaxis.label.set_size(QUAD_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=QUAD_TICK_SIZE)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(QUAD_TICK_SIZE)


def _add_quadrant_title(fig: plt.Figure, axes: np.ndarray, title: str) -> None:
    positions = [ax.get_position() for ax in np.ravel(axes) if ax.get_visible()]
    if not positions:
        return
    left = min(p.x0 for p in positions)
    right = max(p.x1 for p in positions)
    top = max(p.y1 for p in positions)
    x = (left + right) / 2.0
    y = min(top + 0.025, 0.985)
    fig.text(
        x,
        y,
        title,
        ha="center",
        va="bottom",
        fontsize=QUAD_QUADRANT_TITLE_SIZE,
        fontweight="bold",
    )


def _finalize_quad(fig: plt.Figure, axes_list: Optional[Sequence[np.ndarray]] = None) -> None:
    if axes_list:
        for axes in axes_list:
            _scale_axes_text(axes)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.10, wspace=0.25, hspace=0.40)


def _phase_timeline_x(df: pd.DataFrame) -> Optional[pd.Series]:
    if "timestamp_aligned_unix" not in df.columns:
        return None
    ts = pd.to_numeric(df["timestamp_aligned_unix"], errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return None
    return (ts - ts.min()) / 60.0


def _phase_focus_df(
    run: RunPaths,
    focus_window: Tuple[int, int],
) -> Optional[Tuple[pd.DataFrame, str]]:
    df = run.annotated_df.copy()
    if "iteration" not in df.columns or "phase_name" not in df.columns:
        return None
    df = df[df["phase_name"] != "idle"]
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.dropna(subset=["iteration"])
    if focus_window:
        start_iter, end_iter = focus_window
        df = df[(df["iteration"] >= start_iter) & (df["iteration"] <= end_iter)]
    if df.empty:
        return None
    if "timestamp_aligned_unix" in df.columns:
        time_raw = pd.to_numeric(df["timestamp_aligned_unix"], errors="coerce")
        time_ref = time_raw.min()
        df["focus_time_min"] = (time_raw - time_ref) / 60.0
        return df, "focus_time_min"
    if "elapsed_seconds" in df.columns:
        time_raw = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
        time_ref = time_raw.min()
        df["focus_time_min"] = (time_raw - time_ref) / 60.0
        return df, "focus_time_min"
    return df, "iteration"


def _phase_aggregate_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "phase_name" not in df.columns:
        return None
    work = df.copy()
    work = work[work["phase_name"] != "idle"]
    if work.empty:
        return None
    dt = compute_sample_durations_seconds(work)
    work["sample_dt_s"] = dt
    work["sample_energy_j"] = pd.to_numeric(work.get("power_draw_w"), errors="coerce") * dt
    grouped = work.groupby("phase_name", dropna=False)
    summary = grouped.agg(
        avg_power_w=("power_draw_w", "mean"),
        avg_gpu_util=("gpu_util_percent", "mean"),
        avg_temp_c=("temperature_c", "mean"),
        total_energy_j=("sample_energy_j", "sum"),
    ).reset_index()
    summary["energy_wh"] = summary["total_energy_j"] / 3600.0
    summary["phase_name"] = summary["phase_name"].fillna("unknown").astype(str)
    return summary


def _operation_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "operation" not in df.columns:
        return None
    work = df.copy()
    work["operation"] = work["operation"].fillna("unknown").astype(str)
    work = work[~work["operation"].isin(EXCLUDED_OPERATIONS)]
    if work.empty:
        return None
    dt = compute_sample_durations_seconds(work)
    power = pd.to_numeric(work.get("power_draw_w"), errors="coerce")
    work["sample_dt_s"] = dt
    work["sample_energy_j"] = power * dt
    grouped = work.groupby("operation", dropna=False)
    summary = grouped.agg(
        avg_power_w=("power_draw_w", "mean"),
        avg_gpu_util=("gpu_util_percent", "mean"),
        total_duration_s=("sample_dt_s", "sum"),
        total_energy_j=("sample_energy_j", "sum"),
        sample_count=("operation", "size"),
    ).reset_index()
    summary["energy_wh"] = summary["total_energy_j"] / 3600.0
    summary = summary.sort_values("energy_wh", ascending=False)
    return summary


def _learning_price_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    reward_col = "data.val-core/openai/gsm8k/reward/mean@1"
    tokens_col = "data.perf/total_num_tokens"
    reward = pd.to_numeric(df.get(reward_col), errors="coerce")
    tokens = pd.to_numeric(df.get(tokens_col), errors="coerce")
    if "step" in df.columns:
        df = df.sort_values("step")
        reward = pd.to_numeric(df.get(reward_col), errors="coerce")
        tokens = pd.to_numeric(df.get(tokens_col), errors="coerce")
    diffs = tokens.diff()
    non_decreasing_ratio = (diffs.dropna() >= 0).mean() if diffs.notna().any() else 0.0
    if non_decreasing_ratio >= 0.9:
        tokens_cum = tokens
    else:
        throughput = pd.to_numeric(df.get("data.perf/throughput"), errors="coerce")
        time_per_step = pd.to_numeric(df.get("data.perf/time_per_step"), errors="coerce")
        per_step_tokens = throughput * time_per_step
        tokens_cum = per_step_tokens.fillna(0).cumsum()
    return tokens_cum, reward


def _draw_hardware_roi(ax: plt.Axes, df: Optional[pd.DataFrame], reward_col: str) -> Optional[plt.Axes]:
    if df is None:
        ax.set_title("Missing merged sweep CSV.")
        return None
    if "step" not in df.columns:
        ax.set_title("Missing step column.")
        return None

    work = df.copy().sort_values("step")
    x = pd.to_numeric(work["step"], errors="coerce")

    mfu_actor = pd.to_numeric(work.get("data.perf/mfu/actor"), errors="coerce")
    mfu_critic = pd.to_numeric(work.get("data.perf/mfu/critic"), errors="coerce")
    reward = pd.to_numeric(work.get(reward_col), errors="coerce")

    if not isinstance(mfu_actor, pd.Series):
        mfu_actor = pd.Series([np.nan] * len(work), index=work.index)
    if not isinstance(mfu_critic, pd.Series):
        mfu_critic = pd.Series([np.nan] * len(work), index=work.index)

    x_vals = x.to_numpy()
    mfu_actor = mfu_actor.reindex(work.index)
    mfu_critic = mfu_critic.reindex(work.index)
    if len(mfu_actor) != len(x_vals) or len(mfu_critic) != len(x_vals):
        ax.set_title("MFU series length mismatch.")
        return None

    ax.plot(x_vals, mfu_actor.to_numpy(), label="MFU Actor", color="#2980b9", linewidth=1.2)
    ax.plot(x_vals, mfu_critic.to_numpy(), label="MFU Critic", color="#8e44ad", linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("MFU")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    if isinstance(reward, pd.Series):
        reward_roll = reward.rolling(window=10, min_periods=3).mean()
        reward_roll = reward_roll.reindex(work.index)
        if len(reward_roll) == len(x):
            reward_vals = reward_roll.to_numpy()
            if np.isfinite(reward_vals).sum() > 1:
                ax2.plot(x_vals, reward_vals, label="Reward (10-step mean)", color="#2ecc71", linewidth=1.5)
                ax2.set_ylabel("Reward")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines2:
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    elif lines:
        ax.legend(lines, labels, loc="upper right")

    ax.set_title("Hardware ROI (MFU & Reward)")
    return ax2

def _quad_gpu_overview(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = _bounds_for_column(runs, "annotated_df", "elapsed_minutes")
    metrics = [
        ("gpu_util_percent", (0, 0)),
        ("memory_used_gb", (0, 1)),
        ("power_draw_w", (0, 2)),
        ("temperature_c", (1, 0)),
        ("sm_clock_mhz", (1, 1)),
        ("memory_util_percent", (1, 2)),
    ]
    y_bounds = {metric: _bounds_for_column(runs, "annotated_df", metric) for metric, _ in metrics}

    apply_theme(theme)
    fig, outer = _create_quad_figure((16, 10))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 2, 3, sharex=True)
        plotter = GPUOverviewPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        for metric, (r, c) in metrics:
            ax = axes[r, c]
            _apply_limits(ax, x_bounds, y_bounds.get(metric))
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{GPUOverviewPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_smoothed_timeseries(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metrics = SmoothedTimeSeriesPlotter.metrics
    x_bounds = _bounds_for_column(runs, "annotated_df", "elapsed_minutes")
    y_bounds = {metric: _bounds_for_column(runs, "annotated_df", metric) for metric, _ in metrics}

    apply_theme(theme)
    fig, outer = _create_quad_figure((14, 12))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, len(metrics), 1, sharex=True)
        plotter = SmoothedTimeSeriesPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        for ax, (metric, _) in zip(np.ravel(axes), metrics):
            _apply_limits(ax, x_bounds, y_bounds.get(metric))
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{SmoothedTimeSeriesPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_thermal_steady_state(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = _bounds_for_column(runs, "annotated_df", "elapsed_minutes")
    y_bounds = _bounds_for_column(runs, "annotated_df", "temperature_c")

    apply_theme(theme)
    fig, outer = _create_quad_figure((12, 5))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = ThermalSteadyStatePlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, y_bounds)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{ThermalSteadyStatePlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_phase_compute_density(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = _bounds_for_column(runs, "annotated_df", "gpu_util_percent")
    y_bounds = _bounds_for_column(runs, "annotated_df", "power_draw_w")

    apply_theme(theme)
    fig, outer = _create_quad_figure((8, 6))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = PhaseComputeDensityPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, y_bounds)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{PhaseComputeDensityPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_phase_timeline(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metrics = PhaseTimelinePlotter.metric_columns
    x_bounds = None
    y_bounds = {metric: None for metric, _ in metrics}

    for run in runs:
        x_vals = _phase_timeline_x(run.annotated_df)
        if x_vals is not None:
            x_bounds = _merge_bounds(x_bounds, _finite_min_max(x_vals))
        for metric, _ in metrics:
            if metric not in run.annotated_df.columns:
                continue
            series = pd.to_numeric(run.annotated_df[metric], errors="coerce")
            y_bounds[metric] = _merge_bounds(y_bounds[metric], _finite_min_max(series))

    x_bounds = _pad_bounds(x_bounds, min_floor=0.0)
    y_bounds = {metric: _pad_bounds(bounds) for metric, bounds in y_bounds.items()}

    apply_theme(theme)
    fig, outer = _create_quad_figure((14, 9))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, len(metrics), 1, sharex=True)
        plotter = PhaseTimelinePlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        plotter.annotate(fig, axes)
        for ax, (metric, _) in zip(np.ravel(axes), metrics):
            _apply_limits(ax, x_bounds, y_bounds.get(metric))
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{PhaseTimelinePlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_phase_aggregate(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metrics = [
        ("avg_gpu_util", "Average GPU Util (%)"),
        ("avg_power_w", "Average Power (W)"),
        ("avg_temp_c", "Average Temp (°C)"),
        ("energy_wh", "Total Energy (Wh)"),
    ]

    summaries: List[pd.DataFrame] = []
    run_labels: List[str] = []
    run_meta: List[Tuple[str, Optional[int], str]] = []
    for run in runs:
        summary = _phase_aggregate_summary(run.annotated_df)
        if summary is None:
            continue
        summaries.append(summary)
        label = get_clean_label(run.run_name)
        run_labels.append(label)
        algo_label, gpu_count = _extract_algo_and_gpu(run.run_name)
        run_meta.append((algo_label, gpu_count, label))

    apply_theme(theme)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = np.asarray(axes)

    if not summaries:
        for ax in np.ravel(axes):
            ax.set_title("No non-idle phase data.")
            ax.axis("off")
        out_path = output_dir / f"{PhaseAggregatePlotter.plot_name}_quad_view.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
        plt.close(fig)
        return out_path

    phases_set = set()
    for summary in summaries:
        phases_set.update(summary["phase_name"].dropna().astype(str).tolist())

    preferred_phases = ["rollout", "rl_policy", "training", "idle", "unknown"]
    phases = [p for p in preferred_phases if p in phases_set]
    phases.extend(sorted(p for p in phases_set if p not in phases))

    y_bounds: Dict[str, Optional[Tuple[float, float]]] = {metric: None for metric, _ in metrics}
    for summary in summaries:
        for metric, _ in metrics:
            if metric not in summary.columns:
                continue
            series = pd.to_numeric(summary[metric], errors="coerce")
            y_bounds[metric] = _merge_bounds(y_bounds[metric], _finite_min_max(series))
    y_bounds = {metric: _pad_bounds(bounds) for metric, bounds in y_bounds.items()}

    num_runs = len(summaries)
    group_width = 0.8
    bar_width = group_width / max(1, num_runs)
    x_base = np.arange(len(phases))

    algo_hatches = {
        "ReMax": "\\\\",
    }

    algo_cmaps = {
        "PPO": "Blues",
        "ReMax": "Oranges",
        "DPO": "Greens",
        "SFT": "Purples",
    }

    algo_gpu_counts: Dict[str, List[int]] = {}
    for algo_label, gpu_count, _ in run_meta:
        if gpu_count is not None:
            algo_gpu_counts.setdefault(algo_label, []).append(gpu_count)

    algo_color_map: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}
    for algo_label, counts in algo_gpu_counts.items():
        unique_counts = sorted(set(counts))
        if not unique_counts:
            continue
        cmap_name = algo_cmaps.get(algo_label, "Greys")
        cmap = plt.get_cmap(cmap_name)
        if len(unique_counts) == 1:
            algo_color_map[(algo_label, unique_counts[0])] = cmap(0.65)
            continue
        for idx, count in enumerate(unique_counts):
            frac = idx / (len(unique_counts) - 1)
            shade = 0.35 + 0.5 * frac
            algo_color_map[(algo_label, count)] = cmap(shade)

    unknown_color = "#7f8c8d"

    for metric_idx, (metric, title) in enumerate(metrics):
        ax = np.ravel(axes)[metric_idx]
        for run_idx, summary in enumerate(summaries):
            algo_label, gpu_count, _ = run_meta[run_idx]
            color = (
                algo_color_map.get((algo_label, gpu_count))
                if gpu_count is not None
                else None
            )
            if color is None:
                color = unknown_color
            hatch = algo_hatches.get(algo_label, "")
            values = []
            for phase in phases:
                match = summary.loc[summary["phase_name"] == phase, metric]
                value = match.iloc[0] if not match.empty else np.nan
                values.append(value)
            offset = x_base - group_width / 2 + (run_idx + 0.5) * bar_width
            label = run_labels[run_idx] if metric_idx == 0 else "_nolegend_"
            ax.bar(
                offset,
                values,
                width=bar_width,
                color=color,
                alpha=0.9,
                label=label,
                hatch=hatch,
                edgecolor="#2c3e50",
                linewidth=0.4,
            )

        ax.set_title(title)
        ax.set_xticks(x_base)
        ax.set_xticklabels(phases, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=theme.grid_alpha)
        _apply_limits(ax, None, y_bounds.get(metric))

    legend_handles = []
    for algo_label, gpu_count, run_label in run_meta:
        color = (
            algo_color_map.get((algo_label, gpu_count))
            if gpu_count is not None
            else None
        )
        if color is None:
            color = unknown_color
        hatch = algo_hatches.get(algo_label, "")
        legend_handles.append(Patch(facecolor=color, edgecolor="#2c3e50", hatch=hatch, label=run_label))

    fig.suptitle("Phase Aggregates (Grouped by Run)", fontsize=12, fontweight="bold", y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=min(4, len(legend_handles)),
        frameon=False,
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.35, wspace=0.25)

    out_path = output_dir / f"{PhaseAggregatePlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_phase_focus(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
    plotter_cls,
    plot_name: str,
) -> Optional[Path]:
    metrics = plotter_cls.metrics
    focus_window = plotter_cls.focus_window
    focus_phase = plotter_cls.focus_phase

    run_inputs: List[Tuple[RunPaths, pd.DataFrame, str]] = []
    force_iteration = False
    for run in runs:
        result = _phase_focus_df(run, focus_window)
        if result is None:
            continue
        df, x_col = result
        if x_col == "iteration":
            force_iteration = True
        run_inputs.append((run, df, x_col))

    run_data: List[Tuple[pd.DataFrame, str, str, Optional[int], str]] = []
    for run, df, x_col in run_inputs:
        if "phase_name" not in df.columns:
            continue
        df = df[df["phase_name"] == focus_phase].copy()
        if df.empty:
            continue
        if force_iteration:
            if "iteration" not in df.columns:
                continue
            df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
            x_col = "iteration"
        label = get_clean_label(run.run_name)
        algo_label, gpu_count = _extract_algo_and_gpu(run.run_name)
        run_data.append((df, x_col, label, gpu_count, algo_label))

    if not run_data:
        return None

    x_bounds = None
    y_bounds = {metric: None for metric, _ in metrics}
    for df, x_col, _, _, _ in run_data:
        if x_col in df.columns:
            x_bounds = _merge_bounds(x_bounds, _finite_min_max(pd.to_numeric(df[x_col], errors="coerce")))
        for metric, _ in metrics:
            if metric not in df.columns:
                continue
            series = pd.to_numeric(df[metric], errors="coerce")
            y_bounds[metric] = _merge_bounds(y_bounds[metric], _finite_min_max(series))

    x_bounds = _pad_bounds(x_bounds)
    y_bounds = {metric: _pad_bounds(bounds) for metric, bounds in y_bounds.items()}

    algo_cmaps = {
        "PPO": "Blues",
        "ReMax": "Oranges",
        "DPO": "Greens",
        "SFT": "Purples",
    }

    algo_gpu_counts: Dict[str, List[int]] = {}
    for _, _, _, gpu_count, algo_label in run_data:
        if gpu_count is not None:
            algo_gpu_counts.setdefault(algo_label, []).append(gpu_count)

    algo_color_map: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}
    for algo_label, counts in algo_gpu_counts.items():
        unique_counts = sorted(set(counts))
        if not unique_counts:
            continue
        cmap_name = algo_cmaps.get(algo_label, "Greys")
        cmap = plt.get_cmap(cmap_name)
        if len(unique_counts) == 1:
            algo_color_map[(algo_label, unique_counts[0])] = cmap(0.65)
            continue
        for idx, count in enumerate(unique_counts):
            frac = idx / (len(unique_counts) - 1)
            shade = 0.35 + 0.5 * frac
            algo_color_map[(algo_label, count)] = cmap(shade)

    unknown_color = "#7f8c8d"
    algo_linestyles = {
        "PPO": "-",
        "ReMax": (0, (6, 4)),
        "DPO": (0, (5, 2, 1, 2)),
        "SFT": (0, (1, 6)),
    }

    apply_theme(theme)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 12), sharex=True)
    axes = np.asarray(axes)

    for ax, (metric, ylabel) in zip(np.ravel(axes), metrics):
        for df, x_col, label, gpu_count, algo_label in run_data:
            if metric not in df.columns or x_col not in df.columns:
                continue
            x_vals = pd.to_numeric(df[x_col], errors="coerce")
            y_vals = pd.to_numeric(df[metric], errors="coerce")
            mask = x_vals.notna() & y_vals.notna()
            if not mask.any():
                continue
            color = (
                algo_color_map.get((algo_label, gpu_count))
                if gpu_count is not None
                else None
            )
            if color is None:
                color = unknown_color
            linestyle = algo_linestyles.get(algo_label, "-")
            ax.plot(
                x_vals[mask],
                y_vals[mask],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                alpha=0.9,
                label=label,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=theme.grid_alpha)
        _apply_limits(ax, x_bounds, y_bounds.get(metric))

    x_label = "Iteration" if force_iteration else "Time (minutes)"
    np.ravel(axes)[-1].set_xlabel(x_label)

    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=min(4, len(labels)),
            frameon=False,
        )

    fig.suptitle(f"Iteration Focus: {focus_phase}", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.35)

    out_path = output_dir / f"{plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_phase_energy_time_stacked(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    apply_theme(theme)
    fig, outer = _create_quad_figure((10, 6))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = PhaseEnergyTimeStackedPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        ax.set_ylim(0, 1)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{PhaseEnergyTimeStackedPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_phase_boxplots(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metrics = [
        ("gpu_util_percent", "GPU Util (%)"),
        ("power_draw_w", "Power (W)"),
        ("temperature_c", "Temp (°C)"),
        ("memory_used_gb", "Mem Used (GB)"),
    ]

    if sns is None:
        apply_theme(theme)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax in np.ravel(axes):
            ax.set_title("Seaborn not available for boxplots.")
            ax.axis("off")
        out_path = output_dir / f"{PhaseBoxplotPlotter.plot_name}_quad_view.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
        plt.close(fig)
        return out_path

    frames: List[pd.DataFrame] = []
    run_meta: List[Tuple[str, Optional[int], str]] = []
    for run in runs:
        df = run.annotated_df
        if "phase_name" not in df.columns:
            continue
        df = df[df["phase_name"] != "idle"].copy()
        if df.empty:
            continue
        label = get_clean_label(run.run_name)
        algo_label, gpu_count = _extract_algo_and_gpu(run.run_name)
        df["run_label"] = label
        df["algo_label"] = algo_label
        df["gpu_count"] = gpu_count
        frames.append(df)
        run_meta.append((algo_label, gpu_count, label))

    apply_theme(theme)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = np.asarray(axes)

    if not frames:
        for ax in np.ravel(axes):
            ax.set_title("No non-idle phase data.")
            ax.axis("off")
        out_path = output_dir / f"{PhaseBoxplotPlotter.plot_name}_quad_view.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
        plt.close(fig)
        return out_path

    combined = pd.concat(frames, ignore_index=True)

    phases_set = set(combined["phase_name"].dropna().astype(str).tolist())
    preferred_phases = ["rollout", "rl_policy", "training", "idle", "unknown"]
    phases = [p for p in preferred_phases if p in phases_set]
    phases.extend(sorted(p for p in phases_set if p not in phases))

    run_labels = [label for _, _, label in run_meta]
    remax_labels = {label for algo_label, _, label in run_meta if algo_label == "ReMax"}

    algo_cmaps = {
        "PPO": "Blues",
        "ReMax": "Oranges",
        "DPO": "Greens",
        "SFT": "Purples",
    }

    algo_gpu_counts: Dict[str, List[int]] = {}
    for algo_label, gpu_count, _ in run_meta:
        if gpu_count is not None:
            algo_gpu_counts.setdefault(algo_label, []).append(gpu_count)

    algo_color_map: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}
    for algo_label, counts in algo_gpu_counts.items():
        unique_counts = sorted(set(counts))
        if not unique_counts:
            continue
        cmap_name = algo_cmaps.get(algo_label, "Greys")
        cmap = plt.get_cmap(cmap_name)
        if len(unique_counts) == 1:
            algo_color_map[(algo_label, unique_counts[0])] = cmap(0.65)
            continue
        for idx, count in enumerate(unique_counts):
            frac = idx / (len(unique_counts) - 1)
            shade = 0.35 + 0.5 * frac
            algo_color_map[(algo_label, count)] = cmap(shade)

    unknown_color = "#7f8c8d"
    palette: Dict[str, Tuple[float, float, float, float]] = {}
    for algo_label, gpu_count, label in run_meta:
        color = (
            algo_color_map.get((algo_label, gpu_count))
            if gpu_count is not None
            else None
        )
        if color is None:
            color = unknown_color
        palette[label] = color

    y_bounds: Dict[str, Optional[Tuple[float, float]]] = {metric: None for metric, _ in metrics}
    for metric, _ in metrics:
        if metric not in combined.columns:
            continue
        series = pd.to_numeric(combined[metric], errors="coerce")
        y_bounds[metric] = _merge_bounds(y_bounds[metric], _finite_min_max(series))
    y_bounds = {metric: _pad_bounds(bounds) for metric, bounds in y_bounds.items()}

    for idx, (metric, title) in enumerate(metrics):
        ax = np.ravel(axes)[idx]
        if metric not in combined.columns:
            ax.set_title(f"{title} (missing column: {metric})")
            ax.axis("off")
            continue
        sns.boxplot(
            data=combined,
            x="phase_name",
            y=metric,
            hue="run_label",
            order=phases,
            hue_order=run_labels,
            ax=ax,
            palette=palette,
        )
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)
        for label_text in ax.get_xticklabels():
            label_text.set_ha("right")
        ax.grid(True, axis="y", alpha=theme.grid_alpha)
        _apply_limits(ax, None, y_bounds.get(metric))

        legend = ax.get_legend()
        if legend:
            legend.remove()

        num_hue = len(run_labels)
        if num_hue:
            box_patches = list(ax.artists)
            if not box_patches:
                box_patches = [p for p in ax.patches if isinstance(p, PathPatch)]

            def _closest_label(color) -> Optional[str]:
                if color is None:
                    return None
                color_arr = np.asarray(color, dtype=float)
                if color_arr.ndim > 1:
                    color_arr = color_arr[0]
                if color_arr.size >= 3:
                    color_arr = color_arr[:3]
                best_label = None
                best_dist = None
                for label, palette_color in palette.items():
                    pal = np.asarray(palette_color, dtype=float)[:3]
                    dist = float(np.linalg.norm(color_arr - pal))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_label = label
                return best_label

            for patch in box_patches:
                run_label = _closest_label(patch.get_facecolor())
                patch.set_hatch("." if run_label in remax_labels else "")
                patch.set_edgecolor("#2c3e50")
                patch.set_linewidth(0.6)

    legend_handles = []
    for algo_label, gpu_count, label in run_meta:
        color = (
            algo_color_map.get((algo_label, gpu_count))
            if gpu_count is not None
            else None
        )
        if color is None:
            color = unknown_color
        hatch = "." if label in remax_labels else ""
        legend_handles.append(Patch(facecolor=color, edgecolor="#2c3e50", hatch=hatch, label=label))

    fig.suptitle("Phase Boxplots (Grouped by Run)", fontsize=12, fontweight="bold", y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=min(4, len(legend_handles)),
        frameon=False,
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.35, wspace=0.25)

    out_path = output_dir / f"{PhaseBoxplotPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_phase_correlations(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    outputs: List[Path] = []
    for phase in ("rl_policy", "rollout", "training"):
        out_path = _quad_phase_correlations_for_phase(runs, output_dir, theme, phase)
        if out_path is not None:
            outputs.append(out_path)
    return outputs[0] if outputs else None


def _quad_phase_correlations_for_phase(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
    target_phase: str,
) -> Optional[Path]:
    metric_candidates = [m for m, _ in PhaseCorrelationPlotter.metrics]
    metric_cols: Optional[List[str]] = None

    for run in runs:
        df = run.annotated_df
        cols = [m for m in metric_candidates if m in df.columns]
        metric_cols = cols if metric_cols is None else [m for m in metric_cols if m in cols]

    if not metric_cols:
        return None

    def _phase_has_data(df: pd.DataFrame, phase: str) -> bool:
        if "phase_name" not in df.columns:
            return False
        phase_df = df[df["phase_name"] == phase]
        return phase_df[metric_cols].dropna().shape[0] >= 2

    apply_theme(theme)
    fig, outer = _create_quad_figure((8, 7))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        ax = np.ravel(axes)[0]
        if sns is None:
            ax.set_title("Seaborn not available for heatmaps.")
        else:
            df = run.annotated_df
            if not _phase_has_data(df, target_phase):
                ax.set_title(f"{target_phase} correlations (missing data)")
                ax.axis("off")
            else:
                phase_df = df[df["phase_name"] == target_phase]
                corr = phase_df[metric_cols].corr()
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar=False,
                    ax=ax,
                )
                ax.set_title(f"{target_phase} correlations")
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    fig.subplots_adjust(right=0.90)

    mappables = []
    for axes in axes_list:
        for ax in np.ravel(axes):
            if ax.collections:
                mappables.append(ax.collections[0])
    if mappables:
        cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(mappables[0], cax=cax)
        cbar.set_label("Correlation")
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{PhaseCorrelationPlotter.plot_name}_{target_phase}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_hierarchical_waterfall(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    max_total = None
    for run in runs:
        df = run.timings_df
        if df is None or df.empty:
            continue
        work = df.copy()
        work["iteration"] = pd.to_numeric(work["iteration"], errors="coerce")
        work = work.dropna(subset=["iteration"])
        if work.empty:
            continue
        target_iter = HierarchicalWaterfallPlotter.target_iteration
        if target_iter not in work["iteration"].unique():
            unique_iters = sorted(work["iteration"].unique())
            if not unique_iters:
                continue
            target_iter = unique_iters[len(unique_iters) // 2]
        iter_df = work[work["iteration"] == target_iter]
        if iter_df.empty:
            continue
        phase_totals: List[float] = []
        for phase in ["rollout", "rl_policy", "training"]:
            phase_row = iter_df[iter_df["phase"] == phase]
            if phase_row.empty:
                continue
            row = phase_row.iloc[0].to_dict()
            ops = {
                k: float(v)
                for k, v in row.items()
                if k not in {"iteration", "phase", "timestamp"}
                and isinstance(v, (int, float))
            }
            ordered_ops = PHASE_OPERATION_ORDER.get(phase, [])
            durations = []
            if ordered_ops:
                for op in ordered_ops:
                    if op in ops and op not in EXCLUDED_OPERATIONS:
                        durations.append(ops[op])
            else:
                for op, dur in ops.items():
                    if op in EXCLUDED_OPERATIONS:
                        continue
                    durations.append(dur)
            if durations:
                phase_totals.append(sum(durations))
        if phase_totals:
            total_max = max(phase_totals)
            max_total = total_max if max_total is None else max(max_total, total_max)

    x_bounds = _pad_bounds((0.0, max_total)) if max_total is not None else None

    apply_theme(theme)
    fig, outer = _create_quad_figure((12, 6))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = HierarchicalWaterfallPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, None)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{HierarchicalWaterfallPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_sweep_metrics(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metric_col, metric_label = ("data.perf/throughput", "Throughput (tokens/s)")
    x_bounds = _bounds_for_column(runs, "merged_df", "step")
    y_bounds = None

    for run in runs:
        df = run.merged_df
        if df is None or metric_col not in df.columns:
            continue
        series = pd.to_numeric(df[metric_col], errors="coerce")
        y_bounds = _merge_bounds(y_bounds, _finite_min_max(series))

    y_bounds = _pad_bounds(y_bounds)

    apply_theme(theme)
    fig, outer = _create_quad_figure((12, 5))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1, sharex=True)
        ax = np.ravel(axes)[0]
        df = run.merged_df
        if df is None or "step" not in df.columns:
            ax.set_title("Missing merged sweep CSV.")
            axes_list.append(axes)
            continue
        if metric_col not in df.columns:
            ax.set_title(f"{metric_label} (missing column)")
            axes_list.append(axes)
            continue
        x = pd.to_numeric(df["step"], errors="coerce")
        y = pd.to_numeric(df[metric_col], errors="coerce")
        ax.plot(x, y, linewidth=1.2, alpha=0.9, color="#34495e")
        ax.set_title(metric_label)
        ax.set_ylabel(metric_label)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=theme.grid_alpha)
        _apply_limits(ax, x_bounds, y_bounds)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{SweepMetricsPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_time_per_step(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metrics = [("data.perf/time_per_step", "Time Per Step (s)")]
    x_bounds = _bounds_for_column(runs, "merged_df", "step")
    y_bounds = {col: None for col, _ in metrics}

    for run in runs:
        df = run.merged_df
        if df is None:
            continue
        for col, _ in metrics:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            y_bounds[col] = _merge_bounds(y_bounds[col], _finite_min_max(series))

    y_bounds = {col: _pad_bounds(bounds) for col, bounds in y_bounds.items()}

    apply_theme(theme)
    fig, outer = _create_quad_figure((12, 5))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, len(metrics), 1, sharex=True)
        if run.merged_df is None or "step" not in run.merged_df.columns:
            ax = np.ravel(axes)[0]
            ax.set_title("Missing merged sweep CSV.")
            axes_list.append(axes)
            continue

        df = run.merged_df
        x = pd.to_numeric(df["step"], errors="coerce")
        ax = np.ravel(axes)[0]
        col, label = metrics[0]
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            ax.plot(x, y, linewidth=1.2, alpha=0.9, color="#34495e")
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=theme.grid_alpha)
        _apply_limits(ax, x_bounds, y_bounds.get(col))
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / "sweep_time_per_step_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_mfu_comparison(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = _bounds_for_column(runs, "merged_df", "step")
    mfu_bounds = None
    for run in runs:
        df = run.merged_df
        if df is None:
            continue
        for col in ["data.perf/mfu/actor", "data.perf/mfu/critic"]:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            mfu_bounds = _merge_bounds(mfu_bounds, _finite_min_max(series))
    mfu_bounds = _pad_bounds(mfu_bounds)

    apply_theme(theme)
    fig, outer = _create_quad_figure((10, 5))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = MFUComparisonPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, mfu_bounds)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{MFUComparisonPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_throughput_vs_length(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = _bounds_for_column(runs, "merged_df", "data.global_seqlen/mean")
    y_bounds = _bounds_for_column(runs, "merged_df", "data.perf/throughput")

    apply_theme(theme)
    fig, outer = _create_quad_figure((8, 6))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = ThroughputVsLengthPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, y_bounds)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{ThroughputVsLengthPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_throughput_reward_frontier(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = _bounds_for_column(
        runs,
        "merged_df",
        "data.val-core/openai/gsm8k/reward/mean@1",
    )
    y_bounds = _bounds_for_column(runs, "merged_df", "data.perf/throughput")

    apply_theme(theme)
    fig, outer = _create_quad_figure((8, 6))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = ThroughputRewardFrontierPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, y_bounds)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{ThroughputRewardFrontierPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_hardware_roi(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = _bounds_for_column(runs, "merged_df", "step")
    mfu_bounds = None
    reward_bounds = None
    reward_col = "data.val-core/openai/gsm8k/reward/mean@1"

    for run in runs:
        df = run.merged_df
        if df is None:
            continue
        for col in ["data.perf/mfu/actor", "data.perf/mfu/critic"]:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            mfu_bounds = _merge_bounds(mfu_bounds, _finite_min_max(series))
        if reward_col in df.columns:
            reward = pd.to_numeric(df[reward_col], errors="coerce")
            reward_bounds = _merge_bounds(reward_bounds, _finite_min_max(reward))

    mfu_bounds = _pad_bounds(mfu_bounds)
    reward_bounds = _pad_bounds(reward_bounds)

    apply_theme(theme)
    fig, outer = _create_quad_figure((10, 5))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        ax = np.ravel(axes)[0]
        twin_ax = _draw_hardware_roi(ax, run.merged_df, reward_col)
        _apply_limits(ax, x_bounds, mfu_bounds)
        if twin_ax is not None:
            _apply_limits(twin_ax, x_bounds, reward_bounds)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{HardwareROIPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_learning_price(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    x_bounds = None
    y_bounds = None
    for run in runs:
        df = run.merged_df
        if df is None:
            continue
        tokens_cum, reward = _learning_price_series(df.copy())
        x_bounds = _merge_bounds(x_bounds, _finite_min_max(tokens_cum))
        y_bounds = _merge_bounds(y_bounds, _finite_min_max(reward))

    x_bounds = _pad_bounds(x_bounds, min_floor=0.0)
    y_bounds = _pad_bounds(y_bounds)

    apply_theme(theme)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    algo_linestyles = {
        "PPO": "-",
        "ReMax": (0, (6, 4)),
        "DPO": (0, (5, 2, 1, 2)),
        "SFT": (0, (1, 6)),
    }

    algo_cmaps = {
        "PPO": "Blues",
        "ReMax": "Oranges",
        "DPO": "Greens",
        "SFT": "Purples",
    }

    algo_gpu_counts: Dict[str, List[int]] = {}
    for run in runs:
        algo_label, gpu_count = _extract_algo_and_gpu(run.run_name)
        if gpu_count is None:
            continue
        algo_gpu_counts.setdefault(algo_label, []).append(gpu_count)

    algo_color_map: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}
    for algo_label, counts in algo_gpu_counts.items():
        unique_counts = sorted(set(counts))
        if not unique_counts:
            continue
        cmap_name = algo_cmaps.get(algo_label, "Greys")
        cmap = plt.get_cmap(cmap_name)
        if len(unique_counts) == 1:
            algo_color_map[(algo_label, unique_counts[0])] = cmap(0.65)
            continue
        for idx, count in enumerate(unique_counts):
            frac = idx / (len(unique_counts) - 1)
            shade = 0.35 + 0.5 * frac
            algo_color_map[(algo_label, count)] = cmap(shade)

    unknown_color = "#7f8c8d"

    plotted = 0
    for run in runs:
        df = run.merged_df
        if df is None:
            continue
        tokens_cum, reward = _learning_price_series(df.copy())
        mask = tokens_cum.notna() & reward.notna()
        if not mask.any():
            continue
        x = tokens_cum[mask].to_numpy()
        y = reward[mask].to_numpy()
        order = np.argsort(x)
        label = get_clean_label(run.run_name)
        algo_label, gpu_count = _extract_algo_and_gpu(run.run_name)
        linestyle = algo_linestyles.get(algo_label, "-")
        color = (
            algo_color_map.get((algo_label, gpu_count))
            if gpu_count is not None
            else None
        )
        if color is None:
            color = unknown_color
        ax.plot(x[order], y[order], linewidth=1.2, alpha=0.9, label=label, linestyle=linestyle, color=color)
        plotted += 1

    if plotted == 0:
        ax.set_title("Missing valid reward/token data.")
    else:
        ax.set_title("Learning Price (All Runs)")
        ax.legend(loc="best", fontsize=QUAD_TICK_SIZE)

    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=theme.grid_alpha)
    _apply_limits(ax, x_bounds, y_bounds)
    fig.tight_layout()

    out_path = output_dir / f"{LearningPricePlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_token_bottlenecks(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    max_mean = None
    for run in runs:
        df = run.merged_df
        if df is None:
            continue
        timing_cols = [c for c in df.columns if c.startswith("data.timing_per_token_ms/")]
        if not timing_cols:
            continue
        means = [pd.to_numeric(df[col], errors="coerce").mean() for col in timing_cols]
        bounds = _finite_min_max(means)
        if bounds is not None:
            max_mean = max_mean if max_mean is not None else bounds[1]
            max_mean = max(max_mean, bounds[1])

    x_bounds = _pad_bounds((0.0, max_mean)) if max_mean is not None else None

    apply_theme(theme)
    fig, outer = _create_quad_figure((10, 6))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = TokenBottlenecksPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, None)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{TokenBottlenecksPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_bottleneck_evolution(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    max_val = None
    reward_col = "data.val-core/openai/gsm8k/reward/mean@1"
    for run in runs:
        df = run.merged_df
        if df is None or reward_col not in df.columns:
            continue
        timing_cols = [c for c in df.columns if c.startswith("data.timing_per_token_ms/")]
        if not timing_cols:
            continue
        reward = pd.to_numeric(df[reward_col], errors="coerce")
        early_mask = reward < 0.1
        mature_mask = reward > 0.4
        if early_mask.sum() == 0 or mature_mask.sum() == 0:
            continue
        values = []
        for col in timing_cols:
            early_mean = pd.to_numeric(df.loc[early_mask, col], errors="coerce").mean()
            mature_mean = pd.to_numeric(df.loc[mature_mask, col], errors="coerce").mean()
            values.extend([early_mean, mature_mean])
        bounds = _finite_min_max(values)
        if bounds is not None:
            max_val = max_val if max_val is not None else bounds[1]
            max_val = max(max_val, bounds[1])

    x_bounds = _pad_bounds((0.0, max_val)) if max_val is not None else None

    apply_theme(theme)
    fig, outer = _create_quad_figure((10, 6))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 1)
        plotter = BottleneckEvolutionPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        ax = np.ravel(axes)[0]
        _apply_limits(ax, x_bounds, None)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{BottleneckEvolutionPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_operation_aggregate(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metrics = {
        "avg_power_w": 0,
        "avg_gpu_util": 1,
        "energy_wh": 2,
    }
    x_bounds = {metric: None for metric in metrics}

    for run in runs:
        summary = _operation_summary(run.annotated_df)
        if summary is None:
            continue
        for metric in metrics:
            if metric not in summary.columns:
                continue
            series = pd.to_numeric(summary[metric], errors="coerce")
            x_bounds[metric] = _merge_bounds(x_bounds[metric], _finite_min_max(series))

    x_bounds = {metric: _pad_bounds(bounds, min_floor=0.0) for metric, bounds in x_bounds.items()}

    apply_theme(theme)
    fig, outer = _create_quad_figure((18, 7))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 1, 3, sharey=True)
        plotter = OperationAggregatePlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        for metric, col_idx in metrics.items():
            ax = np.ravel(axes)[col_idx]
            _apply_limits(ax, x_bounds.get(metric), None)
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{OperationAggregatePlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_operation_comparison(
    runs: Sequence[RunPaths],
    output_dir: Path,
    theme: ThemeConfig,
) -> Optional[Path]:
    metrics = OperationComparisonPlotter.metrics
    y_bounds = {metric: None for metric, _ in metrics}

    for run in runs:
        df = run.annotated_df
        if "operation" not in df.columns:
            continue
        df = df.copy()
        df["operation"] = df["operation"].fillna("unknown").astype(str)
        df = df[~df["operation"].isin(EXCLUDED_OPERATIONS)]
        if df.empty:
            continue
        for metric, _ in metrics:
            if metric not in df.columns:
                continue
            series = pd.to_numeric(df[metric], errors="coerce")
            y_bounds[metric] = _merge_bounds(y_bounds[metric], _finite_min_max(series))

    y_bounds = {metric: _pad_bounds(bounds) for metric, bounds in y_bounds.items()}

    apply_theme(theme)
    fig, outer = _create_quad_figure((18, 12))
    axes_list = []
    for idx, run in enumerate(runs):
        axes = _make_inner_axes(fig, outer, idx, 2, 3)
        plotter = OperationComparisonPlotter(run, output_dir, theme)
        plotter.draw(fig, axes)
        flat_axes = np.ravel(axes)
        for ax, (metric, _) in zip(flat_axes, metrics):
            _apply_limits(ax, None, y_bounds.get(metric))
        axes_list.append(axes)

    _finalize_quad(fig, axes_list)
    for run, axes in zip(runs, axes_list):
        _add_quadrant_title(fig, axes, get_clean_label(run.run_name))

    out_path = output_dir / f"{OperationComparisonPlotter.plot_name}_quad_view.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=theme.save_dpi)
    plt.close(fig)
    return out_path


def _quad_plotters() -> Mapping[str, callable]:
    return {
        GPUOverviewPlotter.plot_name: _quad_gpu_overview,
        PhaseTimelinePlotter.plot_name: _quad_phase_timeline,
        PhaseAggregatePlotter.plot_name: _quad_phase_aggregate,
        PhaseFocusRolloutPlotter.plot_name: lambda runs, out, theme: _quad_phase_focus(
            runs, out, theme, PhaseFocusRolloutPlotter, PhaseFocusRolloutPlotter.plot_name
        ),
        PhaseFocusRLPolicyPlotter.plot_name: lambda runs, out, theme: _quad_phase_focus(
            runs, out, theme, PhaseFocusRLPolicyPlotter, PhaseFocusRLPolicyPlotter.plot_name
        ),
        PhaseFocusTrainingPlotter.plot_name: lambda runs, out, theme: _quad_phase_focus(
            runs, out, theme, PhaseFocusTrainingPlotter, PhaseFocusTrainingPlotter.plot_name
        ),
        PhaseEnergyTimeStackedPlotter.plot_name: _quad_phase_energy_time_stacked,
        PhaseBoxplotPlotter.plot_name: _quad_phase_boxplots,
        PhaseCorrelationPlotter.plot_name: _quad_phase_correlations,
        SmoothedTimeSeriesPlotter.plot_name: _quad_smoothed_timeseries,
        ThermalSteadyStatePlotter.plot_name: _quad_thermal_steady_state,
        PhaseComputeDensityPlotter.plot_name: _quad_phase_compute_density,
        HierarchicalWaterfallPlotter.plot_name: _quad_hierarchical_waterfall,
        MFUComparisonPlotter.plot_name: _quad_mfu_comparison,
        ThroughputVsLengthPlotter.plot_name: _quad_throughput_vs_length,
        ThroughputRewardFrontierPlotter.plot_name: _quad_throughput_reward_frontier,
        HardwareROIPlotter.plot_name: _quad_hardware_roi,
        LearningPricePlotter.plot_name: _quad_learning_price,
        TokenBottlenecksPlotter.plot_name: _quad_token_bottlenecks,
        BottleneckEvolutionPlotter.plot_name: _quad_bottleneck_evolution,
        SweepMetricsPlotter.plot_name: _quad_sweep_metrics,
        "sweep_time_per_step": _quad_time_per_step,
        OperationAggregatePlotter.plot_name: _quad_operation_aggregate,
        OperationComparisonPlotter.plot_name: _quad_operation_comparison,
    }


def generate_quad_views(
    runs: Sequence[RunPaths],
    output_dir: Path,
    plotter_classes: Optional[Mapping[str, type]] = None,
    theme: Optional[ThemeConfig] = None,
) -> List[Path]:
    """Generate quad-view comparisons for the first four runs."""
    run_list = list(runs)
    if len(run_list) < 4:
        print("Quad views skipped: need at least 4 runs.")
        return []

    run_list = run_list[:4]
    out_dir = Path(output_dir) / "quad_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    theme = theme or ThemeConfig()

    available_plot_names = None
    if plotter_classes:
        available_plot_names = {cls.plot_name for cls in plotter_classes.values()}

    outputs: List[Path] = []
    for plot_name, handler in _quad_plotters().items():
        if available_plot_names is not None and plot_name not in available_plot_names:
            continue
        try:
            out_path = handler(run_list, out_dir, theme)
        except Exception as exc:  # pragma: no cover
            print(f"Quad view failed for {plot_name}: {exc}")
            continue
        if out_path is not None:
            outputs.append(out_path)

    return outputs
