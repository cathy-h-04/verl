#!/usr/bin/env python3
"""Modular graphing suite for cleaned Verl monitoring data.

Design goals:
1) A shared, consistent theme across plots.
2) A base class with clear hooks so subclasses can make small layout tweaks
   (e.g., legend placement, spacing, annotations) without copying logic.
3) Multiple concrete plot types that operate on the cleaned artifacts:
   - annotated_*.csv (GPU telemetry with phase/operation labels)
   - merged_sweep_*.csv (stepwise training + validation metrics)

This script intentionally focuses on maintainable structure over cleverness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# NOTE: These imports are expected to exist in the user's environment.
# The sandbox may not have them installed, so runtime verification might not
# be possible here.
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional
    sns = None


CLEANED_ROOT = Path("monitoring_small_cleaned")

# Centralized color choices so they can be reused across plot types.
PHASE_COLORS: Dict[str, str] = {
    "rollout": "#2ecc71",
    "rl_policy": "#f39c12",
    "training": "#3498db",
    "idle": "#95a5a6",
    "unknown": "#7f8c8d",
}

OPERATION_COLORS: Dict[str, str] = {
    "generate_sequences": "#1abc9c",
    "gen": "#2ecc71",
    "gen_max": "#27ae60",
    "reward": "#f39c12",
    "old_log_prob": "#e67e22",
    "Role.RefPolicy": "#d35400",
    "values": "#16a085",
    "adv": "#9b59b6",
    "update_critic": "#3498db",
    "update_actor": "#2980b9",
    "start_profile": "#7f8c8d",
    "generation_timing/max": "#34495e",
    "generation_timing/min": "#2c3e50",
    "generation_timing/topk_ratio": "#22313f",
    "unknown": "#95a5a6",
}

# Operations that should not be treated as real work for aggregate timing/energy.
EXCLUDED_OPERATIONS = {
    "start_profile",
    "generation_timing/max",
    "generation_timing/min",
    "generation_timing/topk_ratio",
    "step",
}

PHASE_OPERATION_ORDER: Dict[str, List[str]] = {
    "rollout": [
        "start_profile",
        "generate_sequences",
        "generation_timing/max",
        "generation_timing/min",
        "generation_timing/topk_ratio",
        "gen",
        "gen_max",
    ],
    "rl_policy": [
        "reward",
        "old_log_prob",
        "Role.RefPolicy",
        "values",
        "adv",
    ],
    "training": [
        "update_critic",
        "update_actor",
    ],
}

# -----------------------------
# Theme + shared configuration
# -----------------------------

@dataclass(frozen=True)
class ThemeConfig:
    """Shared plotting style configuration.

    Subclasses and callers can swap this out, but all plots should default
    to the same visual language.
    """

    style: str = "seaborn-v0_8-paper"
    context: str = "paper"
    palette: str = "deep"
    figure_dpi: int = 150
    save_dpi: int = 300
    font_size: int = 10
    axes_label_size: int = 11
    axes_title_size: int = 12
    legend_font_size: int = 9
    grid_alpha: float = 0.25
    rc_params: Mapping[str, object] = field(
        default_factory=lambda: {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )


def apply_theme(theme: ThemeConfig) -> None:
    """Apply a consistent plotting theme.

    This is intentionally idempotent and cheap so plotters can call it
    without coordinating global state.
    """

    # Matplotlib base style
    try:
        plt.style.use(theme.style)
    except OSError:
        # Fall back gracefully if the style isn't available.
        plt.style.use("default")

    # Seaborn styling when available
    if sns is not None:
        sns.set_theme(context=theme.context, style="whitegrid", palette=theme.palette)

    plt.rcParams.update(dict(theme.rc_params))


def format_title(run_name: str, title: str) -> Tuple[str, str]:
    """Return (suptitle, title) with experiment name above metric title."""
    return run_name, title


# -----------------------------
# Path discovery + data loading
# -----------------------------

@dataclass(frozen=True)
class RunPaths:
    """Resolved data artifacts for a single run folder."""

    run_dir: Path
    annotated_csv: Path
    merged_sweep_csv: Optional[Path]
    cleaned_phase_timings: Optional[Path]

    @property
    def run_name(self) -> str:
        return self.run_dir.name



def _first_match(run_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None



def resolve_run_paths(run_dir: Path) -> Optional[RunPaths]:
    """Resolve the canonical cleaned artifacts for a run directory.

    Returns None when required artifacts are missing.
    """

    annotated = _first_match(run_dir, "annotated_*_phased_*.csv")
    if annotated is None:
        return None

    merged = _first_match(run_dir, "merged_sweep_*.csv")
    cleaned_timings = _first_match(run_dir, "cleaned_phase_timings_*.jsonl")

    return RunPaths(
        run_dir=run_dir,
        annotated_csv=annotated,
        merged_sweep_csv=merged,
        cleaned_phase_timings=cleaned_timings,
    )



def discover_runs(root: Path) -> List[RunPaths]:
    """Discover cleaned run folders under the given root."""

    runs: List[RunPaths] = []
    if resolve_run_paths(root) is not None:
        resolved = resolve_run_paths(root)
        assert resolved is not None
        runs.append(resolved)
        return runs

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        resolved = resolve_run_paths(run_dir)
        if resolved is not None:
            runs.append(resolved)
    return runs



def load_annotated_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp_aligned_unix" in df.columns:
        df["timestamp_aligned_unix"] = pd.to_numeric(df["timestamp_aligned_unix"], errors="coerce")
    df["elapsed_seconds"] = pd.to_numeric(df.get("elapsed_seconds"), errors="coerce")
    return df



def load_merged_sweep_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
    return df


def load_cleaned_phase_timings_df(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    records: List[Dict[str, object]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        return None
    return pd.DataFrame(records)


# -----------------------------
# Derived metrics + utilities
# -----------------------------

GPU_METRIC_COLUMNS = [
    "gpu_util_percent",
    "power_draw_w",
    "temperature_c",
    "memory_used_mb",
    "memory_total_mb",
    "memory_util_percent",
    "sm_clock_mhz",
    "mem_clock_mhz",
]


def add_gpu_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add consistent derived GPU metrics used across plotters."""

    out = df.copy()

    # Memory conversions
    out["memory_used_gb"] = pd.to_numeric(out.get("memory_used_mb"), errors="coerce") / 1024.0
    out["memory_total_gb"] = pd.to_numeric(out.get("memory_total_mb"), errors="coerce") / 1024.0
    out["memory_free_gb"] = out["memory_total_gb"] - out["memory_used_gb"]
    out["memory_usage_ratio"] = out["memory_used_gb"] / out["memory_total_gb"].replace(0, np.nan)

    # Time conversions
    out["elapsed_minutes"] = pd.to_numeric(out.get("elapsed_seconds"), errors="coerce") / 60.0

    # Ratios and efficiencies
    power_draw = pd.to_numeric(out.get("power_draw_w"), errors="coerce")
    power_limit = pd.to_numeric(out.get("power_limit_w"), errors="coerce")
    gpu_util = pd.to_numeric(out.get("gpu_util_percent"), errors="coerce")
    temp_c = pd.to_numeric(out.get("temperature_c"), errors="coerce")

    out["power_to_limit_ratio"] = power_draw / power_limit.replace(0, np.nan)
    out["power_efficiency"] = gpu_util / (power_draw + 1e-6)
    out["thermal_efficiency"] = gpu_util / (temp_c + 1e-6)
    out["power_draw_kw"] = power_draw / 1000.0
    out["temperature_k"] = temp_c + 273.15

    return out



def compute_sample_durations_seconds(df: pd.DataFrame) -> pd.Series:
    """Estimate per-row sample duration from aligned timestamps.

    We use forward differences and fill missing values with the median
    non-null duration. This supports rough energy estimates.
    """

    if "timestamp_aligned_unix" not in df.columns:
        # Fall back to 1-second samples if alignment is missing.
        return pd.Series(np.ones(len(df)), index=df.index, dtype=float)

    ts = pd.to_numeric(df["timestamp_aligned_unix"], errors="coerce")
    dt = ts.diff().shift(-1)

    median_dt = dt[dt > 0].median()
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1.0

    dt = dt.where(dt > 0, median_dt)
    dt = dt.fillna(median_dt)
    return dt.astype(float)



def compute_phase_windows_from_annotated(df: pd.DataFrame) -> List[Tuple[str, float, float]]:
    """Compute contiguous phase windows from annotated telemetry rows.

    Returns a list of (phase_name, start_ts, end_ts).
    """

    required_cols = {"phase_name", "timestamp_aligned_unix"}
    if not required_cols.issubset(df.columns):
        return []

    work = df[["phase_name", "timestamp_aligned_unix"]].copy()
    work["timestamp_aligned_unix"] = pd.to_numeric(work["timestamp_aligned_unix"], errors="coerce")
    work = work.dropna(subset=["timestamp_aligned_unix"]).sort_values("timestamp_aligned_unix")

    windows: List[Tuple[str, float, float]] = []
    if work.empty:
        return windows

    current_phase = str(work.iloc[0]["phase_name"])
    start_ts = float(work.iloc[0]["timestamp_aligned_unix"])
    prev_ts = start_ts

    for _, row in work.iloc[1:].iterrows():
        phase = str(row["phase_name"])
        ts = float(row["timestamp_aligned_unix"])
        if phase != current_phase:
            windows.append((current_phase, start_ts, prev_ts))
            current_phase = phase
            start_ts = ts
        prev_ts = ts

    windows.append((current_phase, start_ts, prev_ts))
    return windows


# -----------------------------
# Base plotter with extensible hooks
# -----------------------------

class BasePlotter:
    """Base plotter with shared data loading, theme, and save logic.

    Subclass hooks intended for small positioning/annotation adjustments:
    - create_figure(...)
    - draw(...)
    - adjust_layout(...)
    - annotate(...)
    - finalize(...)
    """

    plot_name: str = "base"
    plot_title: str = "Plot"

    def __init__(
        self,
        run_paths: RunPaths,
        output_dir: Path,
        theme: ThemeConfig | None = None,
    ) -> None:
        self.run_paths = run_paths
        self.output_dir = output_dir
        self.theme = theme or ThemeConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data once per plotter instance.
        self.annotated_df = add_gpu_derived_columns(load_annotated_csv(run_paths.annotated_csv))
        self.merged_df = load_merged_sweep_csv(run_paths.merged_sweep_csv)
        self.timings_df = load_cleaned_phase_timings_df(run_paths.cleaned_phase_timings)

    # ---- hooks for subclasses ----
    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def adjust_layout(self, fig: plt.Figure, axes: np.ndarray) -> None:
        """Default layout adjustment; subclasses can override."""
        fig.tight_layout()

    def annotate(self, fig: plt.Figure, axes: np.ndarray) -> None:
        """Optional annotations; subclasses can override."""
        return None

    def finalize(self, fig: plt.Figure, axes: np.ndarray) -> None:
        """Last chance adjustments before save; subclasses can override."""
        return None

    # ---- orchestration ----
    def output_path(self) -> Path:
        filename = f"{self.plot_name}_{self.run_paths.run_name}.png"
        return self.output_dir / filename

    def render(self) -> Path:
        apply_theme(self.theme)
        fig, axes = self.create_figure()
        self.draw(fig, axes)
        self.annotate(fig, axes)
        self.adjust_layout(fig, axes)
        self.finalize(fig, axes)

        out_path = self.output_path()
        fig.savefig(out_path, bbox_inches="tight", dpi=self.theme.save_dpi)
        plt.close(fig)
        return out_path


# -----------------------------
# Concrete plotters
# -----------------------------

class GPUOverviewPlotter(BasePlotter):
    """Overview grid for key GPU telemetry metrics."""

    plot_name = "gpu_overview"
    plot_title = "GPU Overview"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        suptitle, title = format_title(self.run_paths.run_name, "GPU Overview")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df
        time_min = df["elapsed_minutes"]

        # 1. GPU utilization
        ax = axes[0, 0]
        ax.plot(time_min, df["gpu_util_percent"], linewidth=1.1, alpha=0.9, color="#2ecc71")
        ax.fill_between(time_min, 0, df["gpu_util_percent"], alpha=0.25, color="#2ecc71")
        mean_util = df["gpu_util_percent"].mean()
        ax.axhline(mean_util, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_util:.1f}%")
        ax.set_title("GPU Compute Utilization")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Utilization (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")

        # 2. Memory usage
        ax = axes[0, 1]
        ax.plot(time_min, df["memory_used_gb"], linewidth=1.1, alpha=0.9, color="#3498db")
        total_gb = df["memory_total_gb"].dropna().iloc[0] if df["memory_total_gb"].notna().any() else np.nan
        if np.isfinite(total_gb):
            ax.axhline(total_gb, color="red", linestyle="--", alpha=0.6, label=f"Total: {total_gb:.1f} GB")
            ax.legend(loc="upper right")
        ax.set_title("GPU Memory Usage")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Memory Used (GB)")
        ax.grid(True, alpha=self.theme.grid_alpha)

        # 3. Power draw
        ax = axes[0, 2]
        ax.plot(time_min, df["power_draw_w"], linewidth=1.1, alpha=0.9, color="#e74c3c")
        power_limit = pd.to_numeric(df.get("power_limit_w"), errors="coerce").dropna()
        if not power_limit.empty:
            ax.axhline(power_limit.iloc[0], color="orange", linestyle="--", alpha=0.6, label=f"Limit: {power_limit.iloc[0]:.0f} W")
        mean_power = pd.to_numeric(df.get("power_draw_w"), errors="coerce").mean()
        if np.isfinite(mean_power):
            ax.axhline(mean_power, color="darkred", linestyle="--", alpha=0.7, label=f"Mean: {mean_power:.1f} W")
        ax.set_title("GPU Power Consumption")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Power (W)")
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")

        # 4. Temperature
        ax = axes[1, 0]
        ax.plot(time_min, df["temperature_c"], linewidth=1.1, alpha=0.9, color="#f39c12")
        mean_temp = pd.to_numeric(df.get("temperature_c"), errors="coerce").mean()
        if np.isfinite(mean_temp):
            ax.axhline(mean_temp, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_temp:.1f}°C")
            ax.legend(loc="upper right")
        ax.set_title("GPU Temperature")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=self.theme.grid_alpha)

        # 5. SM clock
        ax = axes[1, 1]
        ax.plot(time_min, df["sm_clock_mhz"], linewidth=1.1, alpha=0.9, color="#9b59b6")
        ax.set_title("SM Clock Speed")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Clock (MHz)")
        ax.grid(True, alpha=self.theme.grid_alpha)

        # 6. Memory bandwidth utilization
        ax = axes[1, 2]
        ax.plot(time_min, df["memory_util_percent"], linewidth=1.1, alpha=0.9, color="#1abc9c")
        ax.set_title("Memory Bandwidth Utilization")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Memory Util (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=self.theme.grid_alpha)


class PhaseTimelinePlotter(BasePlotter):
    """Time series with phase shading, using annotated telemetry."""

    plot_name = "phase_timeline"
    plot_title = "Phase Timeline"

    metric_columns: Sequence[Tuple[str, str]] = (
        ("gpu_util_percent", "GPU Utilization (%)"),
        ("power_draw_w", "Power Draw (W)"),
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        n = len(self.metric_columns)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), sharex=True)
        if n == 1:
            axes = np.asarray([axes])
        suptitle, title = format_title(self.run_paths.run_name, "Phase Timeline")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df

        if "timestamp_aligned_unix" not in df.columns:
            raise ValueError("Annotated CSV is missing timestamp_aligned_unix; cannot build phase timeline.")

        x = pd.to_numeric(df["timestamp_aligned_unix"], errors="coerce")
        x0 = x.min()
        x_rel_min = (x - x0) / 60.0

        # Phase windows for background shading (only if there is useful variation)
        phase_windows = compute_phase_windows_from_annotated(df)
        unique_phases = {phase for phase, _, _ in phase_windows}
        phase_windows_rel = []
        if len(unique_phases) > 1:
            phase_windows_rel = [
                (phase, (start - x0) / 60.0, (end - x0) / 60.0)
                for phase, start, end in phase_windows
            ]

        for ax, (metric, ylabel) in zip(axes, self.metric_columns):
            if metric not in df.columns:
                ax.set_title(f"{ylabel} (missing column: {metric})")
                continue

            y = pd.to_numeric(df[metric], errors="coerce")
            ax.plot(x_rel_min, y, linewidth=1.0, color="#2c3e50", alpha=0.9)
            ax.set_title(ylabel)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(True, alpha=self.theme.grid_alpha)

            # Shade phases behind the metric
            for phase, start_min, end_min in phase_windows_rel:
                color = PHASE_COLORS.get(phase, PHASE_COLORS["unknown"])
                ax.axvspan(start_min, end_min, color=color, alpha=0.08)

        axes[-1].set_xlabel("Time Since Start (minutes)")

    def annotate(self, fig: plt.Figure, axes: np.ndarray) -> None:
        """Add a compact phase legend that subclasses can reposition."""
        phase_windows = compute_phase_windows_from_annotated(self.annotated_df)
        unique_phases = {phase for phase, _, _ in phase_windows}
        if len(unique_phases) <= 1:
            return

        handles = []
        labels = []
        for phase, color in PHASE_COLORS.items():
            if phase in {"idle", "unknown"}:
                continue
            if phase not in unique_phases:
                continue
            handles.append(plt.Line2D([0], [0], color=color, lw=6, alpha=0.35))
            labels.append(phase)
        if handles:
            axes[0].legend(handles, labels, loc="upper right", title="Phase")


class SweepMetricsPlotter(BasePlotter):
    """Stepwise training + validation metrics from merged sweep CSV."""

    plot_name = "sweep_metrics"
    plot_title = "Sweep Metrics"

    # (column, label)
    sweep_metrics: Sequence[Tuple[str, str]] = (
        ("data.val-core/openai/gsm8k/reward/mean@1", "Reward"),
        ("data.perf/throughput", "Throughput (tokens/s)"),
        ("data.perf/time_per_step", "Time Per Step (s)"),
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(len(self.sweep_metrics), 1, figsize=(12, 4.0 * len(self.sweep_metrics)), sharex=True)
        if len(self.sweep_metrics) == 1:
            axes = np.asarray([axes])
        suptitle, title = format_title(self.run_paths.run_name, "Sweep Metrics")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        if self.merged_df is None:
            for ax in np.ravel(axes):
                ax.set_title("Merged sweep CSV not found; cannot plot sweep metrics.")
            return

        df = self.merged_df.copy()
        if "step" not in df.columns:
            for ax in np.ravel(axes):
                ax.set_title("Merged sweep CSV is missing 'step'.")
            return

        x = pd.to_numeric(df["step"], errors="coerce")

        for ax, (col, label) in zip(np.ravel(axes), self.sweep_metrics):
            if col not in df.columns:
                ax.set_title(f"{label} (missing column: {col})")
                ax.grid(True, alpha=self.theme.grid_alpha)
                continue

            y = pd.to_numeric(df[col], errors="coerce")
            ax.plot(x, y, linewidth=1.2, alpha=0.9, color="#34495e")
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.grid(True, alpha=self.theme.grid_alpha)

        np.ravel(axes)[-1].set_xlabel("Step")


class OperationAggregatePlotter(BasePlotter):
    """Aggregate GPU behavior by operation using annotated telemetry."""

    plot_name = "operation_aggregate"
    plot_title = "Operation Aggregate"

    metrics: Sequence[Tuple[str, str]] = (
        ("avg_power_w", "Average Power (W)"),
        ("avg_gpu_util", "Average GPU Util (%)"),
        ("energy_wh", "Energy (Wh, approx)"),
    )

    def _build_operation_summary(self) -> pd.DataFrame:
        df = self.annotated_df.copy()
        if "operation" not in df.columns:
            raise ValueError("Annotated CSV is missing 'operation'.")

        df["operation"] = df["operation"].fillna("unknown").astype(str)
        df = df[~df["operation"].isin(EXCLUDED_OPERATIONS)]

        dt = compute_sample_durations_seconds(df)
        power = pd.to_numeric(df.get("power_draw_w"), errors="coerce")
        gpu_util = pd.to_numeric(df.get("gpu_util_percent"), errors="coerce")

        df["sample_dt_s"] = dt
        df["sample_energy_j"] = power * dt

        grouped = df.groupby("operation", dropna=False)
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


class OperationComparisonPlotter(BasePlotter):
    """Boxplot comparison of GPU metrics by operation (annotated CSV)."""

    plot_name = "operation_comparison"
    plot_title = "Operation Metric Comparison"

    metrics: Sequence[Tuple[str, str]] = (
        ("power_draw_w", "Power (W)"),
        ("gpu_util_percent", "GPU Util (%)"),
        ("temperature_c", "Temperature (°C)"),
        ("memory_used_gb", "Memory Used (GB)"),
        ("power_to_limit_ratio", "Power / Limit"),
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        suptitle, title = format_title(self.run_paths.run_name, "Operation Metric Comparison")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df.copy()
        if "operation" not in df.columns:
            raise ValueError("Annotated CSV is missing 'operation'.")

        df["operation"] = df["operation"].fillna("unknown").astype(str)
        df = df[~df["operation"].isin(EXCLUDED_OPERATIONS)]

        if df.empty:
            for ax in np.ravel(axes):
                ax.set_title("No operation data after filtering.")
            return

        unique_ops = df["operation"].unique()
        palette = [OPERATION_COLORS.get(op, OPERATION_COLORS["unknown"]) for op in unique_ops]

        flat_axes = np.ravel(axes)
        for idx, (metric, label) in enumerate(self.metrics):
            if idx >= len(flat_axes):
                break
            ax = flat_axes[idx]
            if metric not in df.columns:
                ax.set_title(f"{label} (missing column: {metric})")
                continue
            if sns is None:
                ax.set_title("Seaborn not available for boxplots.")
                continue

            sns.boxplot(
                data=df,
                x="operation",
                y=metric,
                hue="operation",
                legend=False,
                ax=ax,
                palette=palette,
            )
            ax.set_xlabel("")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, axis="y", alpha=self.theme.grid_alpha)
            ax.tick_params(axis="x", labelrotation=45)
            for label_text in ax.get_xticklabels():
                label_text.set_ha("right")
                label_text.set_fontsize(8)

        # Hide any unused axes
        for ax in flat_axes[len(self.metrics):]:
            ax.set_visible(False)


class SmoothedTimeSeriesPlotter(BasePlotter):
    """Smoothed time series for key GPU metrics."""

    plot_name = "smoothed_timeseries"
    plot_title = "Smoothed Time Series"

    metrics: Sequence[Tuple[str, str]] = (
        ("gpu_util_percent", "GPU Utilization (%)"),
        ("power_draw_w", "Power Draw (W)"),
        ("temperature_c", "Temperature (°C)"),
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(14, 4.0 * len(self.metrics)), sharex=True)
        if len(self.metrics) == 1:
            axes = np.asarray([axes])
        suptitle, title = format_title(self.run_paths.run_name, "Smoothed Time Series")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df
        time_min = df["elapsed_minutes"]
        window = max(5, int(len(df) * 0.01))

        for ax, (metric, label) in zip(np.ravel(axes), self.metrics):
            if metric not in df.columns:
                ax.set_title(f"{label} (missing column: {metric})")
                continue
            series = pd.to_numeric(df[metric], errors="coerce")
            smooth = series.rolling(window=window, min_periods=max(1, window // 3)).mean()
            ax.plot(time_min, smooth, linewidth=1.2, color="#34495e")
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.grid(True, alpha=self.theme.grid_alpha)

        np.ravel(axes)[-1].set_xlabel("Time (minutes)")


class PhaseAggregatePlotter(BasePlotter):
    """Aggregate GPU metrics per phase, similar to old phase comparison plots."""

    plot_name = "phase_aggregate"
    plot_title = "Phase Aggregates"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        suptitle, title = format_title(self.run_paths.run_name, "Phase Aggregates")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df.copy()
        if "phase_name" not in df.columns:
            raise ValueError("Annotated CSV is missing 'phase_name'.")

        df = df[df["phase_name"] != "idle"]
        if df.empty:
            for ax in np.ravel(axes):
                ax.set_title("No non-idle phase data.")
            return

        dt = compute_sample_durations_seconds(df)
        df["sample_dt_s"] = dt
        df["sample_energy_j"] = pd.to_numeric(df.get("power_draw_w"), errors="coerce") * dt

        grouped = df.groupby("phase_name", dropna=False)
        summary = grouped.agg(
            avg_power_w=("power_draw_w", "mean"),
            avg_gpu_util=("gpu_util_percent", "mean"),
            avg_temp_c=("temperature_c", "mean"),
            total_energy_j=("sample_energy_j", "sum"),
        ).reset_index()
        summary["energy_wh"] = summary["total_energy_j"] / 3600.0
        summary["phase_name"] = summary["phase_name"].fillna("unknown").astype(str)

        phases = summary["phase_name"].tolist()
        colors = [PHASE_COLORS.get(p, PHASE_COLORS["unknown"]) for p in phases]

        metrics = [
            ("avg_gpu_util", "Average GPU Util (%)", axes[0, 0]),
            ("avg_power_w", "Average Power (W)", axes[0, 1]),
            ("avg_temp_c", "Average Temp (°C)", axes[1, 0]),
            ("energy_wh", "Total Energy (Wh)", axes[1, 1]),
        ]

        for col, label, ax in metrics:
            values = pd.to_numeric(summary[col], errors="coerce")
            ax.bar(phases, values, color=colors, alpha=0.9)
            ax.set_title(label)
            ax.grid(True, axis="y", alpha=self.theme.grid_alpha)
            ax.tick_params(axis="x", labelrotation=20)
            for label_text in ax.get_xticklabels():
                label_text.set_ha("right")


class PhaseFocusMetricsPlotter(BasePlotter):
    """Per-iteration plots grouped by phase (3 panels: util, power, temp)."""

    plot_name = "phase_focus"
    plot_title = "Iteration Focus"
    focus_phase: str = "rollout"
    focus_window: Tuple[int, int] = (225, 235)
    use_time_axis: bool = True

    metrics: Sequence[Tuple[str, str]] = (
        ("gpu_util_percent", "GPU Utilization (%)"),
        ("power_draw_w", "Power Draw (W)"),
        ("temperature_c", "Temperature (°C)"),
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(14, 12), sharex=True)
        suptitle, title = format_title(self.run_paths.run_name, f"Iteration Focus: {self.focus_phase}")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray(axes)

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df.copy()
        if "iteration" not in df.columns or "phase_name" not in df.columns:
            raise ValueError("Annotated CSV is missing 'iteration' or 'phase_name'.")

        df = df[df["phase_name"] != "idle"]
        df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
        df = df.dropna(subset=["iteration"])

        if self.focus_window:
            start_iter, end_iter = self.focus_window
            df = df[(df["iteration"] >= start_iter) & (df["iteration"] <= end_iter)]

        if df.empty:
            for ax in np.ravel(axes):
                ax.set_title("No non-idle phase data.")
            return

        phases = ["rollout", "rl_policy", "training"]

        if self.use_time_axis and "timestamp_aligned_unix" in df.columns:
            time_raw = pd.to_numeric(df["timestamp_aligned_unix"], errors="coerce")
            time_ref = time_raw.min()
            df["focus_time_min"] = (time_raw - time_ref) / 60.0
            x_col = "focus_time_min"
            x_label = "Time (minutes)"
        elif self.use_time_axis and "elapsed_seconds" in df.columns:
            time_raw = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
            time_ref = time_raw.min()
            df["focus_time_min"] = (time_raw - time_ref) / 60.0
            x_col = "focus_time_min"
            x_label = "Time (minutes)"
        else:
            x_col = "iteration"
            x_label = "Iteration"
        for ax, (metric, ylabel) in zip(np.ravel(axes), self.metrics):
            if metric not in df.columns:
                ax.set_title(f"{ylabel} (missing column: {metric})")
                continue

            df_metric = df.copy()
            df_metric[metric] = pd.to_numeric(df_metric[metric], errors="coerce")
            df_metric = df_metric.dropna(subset=[metric])

            for phase in phases:
                phase_df = df_metric[df_metric["phase_name"] == phase]
                if phase_df.empty:
                    continue
                color = PHASE_COLORS.get(phase, PHASE_COLORS["unknown"])
                alpha = 0.95 if phase == self.focus_phase else 0.5
                linewidth = 2.2 if phase == self.focus_phase else 1.0
                marker = "o"
                markersize = 5 if phase == self.focus_phase else 4
                ax.plot(
                    phase_df[x_col],
                    phase_df[metric],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                    label=phase if phase == self.focus_phase else None,
                )

            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=self.theme.grid_alpha)

        np.ravel(axes)[-1].set_xlabel(x_label)


class PhaseFocusRolloutPlotter(PhaseFocusMetricsPlotter):
    plot_name = "phase_focus_rollout"
    focus_phase = "rollout"


class PhaseFocusRLPolicyPlotter(PhaseFocusMetricsPlotter):
    plot_name = "phase_focus_rl_policy"
    focus_phase = "rl_policy"


class PhaseFocusTrainingPlotter(PhaseFocusMetricsPlotter):
    plot_name = "phase_focus_training"
    focus_phase = "training"


class PhaseEnergyTimeStackedPlotter(BasePlotter):
    """Energy and time distribution across phases (100% stacked bars)."""

    plot_name = "phase_energy_time_stacked"
    plot_title = "Energy and Time Distribution"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        suptitle, title = format_title(self.run_paths.run_name, "Energy and Time Distribution")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df.copy()
        if "phase_name" not in df.columns:
            raise ValueError("Annotated CSV is missing 'phase_name'.")

        df = df[df["phase_name"] != "idle"]
        if df.empty:
            for ax in np.ravel(axes):
                ax.set_title("No non-idle phase data.")
            return

        dt = compute_sample_durations_seconds(df)
        df["sample_dt_s"] = dt
        df["sample_energy_j"] = pd.to_numeric(df.get("power_draw_w"), errors="coerce") * dt

        grouped = df.groupby("phase_name", dropna=False).agg(
            total_duration_s=("sample_dt_s", "sum"),
            total_energy_j=("sample_energy_j", "sum"),
        )
        grouped = grouped.reset_index()
        grouped["phase_name"] = grouped["phase_name"].fillna("unknown").astype(str)

        phases = grouped["phase_name"].tolist()
        colors = [PHASE_COLORS.get(p, PHASE_COLORS["unknown"]) for p in phases]

        energy_wh = (grouped["total_energy_j"] / 3600.0).to_numpy()
        time_s = grouped["total_duration_s"].to_numpy()

        energy_frac = energy_wh / energy_wh.sum() if energy_wh.sum() > 0 else np.zeros_like(energy_wh)
        time_frac = time_s / time_s.sum() if time_s.sum() > 0 else np.zeros_like(time_s)

        x_labels = ["Total Time", "Total Energy"]
        x = np.arange(len(x_labels))

        bottoms = np.zeros_like(x, dtype=float)
        for idx, phase in enumerate(phases):
            frac_values = np.array([time_frac[idx], energy_frac[idx]])
            axes[0].bar(x, frac_values, bottom=bottoms, color=colors[idx], label=phase)
            for bar_idx, frac in enumerate(frac_values):
                if frac <= 0:
                    continue
                y = bottoms[bar_idx] + frac / 2.0
                if frac >= 0.02:
                    axes[0].text(
                        x[bar_idx],
                        y,
                        f"{frac:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        fontweight="bold",
                    )
            bottoms += frac_values

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Fraction of Total")
        axes[0].grid(True, axis="y", alpha=self.theme.grid_alpha)
        axes[0].legend(loc="upper right", title="Phase")


class PhaseBoxplotPlotter(BasePlotter):
    """Phase-level metric comparison (box plots) across 5 metrics."""

    plot_name = "phase_boxplots"
    plot_title = "Phase Metric Comparison"

    metrics: Sequence[Tuple[str, str]] = (
        ("gpu_util_percent", "GPU Util (%)"),
        ("power_draw_w", "Power (W)"),
        ("temperature_c", "Temp (°C)"),
        ("memory_used_gb", "Mem Used (GB)"),
        ("power_to_limit_ratio", "Power / Limit"),
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        suptitle, title = format_title(self.run_paths.run_name, "Phase Metric Comparison")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        if sns is None:
            for ax in np.ravel(axes):
                ax.set_title("Seaborn not available for boxplots.")
            return

        df = self.annotated_df.copy()
        if "phase_name" not in df.columns:
            raise ValueError("Annotated CSV is missing 'phase_name'.")

        df = df[df["phase_name"] != "idle"]
        if df.empty:
            for ax in np.ravel(axes):
                ax.set_title("No non-idle phase data.")
            return

        phases = df["phase_name"].unique()
        palette = [PHASE_COLORS.get(p, PHASE_COLORS["unknown"]) for p in phases]

        flat_axes = np.ravel(axes)
        for idx, (metric, label) in enumerate(self.metrics):
            if idx >= len(flat_axes):
                break
            ax = flat_axes[idx]
            if metric not in df.columns:
                ax.set_title(f"{label} (missing column: {metric})")
                continue

            sns.boxplot(
                data=df,
                x="phase_name",
                y=metric,
                hue="phase_name",
                legend=False,
                ax=ax,
                palette=palette,
            )
            ax.set_xlabel("")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.tick_params(axis="x", labelrotation=15)
            for label_text in ax.get_xticklabels():
                label_text.set_ha("right")
            ax.grid(True, axis="y", alpha=self.theme.grid_alpha)

        for ax in flat_axes[len(self.metrics):]:
            ax.set_visible(False)


class ThermalSteadyStatePlotter(BasePlotter):
    """Temperature over time for the full run (thermal stability check)."""

    plot_name = "thermal_steady_state"
    plot_title = "Thermal Steady-State"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        suptitle, title = format_title(self.run_paths.run_name, "Thermal Steady-State")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df
        ax = axes[0]
        if "temperature_c" not in df.columns:
            ax.set_title("Missing temperature_c")
            return
        ax.plot(df["elapsed_minutes"], pd.to_numeric(df["temperature_c"], errors="coerce"), color="#f39c12", linewidth=1.2)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Thermal Steady-State")
        ax.grid(True, alpha=self.theme.grid_alpha)


class PhaseComputeDensityPlotter(BasePlotter):
    """Utilization vs power draw, colored by phase."""

    plot_name = "phase_compute_density"
    plot_title = "Phase Compute Density"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        suptitle, title = format_title(self.run_paths.run_name, "Phase Compute Density")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df.copy()
        df = df[df["phase_name"] != "idle"]
        ax = axes[0]
        if df.empty:
            ax.set_title("No non-idle phase data.")
            return
        if sns is not None:
            sns.scatterplot(
                data=df,
                x="gpu_util_percent",
                y="power_draw_w",
                hue="phase_name",
                palette=PHASE_COLORS,
                alpha=0.6,
                s=12,
                ax=ax,
                legend=True,
            )
        else:
            for phase in df["phase_name"].unique():
                phase_df = df[df["phase_name"] == phase]
                ax.scatter(
                    phase_df["gpu_util_percent"],
                    phase_df["power_draw_w"],
                    s=12,
                    alpha=0.6,
                    color=PHASE_COLORS.get(phase, PHASE_COLORS["unknown"]),
                    label=phase,
                )
            ax.legend(loc="upper right")
        # Centroid + 1-sigma ellipse per phase
        for phase in sorted(df["phase_name"].unique()):
            phase_df = df[df["phase_name"] == phase]
            x = pd.to_numeric(phase_df["gpu_util_percent"], errors="coerce")
            y = pd.to_numeric(phase_df["power_draw_w"], errors="coerce")
            mask = x.notna() & y.notna()
            x = x[mask]
            y = y[mask]
            if len(x) < 2:
                continue
            mean_x = x.mean()
            mean_y = y.mean()
            cov = np.cov(x, y)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            width, height = 2 * np.sqrt(vals)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            color = PHASE_COLORS.get(phase, PHASE_COLORS["unknown"])
            ellipse = Ellipse(
                (mean_x, mean_y),
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                facecolor="none",
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_patch(ellipse)
            ax.scatter([mean_x], [mean_y], color=color, s=60, marker="X", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("GPU Utilization (%)")
        ax.set_ylabel("Power Draw (W)")
        ax.set_title("Phase Compute Density")
        ax.grid(True, alpha=self.theme.grid_alpha)


class OrchestrationTaxPlotter(BasePlotter):
    """Gap analysis between phase timing records (orchestration tax)."""

    plot_name = "orchestration_tax"
    plot_title = "Orchestration Tax (Gaps)"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        suptitle, title = format_title(self.run_paths.run_name, "Orchestration Tax (Gaps)")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.timings_df is None or self.timings_df.empty:
            ax.set_title("Missing cleaned_phase_timings data.")
            return

        df = self.timings_df.copy()
        df = df.sort_values(["iteration", "timestamp"])

        op_cols = [c for c in df.columns if c not in {"iteration", "phase", "timestamp"}]
        for col in op_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["phase_duration_s"] = df[op_cols].sum(axis=1, skipna=True)
        df["phase_end_ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["phase_start_ts"] = df["phase_end_ts"] - df["phase_duration_s"]

        df = df.sort_values("phase_start_ts").reset_index(drop=True)
        next_start = df["phase_start_ts"].iloc[1:].to_numpy()
        current_end = df["phase_end_ts"].iloc[:-1].to_numpy()
        gaps = next_start - current_end
        gaps = gaps[gaps >= 0]

        if len(gaps) == 0:
            ax.set_title("No positive gaps found.")
            return

        x = np.arange(len(gaps))
        ax.plot(x, gaps, color="#7f8c8d", linewidth=1.3, marker="o", markersize=3, alpha=0.9)
        ax.set_xlabel("Transition Index")
        ax.set_ylabel("Gap Duration (s)")
        ax.set_title("Orchestration Tax (Gaps)")
        ax.grid(True, alpha=self.theme.grid_alpha)

class HierarchicalWaterfallPlotter(BasePlotter):
    """Single-iteration waterfall chart showing phase + subphase breakdown."""

    plot_name = "hierarchical_waterfall"
    plot_title = "Hierarchical Work Breakdown"
    target_iteration: int = 230

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        suptitle, title = format_title(self.run_paths.run_name, "Hierarchical Work Breakdown")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.timings_df is None or self.timings_df.empty:
            ax.set_title("Missing cleaned_phase_timings data.")
            return

        df = self.timings_df.copy()
        df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
        df = df.dropna(subset=["iteration"])

        target_iter = self.target_iteration
        if target_iter not in df["iteration"].unique():
            unique_iters = sorted(df["iteration"].unique())
            if not unique_iters:
                ax.set_title("No valid iterations found.")
                return
            target_iter = unique_iters[len(unique_iters) // 2]

        df = df[df["iteration"] == target_iter]
        if df.empty:
            ax.set_title("No data for selected iteration.")
            return

        # Build ordered segments per phase
        phase_segments: Dict[str, List[Tuple[str, float]]] = {}
        phase_order = ["rollout", "rl_policy", "training"]
        for phase in phase_order:
            phase_row = df[df["phase"] == phase]
            if phase_row.empty:
                continue
            row = phase_row.iloc[0].to_dict()
            ops = {k: float(v) for k, v in row.items() if k not in {"iteration", "phase", "timestamp"} and isinstance(v, (int, float))}

            ordered_ops = PHASE_OPERATION_ORDER.get(phase, [])
            segments: List[Tuple[str, float]] = []
            if ordered_ops:
                for op in ordered_ops:
                    if op in ops and op not in EXCLUDED_OPERATIONS:
                        segments.append((op, ops[op]))
            else:
                for op, duration in sorted(ops.items()):
                    if op in EXCLUDED_OPERATIONS:
                        continue
                    segments.append((op, duration))
            if segments:
                phase_segments[phase] = segments

        if not phase_segments:
            ax.set_title("No subphase durations available.")
            return

        y_ticks = []
        y_labels = []
        y_pos = 0
        for phase in phase_order:
            if phase not in phase_segments:
                continue
            start = 0.0
            for op, duration in phase_segments[phase]:
                color = PHASE_COLORS.get(phase, PHASE_COLORS["unknown"])
                ax.barh(y_pos, duration, left=start, height=0.6, color=color, alpha=0.8, edgecolor="white")
                if duration > 0:
                    ax.text(start + duration / 2.0, y_pos, op, ha="center", va="center", fontsize=7, color="black")
                start += duration
            y_ticks.append(y_pos)
            y_labels.append(phase)
            y_pos += 1

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Iteration {int(target_iter)}")
        ax.grid(True, axis="x", alpha=self.theme.grid_alpha)


class MFUComparisonPlotter(BasePlotter):
    """Actor vs critic MFU over step."""

    plot_name = "mfu_comparison"
    plot_title = "MFU Comparison"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        suptitle, title = format_title(self.run_paths.run_name, "MFU Comparison")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df
        if "step" not in df.columns:
            ax.set_title("Missing step column.")
            return
        x = pd.to_numeric(df["step"], errors="coerce")
        for col, label, color in [
            ("data.perf/mfu/actor", "Actor MFU", "#2980b9"),
            ("data.perf/mfu/critic", "Critic MFU", "#8e44ad"),
        ]:
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce")
            ax.plot(x, y, label=label, linewidth=1.3, color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel("MFU")
        ax.set_title("MFU Comparison")
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class ThroughputVsLengthPlotter(BasePlotter):
    """Throughput vs sequence length with regression line."""

    plot_name = "throughput_vs_length"
    plot_title = "Throughput vs Seq Length"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        suptitle, title = format_title(self.run_paths.run_name, "Throughput vs Seq Length")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df
        if "data.global_seqlen/mean" not in df.columns or "data.perf/throughput" not in df.columns:
            ax.set_title("Missing required columns.")
            return
        x = pd.to_numeric(df["data.global_seqlen/mean"], errors="coerce")
        y = pd.to_numeric(df["data.perf/throughput"], errors="coerce")
        ax.scatter(x, y, s=20, alpha=0.6, color="#34495e")
        # Simple regression line
        mask = x.notna() & y.notna()
        if mask.sum() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            line_x = np.linspace(x[mask].min(), x[mask].max(), 100)
            line_y = coeffs[0] * line_x + coeffs[1]
            ax.plot(line_x, line_y, color="#e74c3c", linewidth=1.5)
        ax.set_xlabel("Mean Sequence Length")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title("Throughput vs Seq Length")
        ax.grid(True, alpha=self.theme.grid_alpha)


class LearningPricePlotter(BasePlotter):
    """Reward vs total tokens (learning efficiency)."""

    plot_name = "learning_price"
    plot_title = "Learning Price"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        suptitle, title = format_title(self.run_paths.run_name, "Learning Price")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df
        reward_col = "data.val-core/openai/gsm8k/reward/mean@1"
        tokens_col = "data.perf/total_num_tokens"
        if reward_col not in df.columns or tokens_col not in df.columns:
            ax.set_title("Missing required columns.")
            return
        work = df.copy()
        if "step" in work.columns:
            work = work.sort_values("step")

        tokens = pd.to_numeric(work[tokens_col], errors="coerce")
        reward = pd.to_numeric(work[reward_col], errors="coerce")

        # Determine whether tokens are cumulative; if not, derive cumulative tokens.
        diffs = tokens.diff()
        non_decreasing_ratio = (diffs.dropna() >= 0).mean() if diffs.notna().any() else 0.0
        if non_decreasing_ratio >= 0.9:
            tokens_cum = tokens
        else:
            # Fallback: approximate per-step tokens via throughput * time_per_step.
            throughput = pd.to_numeric(work.get("data.perf/throughput"), errors="coerce")
            time_per_step = pd.to_numeric(work.get("data.perf/time_per_step"), errors="coerce")
            per_step_tokens = throughput * time_per_step
            tokens_cum = per_step_tokens.fillna(0).cumsum()

        mask = tokens_cum.notna() & reward.notna()
        x = tokens_cum[mask]
        y = reward[mask]
        if x.empty or y.empty:
            ax.set_title("Missing valid reward/token data.")
            return
        order = np.argsort(x.to_numpy())
        x_sorted = x.to_numpy()[order]
        y_sorted = y.to_numpy()[order]
        ax.plot(x_sorted, y_sorted, color="#95a5a6", linewidth=0.8, alpha=0.6)
        window = max(3, int(len(y_sorted) * 0.1))
        if len(y_sorted) >= window:
            y_smooth = pd.Series(y_sorted).rolling(window=window, min_periods=max(2, window // 3)).mean().to_numpy()
            ax.plot(x_sorted, y_smooth, color="#2ecc71", linewidth=1.6)
        ax.set_xlabel("Total Tokens")
        ax.set_ylabel("Reward")
        ax.set_title("Learning Price")
        ax.grid(True, alpha=self.theme.grid_alpha)

        # Diagnostic: flag potential reward cratering at high token counts.
        if len(y_sorted) >= max(10, window):
            tail_len = max(10, int(len(y_sorted) * 0.15))
            head_len = max(10, int(len(y_sorted) * 0.15))
            head_mean = np.nanmean(y_sorted[:head_len])
            tail_mean = np.nanmean(y_sorted[-tail_len:])
            if np.isfinite(head_mean) and np.isfinite(tail_mean) and tail_mean < head_mean * 0.7:
                ax.text(
                    0.02,
                    0.02,
                    "Warning: Reward drops in tail; verify logging vs true divergence.",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#c0392b",
                )


class MemoryOverheadPlotter(BasePlotter):
    """Area plot showing reserved vs allocated memory gap."""

    plot_name = "memory_overhead"
    plot_title = "Memory Overhead"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        suptitle, title = format_title(self.run_paths.run_name, "Memory Overhead")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df
        alloc_col = "data.perf/max_memory_allocated_gb"
        reserv_col = "data.perf/max_memory_reserved_gb"
        if alloc_col not in df.columns or reserv_col not in df.columns:
            ax.set_title("Missing required columns.")
            return
        x = pd.to_numeric(df.get("step"), errors="coerce") if "step" in df.columns else np.arange(len(df))
        alloc = pd.to_numeric(df[alloc_col], errors="coerce")
        reserv = pd.to_numeric(df[reserv_col], errors="coerce")
        ax.fill_between(x, alloc, reserv, color="#e74c3c", alpha=0.35, label="Reserved - Allocated")
        ax.plot(x, reserv, color="#c0392b", linewidth=1.0, label="Reserved")
        ax.plot(x, alloc, color="#2980b9", linewidth=1.0, label="Allocated")
        ax.set_xlabel("Step" if "step" in df.columns else "Index")
        ax.set_ylabel("Memory (GB)")
        ax.set_title("Memory Overhead")
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class TokenBottlenecksPlotter(BasePlotter):
    """Horizontal bar chart for per-token timing micro-bottlenecks."""

    plot_name = "token_micro_bottlenecks"
    plot_title = "Token Micro-bottlenecks"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        suptitle, title = format_title(self.run_paths.run_name, "Token Micro-bottlenecks")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df
        timing_cols = [c for c in df.columns if c.startswith("data.timing_per_token_ms/")]
        if not timing_cols:
            ax.set_title("No per-token timing columns found.")
            return
        means = {col: pd.to_numeric(df[col], errors="coerce").mean() for col in timing_cols}
        items = sorted(means.items(), key=lambda x: x[1], reverse=True)
        labels = [k.split("/", 1)[-1] for k, _ in items]
        values = [v for _, v in items]
        ax.barh(labels, values, color="#7f8c8d", alpha=0.85)
        ax.set_xlabel("Time per token (ms)")
        ax.set_title("Token Micro-bottlenecks")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=self.theme.grid_alpha)


class ThroughputRewardFrontierPlotter(BasePlotter):
    """Scatter of reward vs throughput with temporal color gradient."""

    plot_name = "throughput_reward_frontier"
    plot_title = "Throughput-Reward Frontier"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df.copy()
        reward_col = "data.val-core/openai/gsm8k/reward/mean@1"
        thr_col = "data.perf/throughput"
        if reward_col not in df.columns or thr_col not in df.columns:
            ax.set_title("Missing required columns.")
            return
        if "step" in df.columns:
            df = df.sort_values("step")
            t = pd.to_numeric(df["step"], errors="coerce")
        else:
            t = pd.Series(np.arange(len(df)), index=df.index)
        x = pd.to_numeric(df[reward_col], errors="coerce")
        y = pd.to_numeric(df[thr_col], errors="coerce")
        mask = x.notna() & y.notna() & t.notna()
        x = x[mask]
        y = y[mask]
        t = t[mask]
        sc = ax.scatter(x, y, c=t, cmap="viridis", s=25, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="Step")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title(self.plot_title)
        ax.grid(True, alpha=self.theme.grid_alpha)


class HardwareROIPlotter(BasePlotter):
    """Dual-axis line chart of MFU (actor/critic) vs rolling reward."""

    plot_name = "hardware_roi"
    plot_title = "Hardware ROI (MFU & Reward)"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df.copy()
        if "step" not in df.columns:
            ax.set_title("Missing step column.")
            return
        df = df.sort_values("step")
        x = pd.to_numeric(df["step"], errors="coerce")

        mfu_actor = pd.to_numeric(df.get("data.perf/mfu/actor"), errors="coerce")
        mfu_critic = pd.to_numeric(df.get("data.perf/mfu/critic"), errors="coerce")
        reward = pd.to_numeric(df.get("data.val-core/openai/gsm8k/reward/mean@1"), errors="coerce")

        ax.plot(x, mfu_actor, label="MFU Actor", color="#2980b9", linewidth=1.2)
        ax.plot(x, mfu_critic, label="MFU Critic", color="#8e44ad", linewidth=1.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("MFU")
        ax.grid(True, alpha=self.theme.grid_alpha)

        ax2 = ax.twinx()
        reward_roll = reward.rolling(window=10, min_periods=3).mean()
        ax2.plot(x, reward_roll, label="Reward (10-step mean)", color="#2ecc71", linewidth=1.5)
        ax2.set_ylabel("Reward")

        # Merge legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")
        ax.set_title(self.plot_title)


class BottleneckEvolutionPlotter(BasePlotter):
    """Compare per-token timings for early vs mature reward regimes."""

    plot_name = "bottleneck_evolution"
    plot_title = "Bottleneck Evolution"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return
        df = self.merged_df.copy()
        reward_col = "data.val-core/openai/gsm8k/reward/mean@1"
        if reward_col not in df.columns:
            ax.set_title("Missing reward column.")
            return
        timing_cols = [c for c in df.columns if c.startswith("data.timing_per_token_ms/")]
        if not timing_cols:
            ax.set_title("No per-token timing columns found.")
            return

        reward = pd.to_numeric(df[reward_col], errors="coerce")
        early_mask = reward < 0.1
        mature_mask = reward > 0.4

        if early_mask.sum() == 0 or mature_mask.sum() == 0:
            ax.set_title("Insufficient early/mature reward samples.")
            return

        early_means = {col: pd.to_numeric(df.loc[early_mask, col], errors="coerce").mean() for col in timing_cols}
        mature_means = {col: pd.to_numeric(df.loc[mature_mask, col], errors="coerce").mean() for col in timing_cols}

        labels = [c.split("/", 1)[-1] for c in timing_cols]
        early_vals = np.array([early_means[c] for c in timing_cols])
        mature_vals = np.array([mature_means[c] for c in timing_cols])

        y = np.arange(len(labels))
        ax.barh(y - 0.2, early_vals, height=0.35, color="#95a5a6", label="Early (Reward < 0.1)")
        ax.barh(y + 0.2, mature_vals, height=0.35, color="#2ecc71", label="Mature (Reward > 0.4)")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Time per token (ms)")
        ax.set_title(self.plot_title)
        ax.grid(True, axis="x", alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class PhaseCorrelationPlotter(BasePlotter):
    """Correlation matrices per phase across 5 metrics."""

    plot_name = "phase_correlations"
    plot_title = "Phase Correlations"

    metrics: Sequence[Tuple[str, str]] = (
        ("gpu_util_percent", "GPU Util (%)"),
        ("power_draw_w", "Power (W)"),
        ("temperature_c", "Temp (°C)"),
        ("memory_used_gb", "Mem Used (GB)"),
    )

    def render(self) -> Path:
        apply_theme(self.theme)

        if sns is None:
            raise ValueError("Seaborn not available for correlation heatmaps.")

        df = self.annotated_df.copy()
        if "phase_name" not in df.columns:
            raise ValueError("Annotated CSV is missing 'phase_name'.")

        df = df[df["phase_name"] != "idle"]
        if df.empty:
            raise ValueError("No non-idle phase data.")

        out_dir = self.output_dir / "phase_correlations"
        out_dir.mkdir(parents=True, exist_ok=True)

        metric_cols = [m for m, _ in self.metrics if m in df.columns]
        metric_labels = [label for (m, label) in self.metrics if m in df.columns]

        for phase in sorted(df["phase_name"].unique()):
            phase_df = df[df["phase_name"] == phase]
            if phase_df[metric_cols].dropna().shape[0] < 2:
                continue

            corr = phase_df[metric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                xticklabels=metric_labels,
                yticklabels=metric_labels,
                cbar_kws={"label": "Correlation"},
                ax=ax,
            )
            ax.set_title(f"Metric Correlations: {phase}")
            fig.tight_layout()
            save_path = out_dir / f"{self.run_paths.run_name}_corr_{phase}.png"
            fig.savefig(save_path, bbox_inches="tight", dpi=self.theme.save_dpi)
            plt.close(fig)

        # Return a representative path for CLI reporting.
        return out_dir

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(6.0 * len(self.metrics), 7))
        if len(self.metrics) == 1:
            axes = np.asarray([axes])
        suptitle, title = format_title(self.run_paths.run_name, "Operation Aggregate")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray(axes)

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        summary = self._build_operation_summary()
        ops = summary["operation"].tolist()

        colors = [OPERATION_COLORS.get(op, OPERATION_COLORS["unknown"]) for op in ops]

        for ax, (col, label) in zip(np.ravel(axes), self.metrics):
            if col not in summary.columns:
                ax.set_title(f"Missing metric: {col}")
                continue

            values = pd.to_numeric(summary[col], errors="coerce")
            ax.barh(ops, values, color=colors, alpha=0.9)
            ax.set_title(label)
            ax.grid(True, axis="x", alpha=self.theme.grid_alpha)

            # Keep the largest values near the top.
            ax.invert_yaxis()


# -----------------------------
# Plot registry + CLI
# -----------------------------

PLOTTERS: Mapping[str, type[BasePlotter]] = {
    "overview": GPUOverviewPlotter,
    "phase_timeline": PhaseTimelinePlotter,
    "phase_aggregate": PhaseAggregatePlotter,
    "phase_focus_rollout": PhaseFocusRolloutPlotter,
    "phase_focus_rl_policy": PhaseFocusRLPolicyPlotter,
    "phase_focus_training": PhaseFocusTrainingPlotter,
    "phase_energy_time_stacked": PhaseEnergyTimeStackedPlotter,
    "phase_boxplots": PhaseBoxplotPlotter,
    "phase_correlations": PhaseCorrelationPlotter,
    "smoothed_timeseries": SmoothedTimeSeriesPlotter,
    "thermal_steady_state": ThermalSteadyStatePlotter,
    "phase_compute_density": PhaseComputeDensityPlotter,
    "hierarchical_waterfall": HierarchicalWaterfallPlotter,
    "mfu_comparison": MFUComparisonPlotter,
    "throughput_vs_length": ThroughputVsLengthPlotter,
    "throughput_reward_frontier": ThroughputRewardFrontierPlotter,
    "hardware_roi": HardwareROIPlotter,
    "learning_price": LearningPricePlotter,
    "token_micro_bottlenecks": TokenBottlenecksPlotter,
    "bottleneck_evolution": BottleneckEvolutionPlotter,
    "sweep_metrics": SweepMetricsPlotter,
    "operation_aggregate": OperationAggregatePlotter,
    "operation_comparison": OperationComparisonPlotter,
}


def build_plotters(
    run_paths: RunPaths,
    output_dir: Path,
    plot_names: Iterable[str],
    theme: ThemeConfig,
) -> List[BasePlotter]:
    plotters: List[BasePlotter] = []
    for name in plot_names:
        cls = PLOTTERS.get(name)
        if cls is None:
            raise ValueError(f"Unknown plot '{name}'. Use --list-plots to see available options.")
        plotters.append(cls(run_paths=run_paths, output_dir=output_dir, theme=theme))
    return plotters


def default_output_dir(root_output: Path, run_paths: RunPaths) -> Path:
    return root_output / run_paths.run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular graphing suite for monitoring_small_cleaned runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=CLEANED_ROOT,
        help="Path to a single run directory or the cleaned root (default: monitoring_small_cleaned).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Root output directory for generated graphs (default: plots).",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="phase_focus_rollout,phase_focus_rl_policy,phase_focus_training,phase_energy_time_stacked,smoothed_timeseries,phase_boxplots,phase_correlations,phase_aggregate,thermal_steady_state,phase_compute_density,hierarchical_waterfall,mfu_comparison,throughput_vs_length,throughput_reward_frontier,hardware_roi,learning_price,token_micro_bottlenecks,bottleneck_evolution",
        help="Comma-separated list of plots to generate.",
    )
    parser.add_argument(
        "--list-plots",
        action="store_true",
        help="List available plot types and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_plots:
        print("Available plots:")
        default_plots = [p.strip() for p in str(parse_args().plots).split(",") if p.strip()]
        for name in default_plots:
            cls = PLOTTERS[name]
            title = getattr(cls, "plot_title", "Plot")
            print(f"  - {name}: {title}")
        return

    plot_names = [p.strip() for p in str(args.plots).split(",") if p.strip()]

    runs = discover_runs(args.root)
    if not runs:
        raise SystemExit(f"No cleaned runs found under: {args.root}")

    theme = ThemeConfig()

    print(f"Discovered {len(runs)} run(s) under {args.root}")
    for run_paths in runs:
        run_output_dir = default_output_dir(args.output_dir, run_paths)
        plotters = build_plotters(run_paths, run_output_dir, plot_names, theme)
        print(f"\nRun: {run_paths.run_name}")
        print(f"  Annotated CSV: {run_paths.annotated_csv}")
        if run_paths.merged_sweep_csv:
            print(f"  Merged sweep CSV: {run_paths.merged_sweep_csv}")
        else:
            print("  Merged sweep CSV: (missing)")

        for plotter in plotters:
            try:
                out_path = plotter.render()
                print(f"  ✓ {plotter.plot_name}: {out_path}")
            except Exception as exc:  # pragma: no cover - CLI robustness
                print(f"  ✗ {plotter.plot_name}: {exc}")


if __name__ == "__main__":
    main()
