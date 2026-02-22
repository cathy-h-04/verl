#!/usr/bin/env python3
"""Timing/phase-focused plotters."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from ..core.base import (
    BasePlotter,
    PHASE_COLORS,
    PHASE_OPERATION_ORDER,
    EXCLUDED_OPERATIONS,
    OPERATION_COLORS,
    format_title,
)
from ..core.loaders import compute_phase_windows_from_annotated, compute_sample_durations_seconds


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
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df
        if "timestamp_aligned_unix" not in df.columns:
            raise ValueError("Annotated CSV is missing timestamp_aligned_unix; cannot build phase timeline.")

        x = pd.to_numeric(df["timestamp_aligned_unix"], errors="coerce")
        x0 = x.min()
        x_rel_min = (x - x0) / 60.0

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
            ax.grid(True, alpha=self.theme.grid_alpha)

            for phase, start_min, end_min in phase_windows_rel:
                color = PHASE_COLORS.get(phase, PHASE_COLORS["unknown"])
                ax.axvspan(start_min, end_min, color=color, alpha=0.08)

        axes[-1].set_xlabel("Time Since Start (minutes)")

    def annotate(self, fig: plt.Figure, axes: np.ndarray) -> None:
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


class PhaseAggregatePlotter(BasePlotter):
    """Aggregate GPU metrics per phase."""

    plot_name = "phase_aggregate"
    plot_title = "Phase Aggregates"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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
    """Per-iteration plots grouped by phase (3 panels)."""

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
        suptitle, _ = format_title(self.run_paths.run_name, f"Iteration Focus: {self.focus_phase}")
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
            ax.set_title(ylabel)
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


def build_phase_metric_timeseries(
    df: pd.DataFrame,
    metric: str,
    use_time_axis: bool = True,
) -> Tuple[pd.DataFrame, str, str]:
    """Aggregate metric by iteration + phase over full run."""
    if df is None or df.empty or "phase_name" not in df.columns:
        return pd.DataFrame(), "iteration", "Iteration"

    work = df.copy()
    work = work[work["phase_name"] != "idle"]
    work["iteration"] = pd.to_numeric(work.get("iteration"), errors="coerce")
    work = work.dropna(subset=["iteration"])
    if work.empty:
        return pd.DataFrame(), "iteration", "Iteration"

    if use_time_axis and "timestamp_aligned_unix" in work.columns:
        time_raw = pd.to_numeric(work["timestamp_aligned_unix"], errors="coerce")
        time_ref = time_raw.min()
        work["time_min"] = (time_raw - time_ref) / 60.0
        x_col = "time_min"
        x_label = "Time (minutes)"
    elif use_time_axis and "elapsed_seconds" in work.columns:
        time_raw = pd.to_numeric(work["elapsed_seconds"], errors="coerce")
        time_ref = time_raw.min()
        work["time_min"] = (time_raw - time_ref) / 60.0
        x_col = "time_min"
        x_label = "Time (minutes)"
    else:
        x_col = "iteration"
        x_label = "Iteration"

    if metric not in work.columns:
        return pd.DataFrame(), x_col, x_label

    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric, x_col])
    if work.empty:
        return pd.DataFrame(), x_col, x_label

    agg_map = {metric: "mean", x_col: "mean"}
    grouped = work.groupby(["iteration", "phase_name"], dropna=False).agg(agg_map)
    grouped = grouped.reset_index()
    grouped["phase_name"] = grouped["phase_name"].fillna("unknown").astype(str)
    return grouped, x_col, x_label


class PhaseAggregateMetricsPlotter(BasePlotter):
    """Full-run, phase-aggregated metrics over time."""

    plot_name = "phase_aggregate_metrics"
    plot_title = "Phase Metrics (All Iterations)"
    metrics: Sequence[Tuple[str, str]] = ()
    use_time_axis: bool = True

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(14, 4.0 * len(self.metrics)), sharex=True)
        if len(self.metrics) == 1:
            axes = np.asarray([axes])
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray(axes)

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        phases = ["rollout", "rl_policy", "training", "unknown"]
        phase_styles = {
            "rollout": {"marker": "o", "linestyle": "-"},
            "rl_policy": {"marker": "s", "linestyle": "--"},
            "training": {"marker": "^", "linestyle": "-."},
            "unknown": {"marker": "x", "linestyle": ":"},
        }

        for ax, (metric, ylabel) in zip(np.ravel(axes), self.metrics):
            series, x_col, x_label = build_phase_metric_timeseries(
                self.annotated_df, metric, use_time_axis=self.use_time_axis
            )
            if series.empty or metric not in series.columns:
                ax.set_title(f"{ylabel} (missing column: {metric})")
                continue

            for phase in phases:
                phase_df = series[series["phase_name"] == phase]
                if phase_df.empty:
                    continue
                color = PHASE_COLORS.get(phase, PHASE_COLORS["unknown"])
                style = phase_styles.get(phase, {"marker": "o", "linestyle": "-"})
                ax.plot(
                    phase_df[x_col],
                    phase_df[metric],
                    color=color,
                    alpha=0.8,
                    linewidth=1.6,
                    marker=style["marker"],
                    markersize=4,
                    linestyle=style["linestyle"],
                    label=phase,
                )

            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(True, alpha=self.theme.grid_alpha)
            ax.set_xlabel(x_label)

        handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.92, 0.95), frameon=False)


class PhaseAggregateSMClockPlotter(PhaseAggregateMetricsPlotter):
    plot_name = "phase_sm_clock"
    plot_title = "SM Clock by Phase (All Iterations)"
    metrics = (("sm_clock_mhz", "SM Clock (MHz)"),)


class PhaseAggregateMemoryClockPlotter(PhaseAggregateMetricsPlotter):
    plot_name = "phase_memory_clock"
    plot_title = "Memory Clock by Phase (All Iterations)"
    metrics = (("mem_clock_mhz", "Memory Clock (MHz)"),)


class PhaseEnergyTimeStackedPlotter(BasePlotter):
    """Energy and time distribution across phases (100% stacked bars)."""

    plot_name = "phase_energy_time_stacked"
    plot_title = "Energy and Time Distribution"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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
                        fontsize=12,
                        color="white",
                        fontweight="bold",
                    )
            bottoms += frac_values

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Fraction of Total")
        axes[0].set_title("Energy and Time Distribution")
        axes[0].grid(True, axis="y", alpha=self.theme.grid_alpha)
        axes[0].legend(loc="upper right", title="Phase")


class PhasePeakPowerPlotter(BasePlotter):
    """Peak power draw per phase (p95/p99/max)."""

    plot_name = "phase_peak_power"
    plot_title = "Peak Power Draw by Phase"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df.copy()
        if "phase_name" not in df.columns or "power_draw_w" not in df.columns:
            axes[0].set_title("Missing phase_name or power_draw_w.")
            return

        df = df[df["phase_name"] != "idle"]
        if df.empty:
            axes[0].set_title("No non-idle phase data.")
            return

        df["power_draw_w"] = pd.to_numeric(df["power_draw_w"], errors="coerce")
        grouped = df.groupby("phase_name", dropna=False)["power_draw_w"]

        phases = ["rollout", "rl_policy", "training"]
        phases.extend([p for p in grouped.groups.keys() if p not in phases])
        phases = [p for p in phases if p in grouped.groups]

        p95_vals = []
        p99_vals = []
        max_vals = []
        for phase in phases:
            series = grouped.get_group(phase).dropna()
            if series.empty:
                p95_vals.append(np.nan)
                p99_vals.append(np.nan)
                max_vals.append(np.nan)
                continue
            p95_vals.append(float(series.quantile(0.95)))
            p99_vals.append(float(series.quantile(0.99)))
            max_vals.append(float(series.max()))

        x = np.arange(len(phases))
        bar_width = 0.22
        offsets = [-bar_width, 0.0, bar_width]
        metric_labels = ["p95", "p99", "max"]
        hatches = {"p95": "//", "p99": "..", "max": ""}

        for vals, label, offset in zip([p95_vals, p99_vals, max_vals], metric_labels, offsets):
            for idx, phase in enumerate(phases):
                color = PHASE_COLORS.get(phase, PHASE_COLORS["unknown"])
                axes[0].bar(
                    x[idx] + offset,
                    vals[idx],
                    width=bar_width,
                    color=color,
                    alpha=0.9,
                    hatch=hatches[label],
                    edgecolor="#2c3e50",
                    linewidth=0.4,
                )

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(phases, rotation=20, ha="right")
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title("Peak Power Draw by Phase")
        axes[0].grid(True, axis="y", alpha=self.theme.grid_alpha)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#95a5a6", hatch=hatches[label], label=label)
            for label in metric_labels
        ]
        axes[0].legend(handles=legend_handles, loc="upper right", title="Metric")


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
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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


class PhaseCorrelationPlotter(BasePlotter):
    """Correlation matrices per phase."""

    plot_name = "phase_correlations"
    plot_title = "Phase Correlations"

    metrics: Sequence[Tuple[str, str]] = (
        ("gpu_util_percent", "GPU Util (%)"),
        ("power_draw_w", "Power (W)"),
        ("temperature_c", "Temp (°C)"),
        ("memory_used_gb", "Mem Used (GB)"),
    )

    def render(self) -> Path:
        from ..core.base import apply_theme

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

        return out_dir


class HierarchicalWaterfallPlotter(BasePlotter):
    """Mean operation breakdown by phase (stacked bars)."""

    plot_name = "hierarchical_waterfall"
    plot_title = "Hierarchical Work Breakdown"
    target_iteration: int = 230

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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
        if df.empty:
            ax.set_title("No valid iterations found.")
            return

        phase_order = ["rollout", "rl_policy", "training"]
        phase_segments: Dict[str, List[Tuple[str, float]]] = {}

        for phase in phase_order:
            phase_df = df[df["phase"] == phase]
            if phase_df.empty:
                continue
            ops = {
                k: pd.to_numeric(phase_df[k], errors="coerce").mean()
                for k in phase_df.columns
                if k not in {"iteration", "phase", "timestamp"}
            }
            ordered_ops = PHASE_OPERATION_ORDER.get(phase, [])
            segments: List[Tuple[str, float]] = []
            if ordered_ops:
                for op in ordered_ops:
                    if op in ops and op not in EXCLUDED_OPERATIONS and ops[op] and ops[op] > 0:
                        segments.append((op, float(ops[op])))
            else:
                for op, duration in sorted(ops.items(), key=lambda kv: kv[1], reverse=True):
                    if op in EXCLUDED_OPERATIONS:
                        continue
                    if duration and duration > 0:
                        segments.append((op, float(duration)))
            if segments:
                phase_segments[phase] = segments

        if not phase_segments:
            ax.set_title("No subphase durations available.")
            return

        y_ticks = []
        y_labels = []
        legend_handles = []
        legend_labels = []
        y_pos = 0
        for phase in phase_order:
            if phase not in phase_segments:
                continue
            start = 0.0
            for op, duration in phase_segments[phase]:
                color = OPERATION_COLORS.get(op, OPERATION_COLORS.get("unknown", "#95a5a6"))
                if phase == "rl_policy" and op == "values":
                    color = "#f5b041"
                ax.barh(y_pos, duration, left=start, height=0.6, color=color, alpha=0.9, edgecolor="white")
                start += duration
                if op not in legend_labels:
                    legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                    legend_labels.append(op)
            y_ticks.append(y_pos)
            y_labels.append(phase)
            y_pos += 1

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (s)")
        ax.set_title("Mean Operation Durations by Phase")
        ax.grid(True, axis="x", alpha=self.theme.grid_alpha)
        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper right",
                fontsize=8,
                frameon=False,
            )
