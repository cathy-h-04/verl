#!/usr/bin/env python3
"""Hardware-focused plotters."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from ..core.base import BasePlotter, PHASE_COLORS, format_title


class GPUOverviewPlotter(BasePlotter):
    """Overview grid for key GPU telemetry metrics."""

    plot_name = "gpu_overview"
    plot_title = "GPU Overview"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, axes

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df
        time_min = df["elapsed_minutes"]

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

        ax = axes[1, 1]
        ax.plot(time_min, df["sm_clock_mhz"], linewidth=1.1, alpha=0.9, color="#9b59b6")
        ax.set_title("SM Clock Speed")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Clock (MHz)")
        ax.grid(True, alpha=self.theme.grid_alpha)

        ax = axes[1, 2]
        ax.plot(time_min, df["memory_util_percent"], linewidth=1.1, alpha=0.9, color="#1abc9c")
        ax.set_title("Memory Bandwidth Utilization")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Memory Util (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=self.theme.grid_alpha)


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
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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


class ThermalSteadyStatePlotter(BasePlotter):
    """Temperature over time for the full run."""

    plot_name = "thermal_steady_state"
    plot_title = "Thermal Steady-State"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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
    """Utilization vs power draw, colored by phase with centroids/ellipses."""

    plot_name = "phase_compute_density"
    plot_title = "Phase Compute Density"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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
