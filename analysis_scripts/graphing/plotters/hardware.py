#!/usr/bin/env python3
"""Hardware-focused plotters."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Dict, Optional, Sequence, Tuple

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from ..core.base import BasePlotter, PHASE_COLORS, format_title
from ..core.loaders import compute_sample_durations_seconds


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


class EnergyVsSMClockPlotter(BasePlotter):
    """Energy vs SM clock scatter (each sample), colored by phase."""

    plot_name = "energy_vs_sm_clock"
    plot_title = "Energy vs SM Clock"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        df = self.annotated_df.copy()
        ax = axes[0]
        if df is None or df.empty:
            ax.set_title("Missing annotated data.")
            return
        if "power_draw_w" not in df.columns or "sm_clock_mhz" not in df.columns or "phase_name" not in df.columns:
            ax.set_title("Missing power_draw_w, sm_clock_mhz, or phase_name.")
            return

        df = df[df["phase_name"] != "idle"].copy()
        if "iteration" in df.columns:
            iterations = pd.to_numeric(df["iteration"], errors="coerce")
            df = df[iterations > 10]
        if df.empty:
            ax.set_title("No non-idle phase data after iteration 10.")
            return

        dt = compute_sample_durations_seconds(df)
        power = pd.to_numeric(df["power_draw_w"], errors="coerce")
        sm_clock = pd.to_numeric(df["sm_clock_mhz"], errors="coerce")
        energy_j = power * dt
        df["energy_j"] = energy_j
        df["sm_clock_mhz"] = sm_clock

        df = df.dropna(subset=["energy_j", "sm_clock_mhz"])
        if df.empty:
            ax.set_title("No valid energy/clock data.")
            return

        if sns is not None:
            sns.scatterplot(
                data=df,
                x="energy_j",
                y="sm_clock_mhz",
                hue="phase_name",
                palette=PHASE_COLORS,
                alpha=0.6,
                s=14,
                ax=ax,
                legend=True,
            )
        else:
            for phase in df["phase_name"].unique():
                phase_df = df[df["phase_name"] == phase]
                ax.scatter(
                    phase_df["energy_j"],
                    phase_df["sm_clock_mhz"],
                    color=PHASE_COLORS.get(phase, PHASE_COLORS["unknown"]),
                    alpha=0.6,
                    s=14,
                    label=phase,
                )


def _pstate_sort_key(value: str) -> Tuple[int, object]:
    match = re.match(r"[Pp](\d+)", value)
    if match:
        return (0, int(match.group(1)))
    return (1, value)


class PstateDistributionPlotter(BasePlotter):
    """Distribution of P-states by phase (stacked bars)."""

    plot_name = "pstate_distribution"
    plot_title = "P-State Distribution by Phase"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        df = self.annotated_df
        if df is None or df.empty:
            ax.set_title("Missing annotated data.")
            return
        if "pstate" not in df.columns or "phase_name" not in df.columns:
            ax.set_title("Missing pstate or phase_name.")
            return

        work = df.copy()
        work = work[work["phase_name"] != "idle"]
        if work.empty:
            ax.set_title("No non-idle phase data.")
            return

        work["phase_name"] = work["phase_name"].fillna("unknown").astype(str)
        work["pstate"] = work["pstate"].fillna("unknown").astype(str)

        phase_order = ["rollout", "rl_policy", "training"]
        phases = [p for p in phase_order if p in work["phase_name"].unique()]
        phases.extend(sorted(p for p in work["phase_name"].unique() if p not in phases))

        pstates = sorted(work["pstate"].unique().tolist(), key=_pstate_sort_key)
        if not pstates:
            ax.set_title("No pstate data.")
            return

        if sns is not None:
            palette = sns.color_palette("tab10", n_colors=len(pstates))
        else:
            cmap = plt.get_cmap("tab10")
            palette = [cmap(i % 10) for i in range(len(pstates))]

        bottoms = np.zeros(len(phases), dtype=float)
        for idx, pstate in enumerate(pstates):
            values = []
            for phase in phases:
                phase_df = work[work["phase_name"] == phase]
                total = len(phase_df)
                if total == 0:
                    values.append(0.0)
                    continue
                count = (phase_df["pstate"] == pstate).sum()
                values.append(count / total)
            ax.bar(
                phases,
                values,
                bottom=bottoms,
                color=palette[idx],
                label=pstate,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += np.array(values, dtype=float)

        ax.set_xlabel("Phase")
        ax.set_ylabel("Fraction of Samples")
        ax.set_ylim(0, 1.05)
        ax.set_title("P-State Distribution by Phase")
        ax.grid(True, axis="y", alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right", fontsize=9)
        ax.tick_params(axis="x", labelrotation=20)


class UtilizationSmClockScatterPlotter(BasePlotter):
    """GPU utilization vs SM clock scatter by phase."""

    plot_name = "util_sm_clock_scatter"
    plot_title = "GPU Util vs SM Clock"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.annotated_df is None or self.annotated_df.empty:
            ax.set_title("Missing annotated data.")
            return
        df = self.annotated_df.copy()
        if "phase_name" not in df.columns:
            ax.set_title("Missing phase_name.")
            return

        df = df[df["phase_name"] != "idle"]
        if df.empty:
            ax.set_title("No non-idle phase data.")
            return

        df["phase_name"] = df["phase_name"].fillna("unknown").astype(str)
        for phase in sorted(df["phase_name"].unique()):
            phase_df = df[df["phase_name"] == phase]
            x = pd.to_numeric(phase_df.get("gpu_util_percent"), errors="coerce")
            y = pd.to_numeric(phase_df.get("sm_clock_mhz"), errors="coerce")
            mask = x.notna() & y.notna()
            if mask.any():
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=10,
                    alpha=0.45,
                    color=PHASE_COLORS.get(phase, PHASE_COLORS["unknown"]),
                    label=phase,
                )

        ax.set_xlabel("GPU Utilization (%)")
        ax.set_ylabel("SM Clock (MHz)")
        ax.set_title("GPU Util vs SM Clock")
        ax.grid(True, alpha=self.theme.grid_alpha)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=8)


class UtilizationMemoryScatterPlotter(BasePlotter):
    """Memory utilization vs memory used scatter by phase."""

    plot_name = "util_memory_scatter"
    plot_title = "Memory Util vs Memory Used"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.annotated_df is None or self.annotated_df.empty:
            ax.set_title("Missing annotated data.")
            return
        df = self.annotated_df.copy()
        if "phase_name" not in df.columns:
            ax.set_title("Missing phase_name.")
            return

        df = df[df["phase_name"] != "idle"]
        if df.empty:
            ax.set_title("No non-idle phase data.")
            return

        df["phase_name"] = df["phase_name"].fillna("unknown").astype(str)
        for phase in sorted(df["phase_name"].unique()):
            phase_df = df[df["phase_name"] == phase]
            x = pd.to_numeric(phase_df.get("memory_util_percent"), errors="coerce")
            y = pd.to_numeric(phase_df.get("memory_used_gb"), errors="coerce")
            if y.isna().all() and "memory_used_mb" in phase_df.columns:
                y = pd.to_numeric(phase_df.get("memory_used_mb"), errors="coerce") / 1024.0
            mask = x.notna() & y.notna()
            if mask.any():
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=10,
                    alpha=0.45,
                    color=PHASE_COLORS.get(phase, PHASE_COLORS["unknown"]),
                    label=phase,
                )

        ax.set_xlabel("Memory Utilization (%)")
        ax.set_ylabel("Memory Used (GB)")
        ax.set_title("Memory Util vs Memory Used")
        ax.grid(True, alpha=self.theme.grid_alpha)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=8)


def _phase_fingerprint_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if "phase_name" not in df.columns:
        return None

    work = df.copy()
    work = work[work["phase_name"] != "idle"]
    if work.empty:
        return None

    for col in ["power_draw_w", "gpu_util_percent", "sm_clock_mhz"]:
        if col not in work.columns:
            work[col] = np.nan

    dt = compute_sample_durations_seconds(work)
    power = pd.to_numeric(work.get("power_draw_w"), errors="coerce")
    work["sample_energy_j"] = power * dt

    grouped = work.groupby("phase_name", dropna=False)
    summary = grouped.agg(
        avg_power_w=("power_draw_w", "mean"),
        avg_gpu_util=("gpu_util_percent", "mean"),
        avg_sm_clock=("sm_clock_mhz", "mean"),
        total_energy_j=("sample_energy_j", "sum"),
    ).reset_index()

    total_energy = summary["total_energy_j"].sum()
    if total_energy and np.isfinite(total_energy):
        summary["energy_share"] = summary["total_energy_j"] / total_energy
    else:
        summary["energy_share"] = np.nan

    summary["phase_name"] = summary["phase_name"].fillna("unknown").astype(str)
    return summary


class PhaseFingerprintPlotter(BasePlotter):
    """Phase fingerprint panel (rows=phase, columns=key metrics)."""

    plot_name = "phase_fingerprint"
    plot_title = "Phase Fingerprint"

    metrics: Sequence[Tuple[str, str]] = (
        ("avg_power_w", "Power (W)"),
        ("avg_gpu_util", "Util (%)"),
        ("avg_sm_clock", "SM Clock (MHz)"),
        ("energy_share", "Energy Share"),
    )

    metric_maxima: Optional[Dict[str, float]] = None
    show_colorbar: bool = True
    last_mappable: Optional[object] = None

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(9, 4.5))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        summary = _phase_fingerprint_summary(self.annotated_df)
        if summary is None or summary.empty:
            ax.set_title("Missing phase data.")
            return

        phase_order = ["rollout", "rl_policy", "training"]
        phases = [p for p in phase_order if p in summary["phase_name"].unique()]
        phases.extend(sorted(p for p in summary["phase_name"].unique() if p not in phases))

        raw_matrix = np.zeros((len(phases), len(self.metrics)), dtype=float)
        for row_idx, phase in enumerate(phases):
            row = summary[summary["phase_name"] == phase]
            for col_idx, (metric, _) in enumerate(self.metrics):
                value = pd.to_numeric(row.get(metric), errors="coerce").mean()
                raw_matrix[row_idx, col_idx] = value

        norm_matrix = raw_matrix.copy()
        for col_idx, (metric, _) in enumerate(self.metrics):
            denom = None
            if self.metric_maxima is not None:
                denom = self.metric_maxima.get(metric)
            if denom is None or not np.isfinite(denom) or denom == 0:
                col_vals = norm_matrix[:, col_idx]
                denom = np.nanmax(col_vals) if np.isfinite(col_vals).any() else np.nan
            if denom and np.isfinite(denom) and denom != 0:
                norm_matrix[:, col_idx] = norm_matrix[:, col_idx] / denom
            else:
                norm_matrix[:, col_idx] = np.nan

        if sns is not None:
            heatmap = sns.heatmap(
                norm_matrix,
                vmin=0.0,
                vmax=1.0,
                cmap="YlOrRd",
                cbar=self.show_colorbar,
                ax=ax,
                xticklabels=[label for _, label in self.metrics],
                yticklabels=phases,
            )
            try:
                self.last_mappable = heatmap.collections[0]
            except Exception:
                self.last_mappable = None
        else:
            im = ax.imshow(norm_matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlOrRd")
            if self.show_colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(self.metrics)))
            ax.set_xticklabels([label for _, label in self.metrics], rotation=20, ha="right")
            ax.set_yticks(np.arange(len(phases)))
            ax.set_yticklabels(phases)
            self.last_mappable = im

        ax.set_title("Phase Fingerprint (Normalized)")

        ax.set_xlabel("Sample Energy (J)")
        ax.set_ylabel("SM Clock (MHz)")
        ax.set_title("Energy vs SM Clock")
        ax.grid(True, alpha=self.theme.grid_alpha)
