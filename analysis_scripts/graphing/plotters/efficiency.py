#!/usr/bin/env python3
"""Efficiency/ROI plotters using merged sweep metrics."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from ..core.base import BasePlotter, OPERATION_COLORS, EXCLUDED_OPERATIONS, format_title
from ..core.loaders import compute_sample_durations_seconds


class SweepMetricsPlotter(BasePlotter):
    """Stepwise training + validation metrics from merged sweep CSV."""

    plot_name = "sweep_metrics"
    plot_title = "Sweep Metrics"

    sweep_metrics: Sequence[Tuple[str, str]] = (
        ("data.val-core/openai/gsm8k/reward/mean@1", "Reward"),
        ("data.perf/throughput", "Throughput (tokens/s)"),
        ("data.perf/time_per_step", "Time Per Step (s)"),
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(len(self.sweep_metrics), 1, figsize=(12, 4.0 * len(self.sweep_metrics)), sharex=True)
        if len(self.sweep_metrics) == 1:
            axes = np.asarray([axes])
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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


class MFUComparisonPlotter(BasePlotter):
    """Actor vs critic MFU over step."""

    plot_name = "mfu_comparison"
    plot_title = "MFU Comparison"

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
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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
    """Dual-axis line chart of MFU vs rolling reward."""

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

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")
        ax.set_title(self.plot_title)


class LearningPricePlotter(BasePlotter):
    """Reward vs total tokens (learning efficiency)."""

    plot_name = "learning_price"
    plot_title = "Learning Price"

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

        diffs = tokens.diff()
        non_decreasing_ratio = (diffs.dropna() >= 0).mean() if diffs.notna().any() else 0.0
        if non_decreasing_ratio >= 0.9:
            tokens_cum = tokens
        else:
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


class TokenBottlenecksPlotter(BasePlotter):
    """Horizontal bar chart for per-token timing micro-bottlenecks."""

    plot_name = "token_micro_bottlenecks"
    plot_title = "Token Micro-bottlenecks"

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
        ax.set_title("Bottleneck Evolution")
        ax.grid(True, axis="x", alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


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

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(6.0 * len(self.metrics), 7))
        if len(self.metrics) == 1:
            axes = np.asarray([axes])
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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
            ax.invert_yaxis()


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
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
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

        for ax in flat_axes[len(self.metrics):]:
            ax.set_visible(False)
