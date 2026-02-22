#!/usr/bin/env python3
"""Efficiency/ROI plotters using merged sweep metrics."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from ..core.base import BasePlotter, OPERATION_COLORS, EXCLUDED_OPERATIONS, format_title
from ..core.loaders import compute_sample_durations_seconds


def _algo_label(run_name: str) -> str:
    parts = [p for p in run_name.lower().split("_") if p]
    for token in parts:
        if token == "ppo":
            return "PPO"
        if token == "grpo":
            return "GRPO"
        if token == "remax":
            return "ReMax"
    return "Unknown"


def _response_length_cv(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    mean_col = next((c for c in df.columns if "response_length" in c and c.endswith("/mean")), None)
    std_col = next((c for c in df.columns if "response_length" in c and c.endswith("/std")), None)
    if std_col and mean_col:
        mean = pd.to_numeric(df[mean_col], errors="coerce")
        std = pd.to_numeric(df[std_col], errors="coerce")
        return std / mean.replace(0, np.nan), "std"

    max_col = next((c for c in df.columns if "response_length" in c and c.endswith("/max")), None)
    min_col = next((c for c in df.columns if "response_length" in c and c.endswith("/min")), None)
    if mean_col and max_col and min_col:
        mean = pd.to_numeric(df[mean_col], errors="coerce")
        max_val = pd.to_numeric(df[max_col], errors="coerce")
        min_val = pd.to_numeric(df[min_col], errors="coerce")
        approx_std = (max_val - min_val) / 4.0
        return approx_std / mean.replace(0, np.nan), "range"

    return None, None


def _per_step_tokens(df: pd.DataFrame) -> Optional[pd.Series]:
    tokens_col = "data.perf/total_num_tokens"
    if tokens_col in df.columns:
        tokens = pd.to_numeric(df[tokens_col], errors="coerce")
        diffs = tokens.diff()
        if diffs.notna().any() and (diffs >= 0).all():
            return diffs.fillna(tokens)
        return tokens

    if "data.perf/throughput" in df.columns and "data.perf/time_per_step" in df.columns:
        throughput = pd.to_numeric(df["data.perf/throughput"], errors="coerce")
        time_per_step = pd.to_numeric(df["data.perf/time_per_step"], errors="coerce")
        return throughput * time_per_step
    return None


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

        if not isinstance(mfu_actor, pd.Series):
            mfu_actor = pd.Series([np.nan] * len(df), index=df.index)
        if not isinstance(mfu_critic, pd.Series):
            mfu_critic = pd.Series([np.nan] * len(df), index=df.index)
        if not isinstance(reward, pd.Series):
            reward = pd.Series([np.nan] * len(df), index=df.index)

        mfu_actor = mfu_actor.reindex(df.index)
        mfu_critic = mfu_critic.reindex(df.index)
        reward = reward.reindex(df.index)

        if len(mfu_actor) != len(x) or len(mfu_critic) != len(x):
            ax.set_title("MFU series length mismatch.")
            return

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


class EnergyPerTokenPlotter(BasePlotter):
    """Energy per token (or useful token) over time."""

    plot_name = "energy_per_token"
    plot_title = "Energy per Token"

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
        throughput_col = "data.perf/throughput"
        tokens_col = "data.perf/total_num_tokens"
        aborted_col = "data.response/aborted_ratio"

        if throughput_col not in df.columns:
            ax.set_title("Missing throughput data.")
            return

        work = df.copy()
        if "step" in work.columns:
            work = work.sort_values("step")

        throughput = pd.to_numeric(work[throughput_col], errors="coerce")
        tokens = pd.to_numeric(work.get(tokens_col), errors="coerce")

        power_mean = None
        if self.annotated_df is not None and "power_draw_w" in self.annotated_df.columns:
            power_df = self.annotated_df.copy()
            if "phase_name" in power_df.columns:
                power_df = power_df[power_df["phase_name"] != "idle"]
            power_mean = pd.to_numeric(power_df["power_draw_w"], errors="coerce").mean()

        if power_mean is None or not np.isfinite(power_mean):
            ax.set_title("Missing power draw data.")
            return

        use_useful = False
        if aborted_col in work.columns:
            aborted = pd.to_numeric(work[aborted_col], errors="coerce")
            if aborted.notna().any() and float(aborted.mean()) > 0.01:
                use_useful = True
                throughput = throughput * (1.0 - aborted)

        energy_per_token = power_mean / throughput.replace(0, np.nan)

        tokens_cum = None
        if tokens.notna().any():
            diffs = tokens.diff()
            non_decreasing_ratio = (diffs.dropna() >= 0).mean() if diffs.notna().any() else 0.0
            if non_decreasing_ratio >= 0.9:
                tokens_cum = tokens

        if tokens_cum is None:
            time_per_step = pd.to_numeric(work.get("data.perf/time_per_step"), errors="coerce")
            per_step_tokens = throughput * time_per_step
            if per_step_tokens.notna().any():
                tokens_cum = per_step_tokens.fillna(0).cumsum()

        if tokens_cum is not None and tokens_cum.notna().any():
            x = tokens_cum
            x_label = "Total Tokens"
        else:
            x = pd.to_numeric(work.get("step"), errors="coerce")
            x_label = "Step"

        mask = x.notna() & energy_per_token.notna()
        if not mask.any():
            ax.set_title("No valid energy-per-token data.")
            return

        order = np.argsort(x[mask].to_numpy())
        x_sorted = x[mask].to_numpy()[order]
        y_sorted = energy_per_token[mask].to_numpy()[order]

        ax.plot(x_sorted, y_sorted, color="#95a5a6", linewidth=0.8, alpha=0.6)
        window = max(3, int(len(y_sorted) * 0.1))
        if len(y_sorted) >= window:
            y_smooth = pd.Series(y_sorted).rolling(window=window, min_periods=max(2, window // 3)).mean().to_numpy()
            ax.plot(x_sorted, y_smooth, color="#2ecc71", linewidth=1.6)

        ylabel = "Energy per Useful Token (J)" if use_useful else "Energy per Token (J)"
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title("Energy per Token")
        ax.grid(True, alpha=self.theme.grid_alpha)


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
        timing_cols = [c for c in df.columns if c.startswith("data.timing_per_token_ms/")]
        if not timing_cols:
            ax.set_title("No per-token timing columns found.")
            return

        reward = None
        early_mask = None
        mature_mask = None
        subtitle = None

        if reward_col in df.columns:
            reward = pd.to_numeric(df[reward_col], errors="coerce")
            finite_reward = reward.dropna()
            if not finite_reward.empty:
                q_low = float(finite_reward.quantile(0.2))
                q_high = float(finite_reward.quantile(0.8))
                if q_low < q_high:
                    early_mask = reward <= q_low
                    mature_mask = reward >= q_high
                    subtitle = "Early/Mature by reward (p20/p80)"

        if early_mask is None or mature_mask is None or early_mask.sum() == 0 or mature_mask.sum() == 0:
            # Fallback: use step percentiles when reward thresholds are not usable.
            if "step" not in df.columns:
                ax.set_title("Insufficient reward data and missing step column.")
                return
            steps = pd.to_numeric(df["step"], errors="coerce")
            finite_steps = steps.dropna()
            if finite_steps.empty:
                ax.set_title("Insufficient reward data and invalid steps.")
                return
            s_low = float(finite_steps.quantile(0.2))
            s_high = float(finite_steps.quantile(0.8))
            early_mask = steps <= s_low
            mature_mask = steps >= s_high
            subtitle = "Early/Mature by step (p20/p80)"

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
        if subtitle:
            ax.text(
                0.99,
                1.02,
                subtitle,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
        ax.grid(True, axis="x", alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class TailLatencyTaxMapPlotter(BasePlotter):
    """Tail-Latency Tax Map: response length CV vs gen time per token."""

    plot_name = "tail_latency_tax_map"
    plot_title = "Tail-Latency Tax Map"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return

        df = self.merged_df.copy()
        iter_col = "step" if "step" in df.columns else "data.training/global_step"
        if iter_col not in df.columns:
            ax.set_title("Missing iteration column.")
            return

        iterations = pd.to_numeric(df[iter_col], errors="coerce")
        df = df[iterations >= 50].copy()
        if df.empty:
            ax.set_title("No data after iteration 50.")
            return

        cv, cv_method = _response_length_cv(df)
        if cv is None:
            ax.set_title("Missing response length stats for CV.")
            return

        if "data.timing_s/gen" in df.columns:
            gen_time = pd.to_numeric(df["data.timing_s/gen"], errors="coerce")
        elif "data.timing_per_token_ms/gen" in df.columns:
            per_token_ms = pd.to_numeric(df["data.timing_per_token_ms/gen"], errors="coerce")
            gen_time = per_token_ms / 1000.0
        else:
            ax.set_title("Missing gen subphase timing.")
            return

        tokens = _per_step_tokens(df)
        if tokens is None:
            ax.set_title("Missing total token counts.")
            return

        y = gen_time / tokens.replace(0, np.nan)
        x = cv

        algo = _algo_label(self.run_paths.run_name)
        plot_df = pd.DataFrame(
            {
                "cv": x,
                "gen_time_per_token": y,
            }
        ).dropna()

        if plot_df.empty:
            ax.set_title("No valid CV/gen-time data.")
            return

        color_map = {"PPO": "#2980b9", "GRPO": "#27ae60", "ReMax": "#e67e22", "Unknown": "#7f8c8d"}
        ax.scatter(
            plot_df["cv"],
            plot_df["gen_time_per_token"],
            s=28,
            alpha=0.75,
            color=color_map.get(algo, "#7f8c8d"),
            label=algo,
        )

        if len(plot_df) >= 2:
            coeffs = np.polyfit(plot_df["cv"], plot_df["gen_time_per_token"], 1)
            line_x = np.linspace(plot_df["cv"].min(), plot_df["cv"].max(), 100)
            line_y = coeffs[0] * line_x + coeffs[1]
            ax.plot(line_x, line_y, color="#2c3e50", linewidth=1.6, alpha=0.9)

        x_label = "Sequence Length CV"
        if cv_method == "range":
            x_label += " (approx)"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Gen Time per Token (s/token)")
        ax.set_title(self.plot_title)
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class ClockImbalanceMapPlotter(BasePlotter):
    """Clock speed vs batch imbalance using response length spread."""

    plot_name = "clock_imbalance_map"
    plot_title = "Clock Speed vs Batch Imbalance"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.merged_df is None:
            ax.set_title("Missing merged sweep CSV.")
            return

        df = self.merged_df.copy()
        iter_col = "step" if "step" in df.columns else "data.training/global_step"
        if iter_col not in df.columns:
            ax.set_title("Missing iteration column.")
            return

        max_col = next((c for c in df.columns if "response_length" in c and c.endswith("/max")), None)
        min_col = next((c for c in df.columns if "response_length" in c and c.endswith("/min")), None)
        mean_col = next((c for c in df.columns if "response_length" in c and c.endswith("/mean")), None)
        if not (max_col and min_col and mean_col):
            ax.set_title("Missing response length stats.")
            return

        iterations = pd.to_numeric(df[iter_col], errors="coerce")
        max_val = pd.to_numeric(df[max_col], errors="coerce")
        min_val = pd.to_numeric(df[min_col], errors="coerce")
        mean_val = pd.to_numeric(df[mean_col], errors="coerce")
        imbalance = (max_val - min_val) / mean_val.replace(0, np.nan)

        if "iteration" not in self.annotated_df.columns:
            ax.set_title("Missing iteration in annotated CSV.")
            return
        if "sm_clock_mhz" not in self.annotated_df.columns:
            ax.set_title("Missing sm_clock_mhz in annotated CSV.")
            return

        work = self.annotated_df.copy()
        if "operation" in work.columns:
            op_target = "gen_max" if _algo_label(self.run_paths.run_name) == "ReMax" else "gen"
            work = work[work["operation"] == op_target].copy()
        else:
            ax.set_title("Missing operation in annotated CSV.")
            return
        work["iteration"] = pd.to_numeric(work["iteration"], errors="coerce")
        work["sm_clock_mhz"] = pd.to_numeric(work["sm_clock_mhz"], errors="coerce")
        work = work.dropna(subset=["iteration", "sm_clock_mhz"])
        if work.empty:
            ax.set_title("No rollout-phase clock data.")
            return

        sm_clock = work.groupby("iteration")["sm_clock_mhz"].mean()

        plot_df = pd.DataFrame(
            {
                "iteration": iterations,
                "imbalance": imbalance,
            }
        ).dropna()
        plot_df = plot_df.merge(
            sm_clock.rename("sm_clock_mhz"),
            left_on="iteration",
            right_index=True,
            how="inner",
        )

        if plot_df.empty:
            ax.set_title("No matching imbalance/clock data.")
            return

        algo = _algo_label(self.run_paths.run_name)
        color_map = {"PPO": "#2980b9", "GRPO": "#27ae60", "ReMax": "#e67e22", "Unknown": "#7f8c8d"}
        ax.scatter(
            plot_df["imbalance"],
            plot_df["sm_clock_mhz"],
            s=28,
            alpha=0.75,
            color=color_map.get(algo, "#7f8c8d"),
            label=algo,
        )

        if len(plot_df) >= 2:
            coeffs = np.polyfit(plot_df["imbalance"], plot_df["sm_clock_mhz"], 1)
            line_x = np.linspace(plot_df["imbalance"].min(), plot_df["imbalance"].max(), 100)
            line_y = coeffs[0] * line_x + coeffs[1]
            ax.plot(line_x, line_y, color="#2c3e50", linewidth=1.6, alpha=0.9)

        ax.set_xlabel("Response Length Imbalance (max-min)/mean")
        ax.set_ylabel("Mean SM Clock (MHz) during rollout")
        ax.set_title(self.plot_title)
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class BoostHeadroomHistogramPlotter(BasePlotter):
    """Density plot of SM clock during update_actor to show boost headroom."""

    plot_name = "boost_headroom_histogram"
    plot_title = "Boost Headroom (Update Actor)"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(9, 5))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        df = self.annotated_df
        if "operation" not in df.columns:
            ax.set_title("Missing operation in annotated CSV.")
            return
        if "sm_clock_mhz" not in df.columns:
            ax.set_title("Missing sm_clock_mhz in annotated CSV.")
            return

        work = df[df["operation"] == "update_actor"].copy()
        work["sm_clock_mhz"] = pd.to_numeric(work["sm_clock_mhz"], errors="coerce")
        work = work.dropna(subset=["sm_clock_mhz"])
        if work.empty:
            ax.set_title("No update_actor clock data.")
            return

        algo = _algo_label(self.run_paths.run_name)
        color_map = {"PPO": "#2980b9", "GRPO": "#27ae60", "ReMax": "#e67e22", "Unknown": "#7f8c8d"}
        color = color_map.get(algo, "#7f8c8d")

        vals = work["sm_clock_mhz"].to_numpy()
        bins = np.linspace(1300.0, 1500.0, 31)
        hist, edges = np.histogram(vals, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        ax.bar(
            centers,
            hist,
            width=widths,
            color=color,
            alpha=0.75,
            edgecolor="#2c3e50",
            linewidth=0.4,
            label=algo,
        )

        ax.set_xlabel("SM Clock (MHz)")
        ax.set_ylabel("Density")
        ax.set_title(self.plot_title)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class CriticIntensityMapPlotter(BasePlotter):
    """Subphase duration vs peak power for critic/actor subphases."""

    plot_name = "critic_intensity_map"
    plot_title = "Critic Intensity Map"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        if self.timings_df is None or self.timings_df.empty:
            ax.set_title("Missing cleaned phase timings JSONL.")
            return
        if "timestamp_aligned_unix" not in self.annotated_df.columns:
            ax.set_title("Missing timestamp_aligned_unix in annotated CSV.")
            return
        if "power_draw_w" not in self.annotated_df.columns:
            ax.set_title("Missing power_draw_w in annotated CSV.")
            return

        power_df = self.annotated_df.copy()
        power_df["timestamp_aligned_unix"] = pd.to_numeric(power_df["timestamp_aligned_unix"], errors="coerce")
        power_df["power_draw_w"] = pd.to_numeric(power_df["power_draw_w"], errors="coerce")
        power_df = power_df.dropna(subset=["timestamp_aligned_unix", "power_draw_w"])
        if power_df.empty:
            ax.set_title("No power readings available.")
            return

        series = {"values": [], "update_critic": [], "update_actor": []}

        for _, row in self.timings_df.iterrows():
            phase = row.get("phase")
            ts = row.get("timestamp")
            if ts is None:
                continue
            ts = float(ts)

            if phase == "rl_policy" and pd.notna(row.get("values")):
                reward = float(row.get("reward", 0.0) or 0.0)
                old_log_prob = float(row.get("old_log_prob", 0.0) or 0.0)
                dur = float(row.get("values") or 0.0)
                start = ts + reward + old_log_prob
                end = start + dur
                series["values"].append((dur * 1000.0, start, end))

            if phase == "training":
                if pd.notna(row.get("update_critic")):
                    dur = float(row.get("update_critic") or 0.0)
                    start = ts
                    end = start + dur
                    series["update_critic"].append((dur * 1000.0, start, end))
                if pd.notna(row.get("update_actor")):
                    dur = float(row.get("update_actor") or 0.0)
                    start = ts + float(row.get("update_critic", 0.0) or 0.0)
                    end = start + dur
                    series["update_actor"].append((dur * 1000.0, start, end))

        color_map = {
            "values": "#8e44ad",
            "update_critic": "#e67e22",
            "update_actor": "#2980b9",
        }

        plotted = 0
        for name, windows in series.items():
            points = []
            for dur_ms, start, end in windows:
                window = power_df[(power_df["timestamp_aligned_unix"] >= start) & (power_df["timestamp_aligned_unix"] <= end)]
                if window.empty:
                    continue
                peak_power = window["power_draw_w"].max()
                points.append((dur_ms, peak_power))
            if not points:
                continue
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            ax.scatter(x_vals, y_vals, s=26, alpha=0.75, color=color_map[name], label=name)
            plotted += 1

        if plotted == 0:
            ax.set_title("No matching timing/power windows.")
            return

        ax.set_xlabel("Subphase Duration (ms)")
        ax.set_ylabel("Peak Power (W)")
        ax.set_title(self.plot_title)
        ax.grid(True, alpha=self.theme.grid_alpha)
        ax.legend(loc="upper right")


class PhaseToPhaseStabilityBoxplot(BasePlotter):
    """Boxplot of |avg clock(update_actor) - avg clock(gen)| per iteration."""

    plot_name = "phase_to_phase_stability"
    plot_title = "Phase-to-Phase Stability"

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        suptitle, _ = format_title(self.run_paths.run_name, self.plot_title)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig, np.asarray([axes])

    def draw(self, fig: plt.Figure, axes: np.ndarray) -> None:
        ax = axes[0]
        df = self.annotated_df
        if "operation" not in df.columns or "iteration" not in df.columns:
            ax.set_title("Missing operation or iteration in annotated CSV.")
            return
        if "sm_clock_mhz" not in df.columns:
            ax.set_title("Missing sm_clock_mhz in annotated CSV.")
            return

        work = df.copy()
        work["iteration"] = pd.to_numeric(work["iteration"], errors="coerce")
        work["sm_clock_mhz"] = pd.to_numeric(work["sm_clock_mhz"], errors="coerce")
        work = work.dropna(subset=["iteration", "sm_clock_mhz"])
        if work.empty:
            ax.set_title("No clock data.")
            return

        algo = _algo_label(self.run_paths.run_name)
        gen_op = "gen_max" if algo == "ReMax" else "gen"

        gen = work[work["operation"] == gen_op]
        upd = work[work["operation"] == "update_actor"]
        if gen.empty or upd.empty:
            ax.set_title("Missing gen/update_actor data.")
            return

        gen_mean = gen.groupby("iteration")["sm_clock_mhz"].mean()
        upd_mean = upd.groupby("iteration")["sm_clock_mhz"].mean()
        joined = pd.concat([gen_mean.rename("gen"), upd_mean.rename("update")], axis=1).dropna()
        if joined.empty:
            ax.set_title("No overlapping iterations.")
            return

        delta = (joined["update"] - joined["gen"]).abs()
        if sns is not None:
            sns.boxplot(
                data=pd.DataFrame({"Algorithm": [algo] * len(delta), "Delta": delta.values}),
                x="Algorithm",
                y="Delta",
                color="#bdc3c7",
                ax=ax,
            )
        else:
            ax.boxplot([delta.values], labels=[algo])

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Δ SM Clock (MHz)")
        ax.set_title(self.plot_title)
        ax.grid(True, axis="y", alpha=self.theme.grid_alpha)


class EntropyMfuCollapsePlotter(BasePlotter):
    """Entropy vs MFU over step with dual y-axes."""

    plot_name = "entropy_mfu_collapse"
    plot_title = "Entropy vs MFU Collapse"

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
        iter_col = "step" if "step" in df.columns else "data.training/global_step"
        if iter_col not in df.columns:
            ax.set_title("Missing iteration column.")
            return

        x = pd.to_numeric(df[iter_col], errors="coerce")
        entropy = pd.to_numeric(df.get("data.actor/entropy"), errors="coerce")
        mfu = pd.to_numeric(df.get("data.perf/mfu/actor"), errors="coerce")

        mask = x.notna() & entropy.notna() & mfu.notna()
        x = x[mask]
        entropy = entropy[mask]
        mfu = mfu[mask]

        if x.empty:
            ax.set_title("No entropy/MFU data.")
            return

        ax.plot(x, entropy, color="#8e44ad", linewidth=1.5, label="Actor Entropy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Actor Entropy", color="#8e44ad")
        ax.tick_params(axis="y", labelcolor="#8e44ad")
        ax.grid(True, alpha=self.theme.grid_alpha)

        ax2 = ax.twinx()
        ax2.plot(x, mfu, color="#2980b9", linewidth=1.5, label="Actor MFU")
        ax2.set_ylabel("Actor MFU", color="#2980b9")
        ax2.tick_params(axis="y", labelcolor="#2980b9")

        ax.set_title(self.plot_title)


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
    )

    def create_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
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
