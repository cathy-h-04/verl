#!/usr/bin/env python3
"""CLI entrypoint for modular graphing suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script: python analysis_scripts/graphing/main.py
if __package__ is None or __package__ == "":  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    __package__ = "analysis_scripts.graphing"
from typing import Iterable, List, Mapping

from .core.base import ThemeConfig
from .core.loaders import discover_runs
from .plotters.hardware import GPUOverviewPlotter, PhaseComputeDensityPlotter, SmoothedTimeSeriesPlotter, ThermalSteadyStatePlotter
from .plotters.timing import (
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
from .plotters.efficiency import (
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


CLEANED_ROOT = Path("monitoring_small_cleaned")
DEFAULTS_NOTICE = "No arguments provided; using defaults: --root monitoring_small_cleaned --output-dir plots"

PLOTTERS: Mapping[str, type] = {
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

DEFAULT_PLOTS = (
    "phase_focus_rollout,phase_focus_rl_policy,phase_focus_training,"
    "phase_energy_time_stacked,smoothed_timeseries,phase_boxplots,phase_correlations,"
    "phase_aggregate,thermal_steady_state,phase_compute_density,hierarchical_waterfall,"
    "mfu_comparison,throughput_vs_length,throughput_reward_frontier,hardware_roi,"
    "learning_price,token_micro_bottlenecks,bottleneck_evolution"
)


def build_plotters(run_paths, output_dir: Path, plot_names: Iterable[str], theme: ThemeConfig):
    plotters: List = []
    for name in plot_names:
        cls = PLOTTERS.get(name)
        if cls is None:
            raise ValueError(f"Unknown plot '{name}'. Use --list-plots to see available options.")
        plotters.append(cls(run_paths=run_paths, output_dir=output_dir, theme=theme))
    return plotters


def default_output_dir(root_output: Path, run_paths) -> Path:
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
        default=DEFAULT_PLOTS,
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
    if len(sys.argv) == 1:
        print(DEFAULTS_NOTICE)

    if args.list_plots:
        print("Available plots:")
        default_plots = [p.strip() for p in str(DEFAULT_PLOTS).split(",") if p.strip()]
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
            except Exception as exc:  # pragma: no cover
                print(f"  ✗ {plotter.plot_name}: {exc}")


if __name__ == "__main__":
    main()
