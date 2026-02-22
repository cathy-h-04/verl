#!/usr/bin/env python3
"""CLI entrypoint for modular graphing suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import os
import getpass
import shutil
import sys
from pathlib import Path

# Allow running as a script: python analysis_scripts/graphing/main.py
if __package__ is None or __package__ == "":  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    __package__ = "analysis_scripts.graphing"
from typing import Iterable, List, Mapping, Union

from .core.base import ThemeConfig
from .core.style_config import load_style_config
from .core.loaders import discover_runs
from .plotters.hardware import (
    GPUOverviewPlotter,
    PhaseComputeDensityPlotter,
    SmoothedTimeSeriesPlotter,
    ThermalSteadyStatePlotter,
    EnergyVsSMClockPlotter,
    PstateDistributionPlotter,
    UtilizationSmClockScatterPlotter,
    UtilizationMemoryScatterPlotter,
    PhaseFingerprintPlotter,
)
from .plotters.timing import (
    HierarchicalWaterfallPlotter,
    PhaseAggregatePlotter,
    PhaseAggregateMemoryClockPlotter,
    PhaseAggregateSMClockPlotter,
    PhasePeakPowerPlotter,
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
    ClockImbalanceMapPlotter,
    BoostHeadroomHistogramPlotter,
    PhaseToPhaseStabilityBoxplot,
    CriticIntensityMapPlotter,
    EnergyPerTokenPlotter,
    EntropyMfuCollapsePlotter,
    HardwareROIPlotter,
    LearningPricePlotter,
    MFUComparisonPlotter,
    OperationAggregatePlotter,
    OperationComparisonPlotter,
    SweepMetricsPlotter,
    TailLatencyTaxMapPlotter,
    ThroughputRewardFrontierPlotter,
    ThroughputVsLengthPlotter,
    TokenBottlenecksPlotter,
)
from .analytics.comparisons import generate_quad_views


DEFAULT_REPO_ROOT = Path("/n/netscratch/yu_lab/Lab/chou")
DEFAULT_CLEANED_DIR = "monitoring_llama_qwen_cleaned"
DEFAULT_OUTPUT_DIR = "plots"
DEFAULT_ARCHIVE_ROOT = Path("/n/home08/chou/verl_research")
ALLOWED_SCRATCH_ROOT = Path("/n/netscratch/yu_lab/Lab/chou").resolve()
STAGING_DIRNAME = "temp_plots"
STAGE_OWNER_FILE = ".stage_owner"
DEFAULTS_NOTICE = (
    "No arguments provided; using defaults: "
    "--root /n/netscratch/yu_lab/Lab/chou "
    "--cleaned-dir monitoring_llama_qwen_cleaned "
    "--output-dir /n/home08/chou/verl_research"
)

PLOTTERS = {
    "overview": GPUOverviewPlotter,
    "phase_timeline": PhaseTimelinePlotter,
    "phase_aggregate": PhaseAggregatePlotter,
    "phase_sm_clock": PhaseAggregateSMClockPlotter,
    "phase_memory_clock": PhaseAggregateMemoryClockPlotter,
    "phase_peak_power": PhasePeakPowerPlotter,
    "phase_focus_rollout": PhaseFocusRolloutPlotter,
    "phase_focus_rl_policy": PhaseFocusRLPolicyPlotter,
    "phase_focus_training": PhaseFocusTrainingPlotter,
    "phase_energy_time_stacked": PhaseEnergyTimeStackedPlotter,
    "phase_boxplots": PhaseBoxplotPlotter,
    "phase_correlations": PhaseCorrelationPlotter,
    "smoothed_timeseries": SmoothedTimeSeriesPlotter,
    "thermal_steady_state": ThermalSteadyStatePlotter,
    "phase_compute_density": PhaseComputeDensityPlotter,
    "energy_vs_sm_clock": EnergyVsSMClockPlotter,
    "pstate_distribution": PstateDistributionPlotter,
    "util_sm_clock_scatter": UtilizationSmClockScatterPlotter,
    "util_memory_scatter": UtilizationMemoryScatterPlotter,
    "phase_fingerprint": PhaseFingerprintPlotter,
    "hierarchical_waterfall": HierarchicalWaterfallPlotter,
    "mfu_comparison": MFUComparisonPlotter,
    "throughput_vs_length": ThroughputVsLengthPlotter,
    "throughput_reward_frontier": ThroughputRewardFrontierPlotter,
    "hardware_roi": HardwareROIPlotter,
    "learning_price": LearningPricePlotter,
    "energy_per_token": EnergyPerTokenPlotter,
    "entropy_mfu_collapse": EntropyMfuCollapsePlotter,
    "token_micro_bottlenecks": TokenBottlenecksPlotter,
    "bottleneck_evolution": BottleneckEvolutionPlotter,
    "sweep_metrics": SweepMetricsPlotter,
    "tail_latency_tax_map": TailLatencyTaxMapPlotter,
    "clock_imbalance_map": ClockImbalanceMapPlotter,
    "boost_headroom_histogram": BoostHeadroomHistogramPlotter,
    "phase_to_phase_stability": PhaseToPhaseStabilityBoxplot,
    "critic_intensity_map": CriticIntensityMapPlotter,
    "operation_aggregate": OperationAggregatePlotter,
    "operation_comparison": OperationComparisonPlotter,
}

DEFAULT_PLOTS = (
    "phase_focus_rollout,phase_focus_rl_policy,phase_focus_training,"
    "phase_energy_time_stacked,smoothed_timeseries,phase_boxplots,phase_correlations,"
    "phase_aggregate,thermal_steady_state,phase_compute_density,energy_vs_sm_clock,hierarchical_waterfall,"
    "pstate_distribution,util_sm_clock_scatter,util_memory_scatter,phase_fingerprint,"
    "mfu_comparison,throughput_vs_length,throughput_reward_frontier,hardware_roi,"
    "learning_price,energy_per_token,entropy_mfu_collapse,token_micro_bottlenecks,bottleneck_evolution,tail_latency_tax_map,clock_imbalance_map,boost_headroom_histogram,critic_intensity_map,phase_to_phase_stability,"
    "phase_sm_clock,phase_memory_clock,phase_peak_power"
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


def _normalize_dir(root: Path, raw: Union[str, Path]) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def _looks_like_cleaned_root(path: Path) -> bool:
    if not path.exists():
        return False
    if path.name.endswith("_cleaned"):
        return True
    if any(path.glob("annotated_*_phased_*.csv")):
        return True
    for child in path.iterdir():
        if child.is_dir() and any(child.glob("annotated_*_phased_*.csv")):
            return True
    return False


def _plots_label(cleaned_root: Path) -> str:
    name = cleaned_root.name
    if name.startswith("monitoring_"):
        name = name[len("monitoring_") :]
    if name.endswith("_cleaned"):
        name = name[: -len("_cleaned")]
    return name or cleaned_root.name


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.root).expanduser().resolve()

    if args.cleaned_dir:
        cleaned_root = _normalize_dir(root, args.cleaned_dir)
        repo_root = root
    elif args.name:
        cleaned_root = root / f"{args.name}_cleaned"
        repo_root = root
    else:
        if _looks_like_cleaned_root(root):
            cleaned_root = root
            repo_root = root.parent
        else:
            cleaned_root = root / DEFAULT_CLEANED_DIR
            repo_root = root

    if args.output_dir != Path(DEFAULT_OUTPUT_DIR):
        raw_output = Path(args.output_dir).expanduser()
        output_dir = raw_output if raw_output.is_absolute() else DEFAULT_ARCHIVE_ROOT / raw_output
    else:
        output_dir = DEFAULT_ARCHIVE_ROOT
    return cleaned_root, output_dir


def _stage_dir(cleaned_root: Path) -> Path:
    cleaned_root = cleaned_root.resolve()
    if ALLOWED_SCRATCH_ROOT not in cleaned_root.parents and cleaned_root != ALLOWED_SCRATCH_ROOT:
        raise SystemExit(f"Refusing to stage outside {ALLOWED_SCRATCH_ROOT}: {cleaned_root}")
    return cleaned_root / STAGING_DIRNAME


def _ensure_stage_dir(cleaned_root: Path) -> Path:
    stage_dir = _ensure_stage_dir(cleaned_root)

    owner_path = stage_dir / STAGE_OWNER_FILE
    user = getpass.getuser()
    if owner_path.exists():
        existing_owner = owner_path.read_text().strip()
        if existing_owner != user:
            raise SystemExit(
                f"Refusing to use stage dir owned by '{existing_owner}': {stage_dir}"
            )
        # Clean only if owned by current user.
        for entry in stage_dir.iterdir():
            if entry.name == STAGE_OWNER_FILE:
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    else:
        owner_path.write_text(user)

    return stage_dir


def _same_filesystem(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.stat().st_dev == path_b.stat().st_dev
    except FileNotFoundError:
        return False


def _atomic_archive_move(src_dir: Path, dest_dir: Path) -> Path:
    dest_parent = dest_dir.parent
    dest_parent.mkdir(parents=True, exist_ok=True)

    if _same_filesystem(src_dir, dest_parent):
        os.replace(src_dir, dest_dir)
        return dest_dir

    tmp_name = f".{dest_dir.name}.tmp_{os.getpid()}"
    tmp_dir = dest_parent / tmp_name
    if tmp_dir.exists():
        raise SystemExit(f"Refusing to remove existing temp dir: {tmp_dir}")
    shutil.copytree(src_dir, tmp_dir)
    os.replace(tmp_dir, dest_dir)
    shutil.rmtree(src_dir)
    return dest_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular graphing suite for cleaned monitoring runs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Scratch root (default: /n/netscratch/yu_lab/Lab/chou).",
    )
    parser.add_argument(
        "--name",
        help="Base folder name. Uses <name>_cleaned under --root.",
    )
    parser.add_argument(
        "--cleaned-dir",
        help="Explicit cleaned directory (absolute or relative to --root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Archive root for generated graphs (default: /n/home08/chou/verl_research).",
    )
    parser.add_argument(
        "--style-config",
        type=Path,
        help="Optional JSON file defining run-specific colors/linestyles/hatches.",
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

    cleaned_root, archive_root = resolve_paths(args)
    style_config = None
    style_config_path = args.style_config
    if style_config_path is None:
        style_schemes = Path(__file__).resolve().parent / "style_schemes"
        label = _plots_label(cleaned_root)
        candidate = style_schemes / f"{label}.json"
        if candidate.exists():
            style_config_path = candidate
    if style_config_path:
        try:
            style_config = load_style_config(style_config_path)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    runs = discover_runs(cleaned_root)
    if not runs:
        raise SystemExit(f"No cleaned runs found under: {cleaned_root}")

    theme = ThemeConfig()

    stage_dir = _stage_dir(cleaned_root)
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(runs)} run(s) under {cleaned_root}")

    quad_outputs = generate_quad_views(
        runs,
        stage_dir,
        plotter_classes=PLOTTERS,
        theme=theme,
        style_config=style_config,
    )
    if quad_outputs:
        print(f"\nQuad-view plots saved to: {stage_dir / 'quad_plots'}")

    import matplotlib.pyplot as plt

    plt.close("all")

    label = _plots_label(cleaned_root)
    archive_dir = archive_root / f"plots_{label}"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archived = _atomic_archive_move(stage_dir, archive_dir)
    print(f"\nArchived plots to: {archived}")


if __name__ == "__main__":
    main()
