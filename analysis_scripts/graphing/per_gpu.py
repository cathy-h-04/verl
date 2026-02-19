#!/usr/bin/env python3
"""Generate quad-view plots per GPU for cleaned monitoring runs."""

from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd

# Allow running as a script: python analysis_scripts/graphing/per_gpu.py
if __package__ is None or __package__ == "":  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    __package__ = "analysis_scripts.graphing"

from .core.base import ThemeConfig
from .core.loaders import RunPaths, discover_runs, load_annotated_csv, add_gpu_derived_columns
from .core.style_config import load_style_config
from .main import DEFAULT_PLOTS, PLOTTERS, resolve_paths, _plots_label
from .analytics import comparisons as comp


def _discover_gpu_files(run_dir: Path) -> dict[int, Path]:
    gpu_files: dict[int, Path] = {}
    for path in sorted(run_dir.glob("annotated_*_gpu*.csv")):
        stem = path.stem
        if "gpu" not in stem:
            continue
        gpu_suffix = stem.split("gpu")[-1]
        try:
            gpu_id = int(gpu_suffix)
        except ValueError:
            continue
        gpu_files[gpu_id] = path
    return gpu_files


def _normalize_gpu_ids(runs: Sequence[RunPaths]) -> List[int]:
    gpu_ids = set()
    for run in runs:
        gpu_files = _discover_gpu_files(run.run_dir)
        if gpu_files:
            gpu_ids.update(gpu_files.keys())
            continue
        df = run.annotated_df
        if df is None or "gpu_id" not in df.columns:
            continue
        ids = pd.to_numeric(df["gpu_id"], errors="coerce").dropna().unique().tolist()
        for val in ids:
            try:
                gpu_ids.add(int(val))
            except (TypeError, ValueError):
                continue
    return sorted(gpu_ids)


def _filter_runs_for_gpu(runs: Sequence[RunPaths], gpu_id: int) -> List[RunPaths]:
    filtered: List[RunPaths] = []
    for run in runs:
        gpu_files = _discover_gpu_files(run.run_dir)
        if gpu_files and gpu_id in gpu_files:
            annotated_df = add_gpu_derived_columns(load_annotated_csv(gpu_files[gpu_id]))
            filtered.append(
                RunPaths(
                    run_dir=run.run_dir,
                    annotated_csv=gpu_files[gpu_id],
                    merged_sweep_csv=run.merged_sweep_csv,
                    cleaned_phase_timings=run.cleaned_phase_timings,
                    annotated_df=annotated_df,
                    merged_df=run.merged_df,
                    timings_df=run.timings_df,
                )
            )
            continue

        df = run.annotated_df
        if df is None or "gpu_id" not in df.columns:
            continue
        mask = pd.to_numeric(df["gpu_id"], errors="coerce") == gpu_id
        if not mask.any():
            continue
        df_gpu = df.loc[mask].copy()
        filtered.append(
            RunPaths(
                run_dir=run.run_dir,
                annotated_csv=run.annotated_csv,
                merged_sweep_csv=run.merged_sweep_csv,
                cleaned_phase_timings=run.cleaned_phase_timings,
                annotated_df=df_gpu,
                merged_df=run.merged_df,
                timings_df=run.timings_df,
            )
        )
    return filtered


def _generate_quad_views(
    runs: Sequence[RunPaths],
    output_dir: Path,
    plotter_classes: Optional[Mapping[str, type]] = None,
    theme: Optional[ThemeConfig] = None,
    style_config: Optional[comp.RunStyleConfig] = None,
) -> List[Path]:
    run_list = sorted(list(runs), key=comp._quad_run_sort_key)
    if not run_list:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    theme = theme or ThemeConfig()

    available_plot_names = None
    if plotter_classes:
        available_plot_names = {cls.plot_name for cls in plotter_classes.values()}

    prev_shape = comp._QUAD_GRID_SHAPE
    comp._QUAD_GRID_SHAPE = comp._grid_dims(len(run_list))

    outputs: List[Path] = []
    try:
        for plot_name, handler in comp._quad_plotters().items():
            if available_plot_names is not None and plot_name not in available_plot_names:
                continue
            try:
                if style_config is not None and "style_config" in inspect.signature(handler).parameters:
                    out_path = handler(run_list, output_dir, theme, style_config)
                else:
                    out_path = handler(run_list, output_dir, theme)
            except Exception as exc:  # pragma: no cover
                print(f"Quad view failed for {plot_name}: {exc}")
                continue
            if out_path is not None:
                outputs.append(out_path)
    finally:
        comp._QUAD_GRID_SHAPE = prev_shape

    return outputs


def _suffix_outputs(temp_dir: Path, dest_dir: Path, gpu_id: int) -> None:
    for path in sorted(temp_dir.glob("*.png")):
        stem = path.stem
        dest = dest_dir / f"{stem}_gpu{gpu_id}.png"
        if dest.exists():
            dest.unlink()
        shutil.move(str(path), str(dest))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-GPU quad-view plots for cleaned monitoring runs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/cathxhou/projects/verl_research"),
        help="Repo root (default: /home/cathxhou/projects/verl_research).",
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
        default=Path("plots"),
        help="Root output directory for generated graphs (default: plots_<cleaned_dir_name>).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_names = [p.strip() for p in str(args.plots).split(",") if p.strip()]

    cleaned_root, output_dir = resolve_paths(args)
    style_config = None
    style_config_path = args.style_config
    if style_config_path is None:
        style_schemes = Path(__file__).resolve().parent / "style_schemes"
        label = _plots_label(cleaned_root)
        candidate = style_schemes / f"{label}.json"
        if candidate.exists():
            style_config_path = candidate
    if style_config_path:
        style_config = load_style_config(style_config_path)

    runs = discover_runs(cleaned_root)
    if not runs:
        raise SystemExit(f"No cleaned runs found under: {cleaned_root}")

    gpu_ids = _normalize_gpu_ids(runs)
    if len(gpu_ids) <= 1:
        print("Per-GPU plots skipped: only one GPU found in logs.")
        return

    per_gpu_runs = {gpu_id: _filter_runs_for_gpu(runs, gpu_id) for gpu_id in gpu_ids}
    bounds_runs: List[RunPaths] = []
    for run_list in per_gpu_runs.values():
        bounds_runs.extend(run_list)

    per_gpu_dir = output_dir / "per_GPU"
    per_gpu_dir.mkdir(parents=True, exist_ok=True)

    plotter_map = {k: v for k, v in PLOTTERS.items() if k in plot_names}

    comp._BOUNDS_REFERENCE_RUNS = bounds_runs if bounds_runs else None
    for gpu_id in gpu_ids:
        filtered = per_gpu_runs.get(gpu_id, [])
        if not filtered:
            continue
        temp_dir = per_gpu_dir / f"tmp_gpu{gpu_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        _generate_quad_views(
            filtered,
            temp_dir,
            plotter_classes=plotter_map,
            theme=ThemeConfig(),
            style_config=style_config,
        )
        _suffix_outputs(temp_dir, per_gpu_dir, gpu_id)
        for leftover in temp_dir.iterdir():
            leftover.unlink()
        temp_dir.rmdir()
    comp._BOUNDS_REFERENCE_RUNS = None


if __name__ == "__main__":
    main()
