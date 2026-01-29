#!/usr/bin/env python3
"""Data loading + preprocessing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import EXCLUDED_OPERATIONS


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    annotated_csv: Path
    merged_sweep_csv: Optional[Path]
    cleaned_phase_timings: Optional[Path]
    annotated_df: pd.DataFrame
    merged_df: Optional[pd.DataFrame]
    timings_df: Optional[pd.DataFrame]

    @property
    def run_name(self) -> str:
        return self.run_dir.name


def _first_match(run_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def resolve_run_paths(run_dir: Path) -> Optional[RunPaths]:
    annotated = _first_match(run_dir, "annotated_*_phased_*.csv")
    if annotated is None:
        return None

    merged = _first_match(run_dir, "merged_sweep_*.csv")
    cleaned_timings = _first_match(run_dir, "cleaned_phase_timings_*.jsonl")

    annotated_df = add_gpu_derived_columns(load_annotated_csv(annotated))
    merged_df = load_merged_sweep_csv(merged)
    timings_df = load_cleaned_phase_timings_df(cleaned_timings)

    return RunPaths(
        run_dir=run_dir,
        annotated_csv=annotated,
        merged_sweep_csv=merged,
        cleaned_phase_timings=cleaned_timings,
        annotated_df=annotated_df,
        merged_df=merged_df,
        timings_df=timings_df,
    )


def discover_runs(root: Path) -> List[RunPaths]:
    runs: List[RunPaths] = []
    resolved = resolve_run_paths(root)
    if resolved is not None:
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


def add_gpu_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["memory_used_gb"] = pd.to_numeric(out.get("memory_used_mb"), errors="coerce") / 1024.0
    out["memory_total_gb"] = pd.to_numeric(out.get("memory_total_mb"), errors="coerce") / 1024.0
    out["memory_free_gb"] = out["memory_total_gb"] - out["memory_used_gb"]
    out["memory_usage_ratio"] = out["memory_used_gb"] / out["memory_total_gb"].replace(0, np.nan)

    out["elapsed_minutes"] = pd.to_numeric(out.get("elapsed_seconds"), errors="coerce") / 60.0

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
    if "timestamp_aligned_unix" not in df.columns:
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
