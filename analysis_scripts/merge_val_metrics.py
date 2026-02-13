#!/usr/bin/env python3
"""Merge validation-only metrics into sweep logs.

This script matches runs between data and val folders by a stable prefix
(ignoring the trailing timestamp), merges records by `step`, and writes
merged sweep JSONL files under a cleaned output folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Default repo root (do not depend on current working directory)
DEFAULT_ROOT = Path("/home/cathxhou/projects/verl_research")
# Default monitoring directory names (used when no name/dir args are provided)
DEFAULT_DATA_DIR = "monitoring_small_data"
DEFAULT_VAL_DIR = "monitoring_small_val"
DEFAULT_OUT_DIR = "monitoring_small_cleaned"

# Validation-only metrics to inject (found via audit)
VAL_EXTRA_KEYS = {
    "val-core/openai/gsm8k/reward/mean@1",
    "timing_s/testing",
}



@dataclass(frozen=True)
class SweepRun:
    prefix: str
    run_dir: Path
    sweep_file: Path


def _is_timestamp_pair(parts: Sequence[str]) -> bool:
    """Return True if the trailing parts look like YYYYMMDD_HHMMSS."""
    if len(parts) < 2:
        return False
    date_part, time_part = parts[-2], parts[-1]
    return (
        len(date_part) == 8
        and len(time_part) == 6
        and date_part.isdigit()
        and time_part.isdigit()
    )


def run_prefix(run_dir_name: str) -> str:
    """Strip the trailing timestamp segment(s) to get a stable prefix."""
    parts = run_dir_name.split("_")
    if _is_timestamp_pair(parts):
        return "_".join(parts[:-2])
    # Fallback: if the last segment is numeric, drop it.
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return run_dir_name


def find_sweep_runs(root: Path) -> List[SweepRun]:
    """Find sweep_*.jsonl runs one level below the given root."""
    runs: List[SweepRun] = []
    for sweep_file in sorted(root.rglob("sweep_*.jsonl")):
        run_dir = sweep_file.parent
        prefix = run_prefix(run_dir.name)
        runs.append(SweepRun(prefix=prefix, run_dir=run_dir, sweep_file=sweep_file))
    return runs


def index_by_prefix(runs: Iterable[SweepRun]) -> Dict[str, SweepRun]:
    """Index runs by prefix, skipping ambiguous duplicates."""
    by_prefix: Dict[str, SweepRun] = {}
    duplicates: Dict[str, List[Path]] = {}
    for run in runs:
        if run.prefix in by_prefix:
            duplicates.setdefault(run.prefix, [by_prefix[run.prefix].run_dir]).append(
                run.run_dir
            )
            continue
        by_prefix[run.prefix] = run

    if duplicates:
        print("⚠ Ambiguous prefixes detected; these will be skipped:")
        for prefix, dirs in sorted(duplicates.items()):
            dir_list = ", ".join(str(d) for d in dirs)
            print(f"  - {prefix}: {dir_list}")
            by_prefix.pop(prefix, None)

    return by_prefix


def load_sweep_by_step(path: Path) -> Tuple[List[int], Dict[int, dict]]:
    """Load a sweep JSONL into an ordered step list and a step->record map."""
    order: List[int] = []
    by_step: Dict[int, dict] = {}
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = record.get("step")
            if not isinstance(step, int):
                raise ValueError(f"Missing/invalid step at {path}:{line_no}")
            order.append(step)
            by_step[step] = record
    return order, by_step


def extract_val_extras(val_record: Mapping[str, object]) -> Dict[str, object]:
    """Pull the validation-only metrics from a val sweep record."""
    data = val_record.get("data")
    if not isinstance(data, Mapping):
        return {}
    return {k: data[k] for k in VAL_EXTRA_KEYS if k in data}


def _normalize_dir(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def resolve_paths(
    root: Path,
    name: str | None,
    data_dir: str | None,
    val_dir: str | None,
    out_dir: str | None,
) -> tuple[Path, Path, Path]:
    """
    Resolve data/val/out directories with precedence:
      1) explicit --data-dir/--val-dir/--out-dir
      2) --name (data=<name>, val=<name>_val, out=<name>_cleaned)
      3) defaults (monitoring_small_data / monitoring_small_val / monitoring_small_cleaned)
    """
    if data_dir or val_dir or out_dir:
        data_root = _normalize_dir(root, data_dir) if data_dir else root / DEFAULT_DATA_DIR
        val_root = _normalize_dir(root, val_dir) if val_dir else root / DEFAULT_VAL_DIR
        out_root = _normalize_dir(root, out_dir) if out_dir else root / DEFAULT_OUT_DIR
        return data_root, val_root, out_root
    if name:
        return root / name, root / f"{name}_val", root / f"{name}_cleaned"
    return root / DEFAULT_DATA_DIR, root / DEFAULT_VAL_DIR, root / DEFAULT_OUT_DIR


def _copy_if_exists(src: Path, dst_dir: Path) -> bool:
    """Copy a file into dst_dir if it exists. Returns True if copied."""
    if not src.exists():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    return True


def copy_run_artifacts(
    data_run: SweepRun,
    data_root: Path,
    out_dir: Path,
) -> None:
    """Copy config and copy phased CSV into the output run folder."""
    config_file = data_run.run_dir / f"{data_run.run_dir.name}_config.json"
    cleaned_timings_name = f"cleaned_phase_timings_{data_run.run_dir.name}.jsonl"
    cleaned_timings_out = out_dir / cleaned_timings_name

    copied_config = _copy_if_exists(config_file, out_dir)
    copied_cleaned_timings = cleaned_timings_out.exists()

    phased_csv_candidates = sorted(
        data_run.run_dir.glob(f"{data_run.run_dir.name}_phased_*.csv")
    )
    if not phased_csv_candidates:
        phased_csv_candidates = sorted(
            data_root.glob(f"{data_run.run_dir.name}_phased_*.csv")
        )
    copied_csv = False
    if phased_csv_candidates:
        csv_src = phased_csv_candidates[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_src, out_dir / csv_src.name)
        copied_csv = True

    print(
        "  artifacts:"
        f" config={'yes' if copied_config else 'no'},"
        f" cleaned_phase_timings_present={'yes' if copied_cleaned_timings else 'no'},"
        f" phased_csv_copied={'yes' if copied_csv else 'no'}"
    )


def merge_run(
    data_run: SweepRun,
    val_run: SweepRun,
    data_root: Path,
    out_root: Path,
) -> Tuple[int, int]:
    """Merge one matched data/val run and write to the output tree."""
    data_order, data_by_step = load_sweep_by_step(data_run.sweep_file)
    _, val_by_step = load_sweep_by_step(val_run.sweep_file)

    out_dir = out_root / data_run.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"merged_{data_run.sweep_file.name}"

    injected_steps = 0
    total_steps = len(data_order)

    with out_file.open("w") as fout:
        for step in data_order:
            merged = json.loads(json.dumps(data_by_step[step]))
            val_record = val_by_step.get(step)
            if val_record is not None:
                extras = extract_val_extras(val_record)
                if extras:
                    merged.setdefault("data", {})
                    if isinstance(merged["data"], dict):
                        merged["data"].update(extras)
                        injected_steps += 1
            fout.write(json.dumps(merged) + "\n")

    copy_run_artifacts(
        data_run,
        data_root=data_root,
        out_dir=out_dir,
    )
    return injected_steps, total_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge validation-only metrics into data sweeps and emit cleaned runs."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Repo root containing monitoring folders (default: %(default)s).",
    )
    parser.add_argument(
        "--name",
        help="Base folder name. Uses <name>, <name>_val, and <name>_cleaned under --root.",
    )
    parser.add_argument(
        "--data-dir",
        help="Explicit data directory (absolute or relative to --root).",
    )
    parser.add_argument(
        "--val-dir",
        help="Explicit val directory (absolute or relative to --root).",
    )
    parser.add_argument(
        "--out-dir",
        help="Explicit output directory (absolute or relative to --root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    data_root, val_root, out_root = resolve_paths(
        root=root,
        name=args.name,
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        out_dir=args.out_dir,
    )

    data_runs = find_sweep_runs(data_root)
    val_runs = find_sweep_runs(val_root)

    data_by_prefix = index_by_prefix(data_runs)
    val_by_prefix = index_by_prefix(val_runs)

    matched_prefixes = sorted(set(data_by_prefix) & set(val_by_prefix))
    missing_in_val = sorted(set(data_by_prefix) - set(val_by_prefix))
    missing_in_data = sorted(set(val_by_prefix) - set(data_by_prefix))

    print(f"Matched prefixes: {len(matched_prefixes)}")
    if missing_in_val:
        print("⚠ Prefixes missing in val:")
        for prefix in missing_in_val:
            print(f"  - {prefix}")
    if missing_in_data:
        print("⚠ Prefixes missing in data:")
        for prefix in missing_in_data:
            print(f"  - {prefix}")

    merged_files = 0
    injected_total = 0
    steps_total = 0

    for prefix in matched_prefixes:
        data_run = data_by_prefix[prefix]
        val_run = val_by_prefix[prefix]
        injected_steps, total_steps = merge_run(
            data_run,
            val_run,
            data_root=data_root,
            out_root=out_root,
        )
        merged_files += 1
        injected_total += injected_steps
        steps_total += total_steps
        print(
            f"✓ {prefix}: injected val metrics into {injected_steps}/{total_steps} steps"
        )

    print("-" * 80)
    print(f"Merged files: {merged_files}")
    print(f"Steps with injected val metrics: {injected_total}/{steps_total}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
