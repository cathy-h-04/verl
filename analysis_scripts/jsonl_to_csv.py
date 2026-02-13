#!/usr/bin/env python3
"""Convert selected JSONL files under a cleaned folder to CSV.

Targets:
- merged_sweep_*.jsonl

Writes CSV files next to each JSONL (same basename, .csv extension).
Optionally deletes the original JSONL to "replace" it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Set

# Default repo root (do not depend on current working directory)
DEFAULT_ROOT = Path("/home/cathxhou/projects/verl_research")
# Default cleaned folder name (used when no name/dir args are provided)
DEFAULT_CLEANED_DIR = "monitoring_small_cleaned"
TARGET_PATTERNS = (
    "merged_sweep_*.jsonl",
)

DEFAULT_DELETE_JSONL = True


def flatten_dict(d: Mapping[str, object], prefix: str = "") -> Dict[str, object]:
    """Flatten nested dictionaries using dot notation for nested keys."""
    flat: Dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            flat.update(flatten_dict(v, prefix=key))
        else:
            flat[key] = v
    return flat


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, Mapping):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj)}")
            yield flatten_dict(obj)


def collect_columns(rows: Iterable[Mapping[str, object]]) -> List[str]:
    cols: Set[str] = set()
    for row in rows:
        cols.update(row.keys())
    return sorted(cols)


def convert_one(jsonl_path: Path, delete_jsonl: bool) -> Path:
    rows = list(iter_jsonl(jsonl_path))
    if not rows:
        raise ValueError(f"No rows found in {jsonl_path}")

    columns = collect_columns(rows)
    csv_path = jsonl_path.with_suffix(".csv")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if delete_jsonl:
        jsonl_path.unlink()

    return csv_path


def find_targets(root: Path) -> List[Path]:
    targets: List[Path] = []
    for pattern in TARGET_PATTERNS:
        targets.extend(root.rglob(pattern))
    return sorted(targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert merged_sweep_*.jsonl under a cleaned folder to CSV."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Repo root containing monitoring folders (default: %(default)s).",
    )
    parser.add_argument(
        "--name",
        help="Base folder name. Uses <name>_cleaned under --root.",
    )
    parser.add_argument(
        "--cleaned-dir",
        help="Explicit cleaned directory (absolute or relative to --root).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--delete-jsonl",
        action="store_true",
        help="Delete JSONL after successful CSV conversion (default).",
    )
    group.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Keep JSONL files after conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if args.cleaned_dir:
        cleaned_root = Path(args.cleaned_dir).expanduser()
        cleaned_root = cleaned_root if cleaned_root.is_absolute() else root / cleaned_root
    elif args.name:
        cleaned_root = root / f"{args.name}_cleaned"
    else:
        cleaned_root = root / DEFAULT_CLEANED_DIR
    delete_jsonl = DEFAULT_DELETE_JSONL
    if args.keep_jsonl:
        delete_jsonl = False
    elif args.delete_jsonl:
        delete_jsonl = True

    if not cleaned_root.exists():
        print(f"❌ CLEANED_FOLDER does not exist: {cleaned_root}")
        return

    targets = find_targets(cleaned_root)
    if not targets:
        print(f"⚠ No target JSONL files found under {cleaned_root}")
        return

    print(f"Found {len(targets)} JSONL files to convert.")
    converted = 0

    for path in targets:
        try:
            csv_path = convert_one(path, delete_jsonl=delete_jsonl)
            converted += 1
            print(f"✓ {path.name} -> {csv_path.name}")
        except Exception as exc:
            print(f"✗ Failed to convert {path}: {exc}")

    print("-" * 80)
    print(f"Converted: {converted}/{len(targets)}")
    print(f"Output: CSV files next to originals under {cleaned_root}")
    if delete_jsonl:
        print("Original JSONL files were deleted after conversion.")
    else:
        print("Original JSONL files were kept after conversion.")


if __name__ == "__main__":
    main()
