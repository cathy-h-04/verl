#!/usr/bin/env python3
"""One-shot analysis pipeline runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script: python analysis_scripts/pipeline/main.py
if __package__ is None or __package__ == "":  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    __package__ = "analysis_scripts.pipeline"

from analysis_scripts import annotate_operations_in_csv, data_cleanup, jsonl_to_csv, merge_val_metrics


def _run_step(label: str, func, argv: list[str]) -> None:
    print(f"\n{'=' * 80}")
    print(f"Step: {label}")
    print(f"Args: {' '.join(argv) if argv else '(none)'}")
    print(f"{'=' * 80}")
    old_argv = sys.argv
    sys.argv = [label] + argv
    try:
        func()
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the analysis pipeline end-to-end."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Base folder name (e.g., monitoring_llama_qwen).",
    )
    parser.add_argument(
        "--root",
        default="/home/cathxhou/projects/verl_research",
        help="Repo root containing the monitoring folders.",
    )
    parser.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Keep merged_sweep_*.jsonl after conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name_args = ["--name", args.name, "--root", args.root]

    _run_step("data_cleanup", data_cleanup.main, name_args)
    _run_step("merge_val_metrics", merge_val_metrics.main, name_args)

    jsonl_args = name_args[:]
    if args.keep_jsonl:
        jsonl_args.append("--keep-jsonl")
    _run_step("jsonl_to_csv", jsonl_to_csv.main, jsonl_args)

    _run_step("annotate_operations_in_csv", annotate_operations_in_csv.main, name_args)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
