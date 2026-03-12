"""Data loading, selectors, and manifest utilities for plots."""

from plots.data.loader import KNOWN_VIEWS, load_view
from plots.data.manifest import RunManifest, build_run_manifest, save_manifest, summarize_manifest
from plots.data.selectors import select_baseline, select_comparison_group, select_runs_by_ids

__all__ = [
    "KNOWN_VIEWS",
    "load_view",
    "RunManifest",
    "build_run_manifest",
    "save_manifest",
    "summarize_manifest",
    "select_runs_by_ids",
    "select_baseline",
    "select_comparison_group",
]
