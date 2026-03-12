"""Minimal run-manifest objects for reproducible plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RunManifest:
    """Minimal manifest: run set + data provenance."""

    plot_name: str
    created_at_utc: str
    run_ids: list[str]
    data_sources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plot_name": self.plot_name,
            "created_at_utc": self.created_at_utc,
            "run_ids": list(self.run_ids),
            "data_sources": dict(self.data_sources),
        }


def build_run_manifest(
    *,
    plot_name: str,
    run_rows: Sequence[Mapping[str, Any]] | None = None,
    run_ids: Sequence[str] | None = None,
    data_sources: Mapping[str, Any] | None = None,
) -> RunManifest:
    """Build a minimal RunManifest from rows and/or explicit run IDs."""
    row_ids = [str(row["run_id"]) for row in (run_rows or []) if "run_id" in row]
    explicit_ids = [str(run_id) for run_id in (run_ids or [])]
    merged_ids = list(dict.fromkeys([*row_ids, *explicit_ids]))

    return RunManifest(
        plot_name=plot_name,
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        run_ids=merged_ids,
        data_sources=dict(data_sources or {}),
    )


def save_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Save a RunManifest as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_manifest(manifest: RunManifest) -> str:
    """Print and return a concise summary line."""
    views = manifest.data_sources.get("views", [])
    view_text = ",".join(str(v) for v in views) if views else "none"
    summary = f"[run-manifest] plot={manifest.plot_name} runs={len(manifest.run_ids)} views={view_text}"
    print(summary)
    return summary

