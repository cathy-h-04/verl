"""Tiny loader for analysis-ready parquet views.

Design goals:
- 1-2 calls from plot files.
- No hidden filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path("DATASETS")
KNOWN_VIEWS = {
    "phase_fact_view",
    "step_fact_view",
    "run_summary_view",
    "comparison_view",
}

@dataclass(frozen=True)
class _ResolvedView:
    view_name: str
    view_path: Path


def load_view(
    view_name: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    columns: list[str] | tuple[str, ...] | None = None,
    row_filter: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one analysis-ready view with optional explicit filtering.

    Args:
        view_name: View name (e.g. "phase_fact_view") or parquet filename.
        dataset_root: Root containing view parquet files.
        columns: Optional selected columns.
        row_filter: Optional explicit filter applied after load.
            Supported types:
            - callable: fn(df) -> bool mask (Series/list), or filtered DataFrame
            - str: pandas query expression
            - dict[str, Any | list[Any] | tuple[Any, ...] | set[Any]]
    Returns:
        (df, metadata)
    """
    started = time.perf_counter()
    root = Path(dataset_root).expanduser().resolve()
    resolved = _resolve_view(view_name=view_name, dataset_root=root)

    dataset_version, dataset_version_source = _resolve_dataset_version(root)
    schema_version, schema_version_source, parquet_created_by = _resolve_schema_version(resolved.view_path)

    read_columns = list(columns) if columns is not None else None
    df = pd.read_parquet(resolved.view_path, columns=read_columns)
    df = _apply_row_filter(df=df, row_filter=row_filter)

    metadata: dict[str, Any] = {
        "view_name": resolved.view_name,
        "view_path": str(resolved.view_path),
        "dataset_root": str(root),
        "dataset_version": dataset_version,
        "dataset_version_source": dataset_version_source,
        "schema_version": schema_version,
        "schema_version_source": schema_version_source,
        "parquet_created_by": parquet_created_by,
        "selected_columns": read_columns,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "load_seconds": time.perf_counter() - started,
    }

    df.attrs["data_source"] = {
        "views": [resolved.view_name],
        "dataset_root": str(root),
        "view_path": str(resolved.view_path),
        "dataset_version": dataset_version,
        "schema_version": schema_version,
    }

    return df, metadata


def _resolve_view(view_name: str, dataset_root: Path) -> _ResolvedView:
    if view_name.endswith(".parquet"):
        path = dataset_root / view_name
        canonical = view_name[: -len(".parquet")]
    else:
        path = dataset_root / f"{view_name}.parquet"
        canonical = view_name

    if not path.exists():
        available = sorted(p.stem for p in dataset_root.glob("*_view.parquet"))
        raise FileNotFoundError(
            f"View '{view_name}' not found at {path}. "
            f"Available *_view files under {dataset_root}: {available}"
        )

    return _ResolvedView(view_name=canonical, view_path=path)


def _resolve_dataset_version(dataset_root: Path) -> tuple[str, str]:
    """Try explicit version files first; otherwise build a fingerprint."""
    explicit_candidates = [
        dataset_root / "dataset_version.txt",
        dataset_root / "DATASET_VERSION",
        dataset_root / "metadata.json",
        dataset_root / "manifest.json",
    ]

    for candidate in explicit_candidates:
        if not candidate.exists():
            continue

        if candidate.name in {"dataset_version.txt", "DATASET_VERSION"}:
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value, f"file:{candidate.name}"

        if candidate.suffix == ".json":
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            for key in ("dataset_version", "version"):
                if isinstance(payload.get(key), str) and payload[key].strip():
                    return payload[key].strip(), f"json:{candidate.name}:{key}"

    fingerprints: list[tuple[str, int, int]] = []
    for parquet_path in sorted(dataset_root.glob("*.parquet")):
        stat = parquet_path.stat()
        fingerprints.append((parquet_path.name, int(stat.st_size), int(stat.st_mtime_ns)))

    digest = sha256(repr(fingerprints).encode("utf-8")).hexdigest()[:16]
    return f"fingerprint:{digest}", "computed:fingerprint"


def _resolve_schema_version(view_path: Path) -> tuple[str, str, str | None]:
    metadata = pq.read_metadata(view_path)
    arrow_schema = metadata.schema.to_arrow_schema()

    # Prefer explicit schema version metadata if present.
    schema_meta = arrow_schema.metadata or {}
    for key in (b"schema_version", b"version"):
        raw = schema_meta.get(key)
        if raw:
            value = raw.decode("utf-8", errors="ignore").strip()
            if value:
                return value, f"parquet_metadata:{key.decode('utf-8')}", metadata.created_by

    schema_payload = {
        "fields": [(field.name, str(field.type), field.nullable) for field in arrow_schema],
        "metadata_keys": sorted((k.decode("utf-8", errors="ignore") for k in schema_meta.keys())),
    }
    schema_digest = sha256(repr(schema_payload).encode("utf-8")).hexdigest()[:16]
    return f"schema_hash:{schema_digest}", "computed:schema_hash", metadata.created_by


def _apply_row_filter(df: pd.DataFrame, row_filter: Any) -> pd.DataFrame:
    if row_filter is None:
        return df

    if isinstance(row_filter, str):
        return df.query(row_filter)

    if isinstance(row_filter, dict):
        for col, val in row_filter.items():
            if isinstance(val, (list, tuple, set)):
                df = df[df[col].isin(list(val))]
            else:
                df = df[df[col] == val]
        return df

    if callable(row_filter):
        result = row_filter(df)
        if isinstance(result, pd.DataFrame):
            return result
        return df[result]

    raise TypeError(
        "Unsupported row_filter type. Expected one of: None, str, dict, callable. "
        f"Got {type(row_filter).__name__}."
    )
