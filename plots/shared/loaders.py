"""Load run metadata for plotting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_MONITORING_ROOT = Path("results/monitoring_val")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    run_config_path: Path | None
    tokens_and_steps_path: Path | None
    primary_metrics_path: Path | None
    policy: str | None
    model: str | None
    lineage_root_run_id: str | None
    logical_run_group: str | None

    def to_manifest_dict(self) -> dict[str, str | None]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "run_config_path": str(self.run_config_path) if self.run_config_path else None,
            "tokens_and_steps_path": str(self.tokens_and_steps_path) if self.tokens_and_steps_path else None,
            "primary_metrics_path": str(self.primary_metrics_path) if self.primary_metrics_path else None,
            "policy": self.policy,
            "model": self.model,
            "lineage_root_run_id": self.lineage_root_run_id,
            "logical_run_group": self.logical_run_group,
        }


def load_monitoring_runs(monitoring_root: Path | str = DEFAULT_MONITORING_ROOT) -> list[RunRecord]:
    """Load run records from results/monitoring_val-like folders."""
    root = Path(monitoring_root)
    if not root.exists():
        return []

    runs: list[RunRecord] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        run_id = run_dir.name
        run_config_path = run_dir / "run_config.json"
        tokens_and_steps_path = run_dir / "tokens_and_steps.jsonl"

        run_config = run_config_path if run_config_path.exists() else None
        tokens_and_steps = tokens_and_steps_path if tokens_and_steps_path.exists() else None
        primary_metrics = _resolve_primary_metrics_path(run_dir=run_dir, run_id=run_id, tokens_and_steps=tokens_and_steps)
        run_meta = _read_run_metadata(run_config)

        runs.append(
            RunRecord(
                run_id=run_id,
                run_dir=run_dir,
                run_config_path=run_config,
                tokens_and_steps_path=tokens_and_steps,
                primary_metrics_path=primary_metrics,
                policy=run_meta["policy"],
                model=run_meta["model"],
                lineage_root_run_id=run_meta["lineage_root_run_id"],
                logical_run_group=run_meta["logical_run_group"],
            )
        )
    return runs


def _resolve_primary_metrics_path(run_dir: Path, run_id: str, tokens_and_steps: Path | None) -> Path | None:
    preferred = run_dir / f"{run_id}.jsonl"
    if preferred.exists():
        return preferred
    if tokens_and_steps is not None:
        return tokens_and_steps

    jsonl_files = sorted(run_dir.glob("*.jsonl"))
    return jsonl_files[0] if jsonl_files else None


def _read_run_metadata(run_config_path: Path | None) -> dict[str, str | None]:
    if run_config_path is None:
        return {
            "policy": None,
            "model": None,
            "lineage_root_run_id": None,
            "logical_run_group": None,
        }
    try:
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "policy": None,
            "model": None,
            "lineage_root_run_id": None,
            "logical_run_group": None,
        }

    run_block = payload.get("run", {})
    meta_block = payload.get("meta", {})
    lineage_root = run_block.get("lineage_root_run_id") or meta_block.get("lineage_root_run_id")
    logical_group = run_block.get("logical_run_group") or meta_block.get("logical_run_group")

    return {
        "policy": run_block.get("policy"),
        "model": run_block.get("model"),
        "lineage_root_run_id": lineage_root,
        "logical_run_group": logical_group,
    }
