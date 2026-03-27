"""Alias entrypoint for the older actor-memory non-results filename."""

from __future__ import annotations

from pathlib import Path

import plots.scale.actor_memory_metrics as target


def main() -> None:
    target.OUTPATH = Path("plots/out/scale/non_results/grpo_update_actor_ms_per_train_token_vs_mem.png")
    target.MANIFEST_PATH = target.OUTPATH.with_suffix(".manifest.json")
    target.main()


if __name__ == "__main__":
    main()
