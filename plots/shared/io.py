"""Shared figure output and manifest helpers."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


FIGURES_ROOT = Path("plots/out/figures")


def build_output_paths(plot_module_name: str) -> tuple[Path, Path]:
    """Return deterministic PNG and manifest paths under plots/out/figures/<tier>/."""
    parts = plot_module_name.split(".")
    tier = parts[1] if len(parts) >= 3 else "misc"
    plot_name = parts[-1]

    output_dir = FIGURES_ROOT / tier
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{plot_name}.png", output_dir / f"{plot_name}.manifest.json"


def save_png(fig: Figure, png_path: Path) -> None:
    """Save the figure as a PNG."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
