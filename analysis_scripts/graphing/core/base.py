#!/usr/bin/env python3
"""Shared graphing configuration and base classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional
    sns = None


# Centralized color choices so they can be reused across plot types.
PHASE_COLORS: Dict[str, str] = {
    "rollout": "#2ecc71",
    "rl_policy": "#f39c12",
    "training": "#3498db",
    "idle": "#95a5a6",
    "unknown": "#7f8c8d",
}

OPERATION_COLORS: Dict[str, str] = {
    "generate_sequences": "#1abc9c",
    "gen": "#2ecc71",
    "gen_max": "#27ae60",
    "reward": "#f39c12",
    "old_log_prob": "#e67e22",
    "Role.RefPolicy": "#d35400",
    "values": "#16a085",
    "adv": "#9b59b6",
    "update_critic": "#3498db",
    "update_actor": "#2980b9",
    "start_profile": "#7f8c8d",
    "generation_timing/max": "#34495e",
    "generation_timing/min": "#2c3e50",
    "generation_timing/topk_ratio": "#22313f",
    "unknown": "#95a5a6",
}

# Operations that should not be treated as real work for aggregate timing/energy.
EXCLUDED_OPERATIONS = {
    "start_profile",
    "generation_timing/max",
    "generation_timing/min",
    "generation_timing/topk_ratio",
    "step",
}

PHASE_OPERATION_ORDER: Dict[str, Sequence[str]] = {
    "rollout": [
        "start_profile",
        "generate_sequences",
        "generation_timing/max",
        "generation_timing/min",
        "generation_timing/topk_ratio",
        "gen",
        "gen_max",
    ],
    "rl_policy": [
        "reward",
        "old_log_prob",
        "Role.RefPolicy",
        "values",
        "adv",
    ],
    "training": [
        "update_critic",
        "update_actor",
    ],
}


@dataclass(frozen=True)
class ThemeConfig:
    """Shared plotting style configuration."""

    style: str = "seaborn-v0_8-paper"
    context: str = "paper"
    palette: str = "deep"
    figure_dpi: int = 150
    save_dpi: int = 300
    font_size: int = 12
    axes_label_size: int = 13
    axes_title_size: int = 14
    legend_font_size: int = 11
    grid_alpha: float = 0.25
    rc_params: Mapping[str, object] = field(
        default_factory=lambda: {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
        }
    )


def apply_theme(theme: ThemeConfig) -> None:
    """Apply a consistent plotting theme."""

    try:
        plt.style.use(theme.style)
    except OSError:
        plt.style.use("default")

    if sns is not None:
        sns.set_theme(context=theme.context, style="whitegrid", palette=theme.palette)

    plt.rcParams.update(dict(theme.rc_params))


def format_title(run_name: str, title: str) -> Tuple[str, str]:
    """Return (suptitle, title) with experiment name above metric title."""
    return run_name, title


class BasePlotter:
    """Base plotter with shared data loading, theme, and save logic."""

    plot_name: str = "base"
    plot_title: str = "Plot"

    def __init__(
        self,
        run_paths,
        output_dir: Path,
        theme: ThemeConfig | None = None,
    ) -> None:
        self.run_paths = run_paths
        self.output_dir = output_dir
        self.theme = theme or ThemeConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data once per plotter instance.
        self.annotated_df = run_paths.annotated_df
        self.merged_df = run_paths.merged_df
        self.timings_df = run_paths.timings_df

    def create_figure(self):
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        return fig, axes

    def draw(self, fig, axes) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def adjust_layout(self, fig, axes) -> None:
        # Leave room for suptitles to avoid overlaps with subplot titles.
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    def annotate(self, fig, axes) -> None:
        return None

    def finalize(self, fig, axes) -> None:
        return None

    def output_path(self) -> Path:
        filename = f"{self.plot_name}_{self.run_paths.run_name}.png"
        return self.output_dir / filename

    def render(self) -> Path:
        apply_theme(self.theme)
        fig, axes = self.create_figure()
        self.draw(fig, axes)
        self.annotate(fig, axes)
        self.adjust_layout(fig, axes)
        self.finalize(fig, axes)

        out_path = self.output_path()
        fig.savefig(out_path, bbox_inches="tight", dpi=self.theme.save_dpi)
        plt.close(fig)
        return out_path
