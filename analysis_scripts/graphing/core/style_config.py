#!/usr/bin/env python3
"""Run style configuration for quad-view plots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class RunStyleRule:
    name: Optional[str] = None
    run_match: Optional[str] = None
    label_match: Optional[str] = None
    plot_match: Optional[str] = None
    color: Optional[str] = None
    hatch: Optional[str] = None
    linestyle: Optional[str] = None

    def matches(self, run_name: str, label: str, plot_name: Optional[str] = None) -> bool:
        if not self.run_match and not self.label_match and not self.plot_match:
            return False
        if self.run_match and re.search(self.run_match, run_name) is None:
            return False
        if self.label_match and re.search(self.label_match, label) is None:
            return False
        if self.plot_match and plot_name:
            if re.search(self.plot_match, plot_name) is None:
                return False
        elif self.plot_match and plot_name is None:
            return False
        return True


@dataclass(frozen=True)
class RunStyleConfig:
    rules: List[RunStyleRule]
    defaults: RunStyleRule

    def match(self, run_name: str, label: str, plot_name: Optional[str] = None) -> Optional[RunStyleRule]:
        for rule in self.rules:
            if rule.matches(run_name, label, plot_name):
                return rule
        return None


def _rule_from_dict(data: dict) -> RunStyleRule:
    return RunStyleRule(
        name=data.get("name"),
        run_match=data.get("run_match") or data.get("match"),
        label_match=data.get("label_match"),
        plot_match=data.get("plot_match"),
        color=data.get("color"),
        hatch=data.get("hatch"),
        linestyle=data.get("linestyle"),
    )


def load_style_config(path: Optional[Path]) -> Optional[RunStyleConfig]:
    if path is None:
        return None
    raw = Path(path).expanduser()
    if not raw.exists():
        raise FileNotFoundError(f"Style config not found: {raw}")
    data = json.loads(raw.read_text())
    rules = [_rule_from_dict(item) for item in data.get("runs", []) if isinstance(item, dict)]
    defaults = _rule_from_dict(data.get("defaults", {}))
    return RunStyleConfig(rules=rules, defaults=defaults)
