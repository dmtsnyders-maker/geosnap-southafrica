from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def project_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Create and return standard project paths."""
    paths = {"cache": Path(cfg.get("cache_dir", "data/cache")), "raw": Path(cfg.get("raw_dir", "data/raw")),
             "processed": Path(cfg.get("processed_dir", "data/processed")), "outputs": Path(cfg.get("outputs_dir", "outputs"))}
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    return paths
