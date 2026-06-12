from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    """Return an absolute path inside the project root."""
    return PROJECT_ROOT.joinpath(*map(Path, parts))


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = project_path(file_path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {file_path}")
    return data


def load_settings() -> dict[str, Any]:
    return load_yaml("config/settings.yaml")


def load_strategy_config() -> dict[str, Any]:
    return load_yaml("config/strategy_grid.yaml")


def load_model_config() -> dict[str, Any]:
    return load_yaml("config/model_config.yaml")


def configured_path(config_key: str, filename: str | None = None) -> Path:
    settings = load_settings()
    raw_path = settings["paths"][config_key]
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_path(path)
    if filename is not None:
        path = path / filename
    return path

