"""Configuration resolution: defaults, YAML, and CLI overrides.

Purpose:
    Merge layered config sources into a validated ``MLPilotConfig`` instance.

Interactions:
    - Reads ``defaults.DEFAULT_CONFIG_DICT`` and optional YAML via ``utils.io``.
    - Called by the CLI and ``PipelineRunner`` before stages execute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ml_pilot.config.defaults import DEFAULT_CONFIG_DICT
from ml_pilot.config.schema import MLPilotConfig


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``.

    Args:
        base: Starting dictionary.
        overlay: Values that take precedence.

    Returns:
        Merged dictionary without mutating inputs.
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def merge_overrides(
    config: MLPilotConfig,
    overrides: dict[str, Any] | None = None,
) -> MLPilotConfig:
    """Apply a shallow/deep override dict onto an existing config.

    Args:
        config: Existing validated configuration.
        overrides: Nested dict of fields to replace.

    Returns:
        New ``MLPilotConfig`` with overrides applied.
    """
    if not overrides:
        return config
    merged = _deep_merge(config.model_dump(mode="python"), overrides)
    return MLPilotConfig.model_validate(merged)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> MLPilotConfig:
    """Load configuration from defaults, optional YAML, then CLI overrides.

    Args:
        config_path: Path to a YAML file. Ignored when ``None``.
        overrides: Nested overrides (typically from CLI flags).

    Returns:
        Validated ``MLPilotConfig``.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If YAML content is not a mapping.
    """
    payload: dict[str, Any] = dict(DEFAULT_CONFIG_DICT)

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping, got {type(loaded).__name__}")
        payload = _deep_merge(payload, loaded)

    if overrides:
        payload = _deep_merge(payload, overrides)

    return MLPilotConfig.model_validate(payload)
