"""Configuration loading and validation for MLPilot."""

from ml_pilot.config.defaults import DEFAULT_CONFIG_DICT
from ml_pilot.config.resolver import load_config, merge_overrides
from ml_pilot.config.schema import MLPilotConfig, TaskType

__all__ = [
    "DEFAULT_CONFIG_DICT",
    "MLPilotConfig",
    "TaskType",
    "load_config",
    "merge_overrides",
]
