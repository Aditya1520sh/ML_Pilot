"""Dynamic registration and ordering of pipeline stages.

Purpose:
    Allow stages to be registered by name and retrieved in execution order
    without hard-coding imports inside the runner.

Interactions:
    - Populated by ``core.defaults.build_default_registry``.
    - Queried by ``PipelineRunner`` to obtain the ordered stage list.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ml_pilot.core.stage import BaseStage


class StageRegistry:
    """Ordered registry of ``BaseStage`` instances."""

    def __init__(self) -> None:
        self._stages: dict[str, BaseStage] = {}
        self._order: list[str] = []

    def register(self, stage: BaseStage, *, position: int | None = None) -> None:
        """Register a stage, optionally inserting it at ``position``.

        Args:
            stage: Stage instance to store.
            position: Optional index in the execution order. Appends when None.

        Raises:
            ValueError: If a stage with the same name already exists.
        """
        key = stage.name
        if key in self._stages:
            raise ValueError(f"Stage already registered: {key}")
        self._stages[key] = stage
        if position is None:
            self._order.append(key)
        else:
            self._order.insert(position, key)

    def unregister(self, name: str) -> None:
        """Remove a stage by name if present."""
        self._stages.pop(name, None)
        self._order = [item for item in self._order if item != name]

    def get(self, name: str) -> BaseStage:
        """Return a registered stage.

        Raises:
            KeyError: If the name is unknown.
        """
        if name not in self._stages:
            raise KeyError(f"Unknown stage: {name}")
        return self._stages[name]

    def ordered(self, only: Sequence[str] | None = None) -> list[BaseStage]:
        """Return stages in registration order, optionally filtered.

        Args:
            only: If provided, keep only these names (preserving registry order).

        Returns:
            Ordered list of stage instances.
        """
        names = self._order if only is None else [n for n in self._order if n in only]
        return [self._stages[name] for name in names]

    def names(self) -> list[str]:
        """Return registered stage names in order."""
        return list(self._order)

    def extend(self, stages: Iterable[BaseStage]) -> None:
        """Register many stages in the given sequence order."""
        for stage in stages:
            self.register(stage)

    def __contains__(self, name: str) -> bool:
        return name in self._stages

    def __len__(self) -> int:
        return len(self._stages)
