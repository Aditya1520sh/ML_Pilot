"""Abstract base class for pipeline stages.

Purpose:
    Define the contract every MLPilot stage must implement.

Interactions:
    - Subclassed by modules under ``ml_pilot.stages``.
    - Registered in ``StageRegistry`` and invoked by ``PipelineRunner``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml_pilot.core.context import PipelineContext


class BaseStage(ABC):
    """Single unit of work in an MLPilot pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable stage identifier used in logs and timings."""

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the stage, mutate ``context``, and return it.

        Args:
            context: Shared pipeline state.

        Returns:
            The same context instance after updates.
        """

    def should_skip(self, context: PipelineContext) -> bool:
        """Return True when this stage should be omitted for the current run.

        Args:
            context: Shared pipeline state.

        Returns:
            Whether the runner should skip ``run``.
        """
        return False
