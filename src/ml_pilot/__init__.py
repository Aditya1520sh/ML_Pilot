"""MLPilot — Data In, Insights Out.

End-to-end Machine Learning for tabular datasets: load, explore, clean, engineer features,
compare models, tune, select, evaluate, explain, and export.
"""

from ml_pilot.core.context import PipelineContext
from ml_pilot.core.pipeline import PipelineRunner
from ml_pilot.core.registry import StageRegistry
from ml_pilot.core.stage import BaseStage

__version__ = "0.1.0"
__all__ = [
    "PipelineContext",
    "PipelineRunner",
    "StageRegistry",
    "BaseStage",
    "__version__",
]
