"""Default stage wiring for a standard MLPilot run.

Purpose:
    Construct a ``StageRegistry`` pre-loaded with the canonical end-to-end
    stage sequence.

Interactions:
    - Imports concrete stages from ``ml_pilot.stages``.
    - Used by ``PipelineRunner`` when no custom registry is supplied.
"""

from __future__ import annotations

from ml_pilot.core.registry import StageRegistry
from ml_pilot.stages.clean import CleanStage
from ml_pilot.stages.compare import CompareStage
from ml_pilot.stages.deploy import DeployStage
from ml_pilot.stages.eda import EdaStage
from ml_pilot.stages.evaluate import EvaluateStage
from ml_pilot.stages.explain import ExplainStage
from ml_pilot.stages.feature_eng import FeatureEngStage
from ml_pilot.stages.load import LoadStage
from ml_pilot.stages.preprocess import PreprocessStage
from ml_pilot.stages.save import SaveStage
from ml_pilot.stages.select import SelectStage
from ml_pilot.stages.split import SplitStage
from ml_pilot.stages.tune import TuneStage


def build_default_registry() -> StageRegistry:
    """Create the default ordered stage registry.

    Returns:
        Registry containing Load → … → Deploy.
    """
    registry = StageRegistry()
    registry.extend(
        [
            LoadStage(),
            EdaStage(),
            CleanStage(),
            FeatureEngStage(),
            SplitStage(),
            PreprocessStage(),
            CompareStage(),
            TuneStage(),
            SelectStage(),
            EvaluateStage(),
            ExplainStage(),
            SaveStage(),
            DeployStage(),
        ]
    )
    return registry


# Canonical stage order used when ``compare_only`` truncates the pipeline.
COMPARE_ONLY_STAGES: tuple[str, ...] = (
    "load",
    "eda",
    "clean",
    "feature_eng",
    "split",
    "preprocess",
    "compare",
    "save",
)

FULL_RUN_STAGES: tuple[str, ...] = (
    "load",
    "eda",
    "clean",
    "feature_eng",
    "split",
    "preprocess",
    "compare",
    "tune",
    "select",
    "evaluate",
    "explain",
    "save",
    "deploy",
)
