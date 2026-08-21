"""Common typing aliases used across MLPilot.

Purpose:
    Keep public signatures readable without importing heavyweight types
    everywhere.

Interactions:
    - Referenced by stage modules and utils for annotations.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pandas as pd

FrameLike: TypeAlias = pd.DataFrame
ArrayLike: TypeAlias = np.ndarray | pd.Series | list[Any]
