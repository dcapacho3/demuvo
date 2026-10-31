"""What a planner offers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from robomath import Transform2D

from ..world import World


class Planner(Protocol):
    """What every planner class offers."""

    def plan(self, world: World, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """The path from start to goal, shape (N, 3)."""
