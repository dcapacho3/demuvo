"""What a planner offers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from robomath import Transform2D

from ..costmap import Costmap


class Planner(Protocol):
    """What every planner class offers.

    The navigation calls `configure` once, before the first plan, with the
    robot's global costmap (None if the world turns the costmap off). The costmap
    object stays the same afterwards, so a planner keeps it.
    """

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap the planner searches."""

    def plan(self, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """The path from start to goal, shape (N, 3); shape (0, 3) if there is none."""
