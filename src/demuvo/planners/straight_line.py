"""The straight-line planner."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from robomath import Transform2D

from ..plugins import check_positive
from ..world import World


@dataclass(frozen=True)
class StraightLine:
    """The straight segment from the start to the goal. It ignores obstacles.

    Attributes:
        spacing: distance between the points of the path, in metres.
    """

    spacing: float = 0.05

    def __post_init__(self) -> None:
        """Check the spacing."""
        check_positive(self, "spacing")

    def plan(self, world: World, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """Points `spacing` apart; the last one has the goal's angle."""
        del world
        (x0, y0, yaw0), (x1, y1, yaw1) = start.xyt, goal.xyt
        distance = math.hypot(x1 - x0, y1 - y0)
        heading = math.atan2(y1 - y0, x1 - x0) if distance > 0.0 else yaw0
        n = max(2, math.ceil(distance / self.spacing) + 1)
        path = np.empty((n, 3))
        path[:, 0] = np.linspace(x0, x1, n)
        path[:, 1] = np.linspace(y0, y1, n)
        path[:, 2] = heading
        path[-1, 2] = yaw1
        return path
