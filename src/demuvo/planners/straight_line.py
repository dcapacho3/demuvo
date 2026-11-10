"""The straight-line planner."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from robomath import Transform2D

from ..costmap import Costmap
from ..paths import with_headings
from ..plugins import check_positive


@dataclass(frozen=True)
class StraightLine:
    """The straight segment from the start to the goal. It ignores the costmap.

    Attributes:
        spacing: distance between the points of the path, in metres.
    """

    spacing: float = 0.05

    def __post_init__(self) -> None:
        """Check the spacing."""
        check_positive(self, "spacing")

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap and ignore it.

        This planner shows how a plugin gets its start and goal, and nothing else.
        """

    def plan(self, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """Points `spacing` apart; the last one has the goal's angle."""
        (x0, y0, yaw0), (x1, y1, yaw1) = start.xyt, goal.xyt
        distance = math.hypot(x1 - x0, y1 - y0)
        n = max(2, math.ceil(distance / self.spacing) + 1)
        points = np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])
        path = with_headings(points, yaw1)
        if distance == 0.0:  # no direction to look along, so keep the start's angle
            path[:-1, 2] = yaw0
        return path
