"""The costmap - a grid of costs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FREE = 0
MAX_NON_OBSTACLE = 252
INSCRIBED = 253
LETHAL = 254
NO_INFORMATION = 255


@dataclass(frozen=True, eq=False)
class Costmap:
    """A grid of costs over part of the world.

    A cell of cost `LETHAL` holds an obstacle; one of cost `INSCRIBED` or more is
    one where the robot's centre cannot be; lower costs fall off with distance
    from an obstacle, and `FREE` means nothing is near.

    Attributes:
        data: costs, shape (rows, columns), type uint8. Row 0 is the lowest y,
            column 0 the lowest x.
        resolution: side of a cell in metres.
        origin: world position (x, y) of the lower-left corner of cell (0, 0).
    """

    data: np.ndarray
    resolution: float
    origin: tuple[float, float] = (0.0, 0.0)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """The (row, column) of the cell that holds the point; it may be outside."""
        col = math.floor(round((x - self.origin[0]) / self.resolution, 9))
        row = math.floor(round((y - self.origin[1]) / self.resolution, 9))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """The world position of the centre of a cell."""
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    def in_bounds(self, x: float, y: float) -> bool:
        """True if the point lies inside the grid."""
        row, col = self.world_to_cell(x, y)
        return 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]

    def cost_at(self, x: float, y: float) -> int:
        """The cost of the cell that holds the point; `NO_INFORMATION` outside."""
        if not self.in_bounds(x, y):
            return NO_INFORMATION
        row, col = self.world_to_cell(x, y)
        return int(self.data[row, col])
