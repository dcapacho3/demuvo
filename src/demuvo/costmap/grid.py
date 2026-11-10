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

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The (rows, columns) of the cells that hold the points (m, 2).

        A point may lie outside the grid. The rounding keeps a point on a cell edge
        from landing in the wrong cell through float noise.
        """
        scaled = (np.asarray(xy, dtype=float) - self.origin) / self.resolution
        cells = np.floor(np.round(scaled, 9)).astype(int)
        return cells[:, 1], cells[:, 0]

    def line_cells(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The (rows, columns) of every cell the line from a to b passes through.

        The cells come in order from a to b, and they may lie outside the grid. A
        line that goes exactly through a corner also gives the two cells that touch
        it only there, so it cannot squeeze between two blocked cells.
        """
        start = (a - self.origin) / self.resolution
        move = (b - self.origin) / self.resolution - start
        crossings = []
        for axis in (0, 1):  # where the line crosses each vertical, then horizontal
            if move[axis] == 0:
                crossings.append(np.empty(0))
                continue
            low, high = sorted((start[axis], start[axis] + move[axis]))
            edges = np.arange(math.floor(low) + 1, math.ceil(high))
            crossings.append(np.round((edges - start[axis]) / move[axis], 9))
        times = np.unique(np.concatenate([[0.0, 1.0], *crossings]))
        inside = start + ((times[:-1] + times[1:]) / 2)[:, None] * move
        rows, cols = [np.floor(inside[:, 1])], [np.floor(inside[:, 0])]
        for time in np.intersect1d(
            crossings[0], crossings[1]
        ):  # exactly through a corner
            before = np.floor(start + (time - 1e-6) * move)
            after = np.floor(start + (time + 1e-6) * move)
            rows += [np.array([before[1], after[1]])]
            cols += [np.array([after[0], before[0]])]
        return np.concatenate(rows).astype(int), np.concatenate(cols).astype(int)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """The (row, column) of the cell that holds the point; it may be outside."""
        rows, cols = self.cells(np.array([[x, y]]))
        return int(rows[0]), int(cols[0])

    def centres(self, rows, cols) -> tuple[np.ndarray, np.ndarray]:
        """The world positions (x, y) of the centres of the cells, as arrays."""
        return (
            self.origin[0] + (np.asarray(cols) + 0.5) * self.resolution,
            self.origin[1] + (np.asarray(rows) + 0.5) * self.resolution,
        )

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """The world position of the centre of a cell."""
        x, y = self.centres(row, col)
        return float(x), float(y)

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
