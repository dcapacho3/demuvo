"""The line-of-sight smoother - skips the points of a path that a straight line can."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..costmap import INSCRIBED, Costmap
from ..plugins import check_positive


@dataclass
class LineOfSight:
    """Replaces stretches of a path by straight lines wherever the way is clear.

    From each point it takes the farthest point that a straight line reaches, keeps
    it, and goes on from there. A line is clear when every cell it
    crosses is free (cost under `INSCRIBED`) and none costs more than the dearest
    cell of the stretch it replaces, so the smoother never moves a path onto
    ground the planner would have avoided. The start cell may be blocked, as the
    planner allows. The kept points are joined and spaced `spacing` apart again,
    so a controller that follows path points still works.

    Attributes:
        spacing: distance between the points of the result, in metres.
    """

    spacing: float = 0.05
    _costmap: Costmap | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check the spacing."""
        check_positive(self, "spacing")

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap to check lines against.

        Raises:
            ValueError: if there is none, since this smoother cannot work without.
        """
        if costmap is None:
            raise ValueError("LineOfSight needs a costmap, but the world turns it off")
        self._costmap = costmap

    def smooth(self, path: np.ndarray) -> np.ndarray:
        """The path with its clear stretches made straight, shape (M, 3).

        The first point stays where it is, and the last keeps the goal's angle.

        Raises:
            ValueError: if a point of the path lies off the costmap.
        """
        if self._costmap is None:
            raise ValueError("LineOfSight is not configured; call configure first")
        points = path[:, :2]
        rows, cols = self._cells(points)
        height, width = self._costmap.data.shape
        off = (rows < 0) | (rows >= height) | (cols < 0) | (cols >= width)
        if off.any():
            raise ValueError(f"the path leaves the costmap at point {off.argmax()}")
        costs = self._costmap.data[rows, cols].astype(int)
        kept = [0]
        i = 0
        while i < len(path) - 1:
            worst = np.maximum.accumulate(
                costs[i:]
            )  # dearest cell from i to each point
            for j in range(len(path) - 1, i + 1, -1):  # the farthest clear line first
                if self._clear(points[i], points[j], worst[j - i]):
                    break
            else:
                j = i + 1
            kept.append(j)
            i = j
        return self._resample(points[kept], path[-1, 2])

    def _cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The (row, column) of the cell that holds each point (m, 2)."""
        costmap = self._costmap
        scaled = (xy - costmap.origin) / costmap.resolution
        cells = np.floor(np.round(scaled, 9)).astype(int)
        return cells[:, 1], cells[:, 0]

    def _line_cells(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The (rows, columns) of every cell the line from a to b passes through.

        A line that goes exactly through a corner also gives the two cells that
        touch it only there, so it cannot squeeze between two blocked cells.
        """
        costmap = self._costmap
        start = (a - costmap.origin) / costmap.resolution
        move = (b - costmap.origin) / costmap.resolution - start
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

    def _clear(self, a: np.ndarray, b: np.ndarray, limit: int) -> bool:
        """True if the line from a to b crosses only free cells of cost up to limit."""
        rows, cols = self._line_cells(a, b)
        first_row, first_col = self._cells(a[None])
        seen = self._costmap.data[rows, cols].astype(int)
        seen = seen[(rows != first_row[0]) | (cols != first_col[0])]
        return seen.size == 0 or (seen.max() < INSCRIBED and seen.max() <= limit)

    def _resample(self, corners: np.ndarray, goal_yaw: float) -> np.ndarray:
        """Poses spaced `spacing` apart along the corners; the last has goal_yaw."""
        step = np.diff(corners, axis=0)
        lengths = np.hypot(step[:, 0], step[:, 1])
        counts = np.maximum(1, np.ceil(lengths / self.spacing).astype(int))
        pieces = [
            corners[k]
            + np.linspace(0.0, 1.0, counts[k], endpoint=False)[:, None] * step[k]
            for k in range(len(step))
        ]
        headings = np.concatenate(
            [
                np.full(counts[k], np.arctan2(step[k, 1], step[k, 0]))
                for k in range(len(step))
            ]
        )
        path = np.empty((len(headings) + 1, 3))
        path[:-1, :2] = np.concatenate(pieces)
        path[:-1, 2] = headings
        path[-1] = (*corners[-1], goal_yaw)
        return path
