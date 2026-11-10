"""The line-of-sight smoother - skips the points of a path that a straight line can."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..costmap import INSCRIBED, Costmap
from ..paths import with_headings
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
        rows, cols = self._costmap.cells(points)
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

    def _clear(self, a: np.ndarray, b: np.ndarray, limit: int) -> bool:
        """True if the line from a to b crosses only free cells of cost up to limit."""
        rows, cols = self._costmap.line_cells(a, b)
        first_row, first_col = self._costmap.cells(a[None])
        seen = self._costmap.data[rows, cols].astype(int)
        seen = seen[(rows != first_row[0]) | (cols != first_col[0])]
        return seen.size == 0 or (seen.max() < INSCRIBED and seen.max() <= limit)

    def _resample(self, corners: np.ndarray, goal_yaw: float) -> np.ndarray:
        """Poses spaced `spacing` apart along the corners; the last has goal_yaw."""
        step = np.diff(corners, axis=0)
        counts = np.maximum(1, np.ceil(np.hypot(*step.T) / self.spacing).astype(int))
        pieces = [
            corners[k]
            + np.linspace(0.0, 1.0, counts[k], endpoint=False)[:, None] * step[k]
            for k in range(len(step))
        ]
        return with_headings(np.vstack([*pieces, corners[-1:]]), goal_yaw)
