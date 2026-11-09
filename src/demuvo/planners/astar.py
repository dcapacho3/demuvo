"""The A* planner - a search over the cells of the global costmap."""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from robomath import Transform2D

from ..costmap import INSCRIBED, MAX_NON_OBSTACLE, Costmap

LOGGER = logging.getLogger(__name__)

# The 4 and 8 neighbours of a cell as (row step, column step).
_STEPS_4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
_STEPS_8 = _STEPS_4 + ((1, 1), (1, -1), (-1, -1), (-1, 1))


@dataclass
class AStar:
    """The cheapest path over the cells of the costmap from start to goal.

    A cell of cost `INSCRIBED` or more is blocked. A step costs its length times
    `1 + cost_weight * cost / MAX_NON_OBSTACLE`, with the cost of the cell it
    enters, so the path keeps off walls. The path is the centres of the cells
    it crosses, with the exact start and goal at its ends, and it is empty when
    there is no path. A diagonal step needs both cells beside it free, so the
    path never cuts a corner.

    The start cell is always allowed, since a robot against a wall can sit in a
    blocked cell. A goal that is blocked or off the map gives no path.

    Attributes:
        connectivity: 8 moves to the four sides and the four diagonals; 4 only
            to the four sides, which gives paths of 90-degree turns.
        cost_weight: how much a cell's cost raises the price of a step, 0 or more.
            0 gives the shortest path whatever the costs.
    """

    connectivity: int = 8
    cost_weight: float = 2.0
    _costmap: Costmap | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check the settings."""
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity!r}")
        weight = self.cost_weight
        if (
            not isinstance(weight, int | float)
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(f"cost_weight must be 0 or more, got {weight!r}")

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap to search.

        Raises:
            ValueError: if there is none, since this planner cannot work without.
        """
        if costmap is None:
            raise ValueError("AStar needs a costmap, but the world turns it off")
        self._costmap = costmap

    def plan(self, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """The path from start to goal, shape (N, 3); shape (0, 3) if there is none."""
        if self._costmap is None:
            raise ValueError("AStar is not configured; call configure first")
        costmap = self._costmap
        first = costmap.world_to_cell(*start.xyt[:2])
        last = costmap.world_to_cell(*goal.xyt[:2])
        if not _on_map(costmap, first):
            return _no_path("the start (%.2f, %.2f) is off the map", *start.xyt[:2])
        if not _on_map(costmap, last):
            return _no_path("the goal (%.2f, %.2f) is off the map", *goal.xyt[:2])
        blocked = costmap.data >= INSCRIBED
        if blocked[last]:
            return _no_path(
                "the goal cell %s is blocked (cost %d), so the robot's centre "
                "cannot be there",
                last,
                costmap.data[last],
            )
        price = 1.0 + self.cost_weight * costmap.data / MAX_NON_OBSTACLE
        cells = self._search(blocked, price, first, last)
        if cells is None:
            return _no_path(
                "no way from the start %s to the goal %s; one of them is "
                "walled in by blocked cells",
                first,
                last,
            )
        middle = [costmap.cell_to_world(r, c) for r, c in cells[1:-1]]
        points = np.array([start.xyt[:2], *middle, goal.xyt[:2]])
        return _with_headings(points, goal.xyt[2])

    def _heuristic(self, cell, last):
        """The distance from a cell to the goal cell if nothing were in the way."""
        dr, dc = abs(cell[0] - last[0]), abs(cell[1] - last[1])
        if self.connectivity == 4:
            return self._costmap.resolution * (dr + dc)
        return self._costmap.resolution * (dr + dc + (math.sqrt(2) - 2) * min(dr, dc))

    def _neighbours(self, cell, blocked):
        """The free cells a step from `cell` reaches, with the length of the step."""
        rows, cols = blocked.shape
        steps = _STEPS_8 if self.connectivity == 8 else _STEPS_4
        for dr, dc in steps:
            r, c = cell[0] + dr, cell[1] + dc
            if not (0 <= r < rows and 0 <= c < cols) or blocked[r, c]:
                continue
            if dr and dc and (blocked[cell[0] + dr, cell[1]] or blocked[cell[0], c]):
                continue  # a diagonal step may not cut a corner
            yield (r, c), self._costmap.resolution * math.hypot(dr, dc)

    def _search(self, blocked, price, first, last):
        """The cells from first to last as a list, or None if last cannot be reached."""
        best = {first: 0.0}
        came_from = {first: None}
        # (estimate, distance still to go, order found, cell): a tie goes to the
        # cell nearer the goal, then to the one found first, so a run repeats.
        start_estimate = self._heuristic(first, last)
        queue = [(start_estimate, start_estimate, 0, first)]
        found = 1
        while queue:
            estimate, remaining, _, cell = heapq.heappop(queue)
            if estimate - remaining > best[cell] + 1e-9:
                continue  # a cheaper way to this cell was found after this entry
            if cell == last:
                return _trace(came_from, cell)
            for neighbour, length in self._neighbours(cell, blocked):
                cost = best[cell] + length * price[neighbour]
                if cost < best.get(neighbour, math.inf):
                    best[neighbour] = cost
                    came_from[neighbour] = cell
                    h = self._heuristic(neighbour, last)
                    heapq.heappush(queue, (cost + h, h, found, neighbour))
                    found += 1
        return None


def _on_map(costmap: Costmap, cell) -> bool:
    """True if the (row, column) lies on the costmap."""
    return 0 <= cell[0] < costmap.data.shape[0] and 0 <= cell[1] < costmap.data.shape[1]


def _no_path(reason: str, *values) -> np.ndarray:
    """Say why there is no path, and give the empty path."""
    LOGGER.warning("AStar: " + reason, *values)
    return np.empty((0, 3))


def _trace(came_from, cell):
    """The cells from the first to `cell`, by following where each came from."""
    path = []
    while cell is not None:
        path.append(cell)
        cell = came_from[cell]
    return path[::-1]


def _with_headings(points: np.ndarray, goal_yaw: float) -> np.ndarray:
    """Poses (N, 3) from points (N, 2).

    Each pose looks along its step, and the last has the goal's angle.
    """
    path = np.empty((len(points), 3))
    path[:, :2] = points
    step = np.diff(points, axis=0)
    path[:-1, 2] = np.arctan2(step[:, 1], step[:, 0])
    path[-1, 2] = goal_yaw
    return path
