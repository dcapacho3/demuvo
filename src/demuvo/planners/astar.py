"""The A* planner - a search over the cells of the global costmap."""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from robomath import Transform2D

from ..costmap import INSCRIBED, MAX_NON_OBSTACLE, Costmap
from ..plugins import check_not_negative

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
    blocked cell. A goal off the map gives no path. A goal that is blocked gives
    no path either, unless `tolerance` allows the path to end at the nearest free
    cell within that distance of the goal; the run says so when it does. A goal that
    is free but walled in gives no path.

    Attributes:
        connectivity: 8 moves to the four sides and the four diagonals; 4 only
            to the four sides, which gives paths of 90-degree turns.
        cost_weight: how much a cell's cost raises the price of a step, 0 or more.
            0 gives the shortest path whatever the costs.
        tolerance: how far from a blocked goal, in metres, the path may end; 0 or
            more. The default 0 gives no path to a blocked goal, so a result never
            hides that the goal was out of reach.
    """

    connectivity: int = 8
    cost_weight: float = 2.0
    tolerance: float = 0.0
    _costmap: Costmap | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check the settings."""
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity!r}")
        check_not_negative(self, "cost_weight", "tolerance")

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
        end = goal.xyt[:2]
        if blocked[last]:
            wanted, near = last, self._nearest_free(blocked, goal.xyt[:2])
            if near is None:
                return _no_path(*self._blocked_goal(wanted))
            last = near
            end = costmap.cell_to_world(*last)
            LOGGER.warning(
                "AStar: the goal cell %s is blocked (cost %d), so the path ends at "
                "the nearest free cell %s, %.2f m from the goal",
                wanted,
                costmap.data[wanted],
                last,
                math.dist(end, goal.xyt[:2]),
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
        points = np.array([start.xyt[:2], *middle, end])
        return _with_headings(points, goal.xyt[2])

    def _blocked_goal(self, cell):
        """The message and values that say why a blocked goal has no path."""
        cost = self._costmap.data[cell]
        if self.tolerance == 0:
            return (
                "the goal cell %s is blocked (cost %d), so the robot's centre "
                "cannot be there",
                cell,
                cost,
            )
        return (
            "the goal cell %s is blocked (cost %d) and no free cell lies within "
            "%.2f m of it",
            cell,
            cost,
            self.tolerance,
        )

    def _nearest_free(self, blocked, goal_xy):
        """The free cell nearest the goal within `tolerance` of it, or None."""
        costmap = self._costmap
        row, col = costmap.world_to_cell(*goal_xy)
        reach = math.ceil(round(self.tolerance / costmap.resolution, 9))
        rows = np.arange(max(0, row - reach), min(blocked.shape[0], row + reach + 1))
        cols = np.arange(max(0, col - reach), min(blocked.shape[1], col + reach + 1))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        x = costmap.origin[0] + (cc + 0.5) * costmap.resolution
        y = costmap.origin[1] + (rr + 0.5) * costmap.resolution
        distance = np.hypot(x - goal_xy[0], y - goal_xy[1])
        distance[blocked[rr, cc] | (distance > self.tolerance)] = np.inf
        nearest = int(distance.argmin())  # the first of equals in row order
        if not np.isfinite(distance.flat[nearest]):
            return None
        return int(rr.flat[nearest]), int(cc.flat[nearest])

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
