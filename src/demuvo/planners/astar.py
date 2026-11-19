"""The A* planner - a search over the cells of the global costmap."""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from robomath import Transform2D

from ..costmap import INSCRIBED, MAX_NON_OBSTACLE, Costmap
from ..paths import with_headings
from ..plugins import check_not_negative
from .base import no_path

LOGGER = logging.getLogger(__name__)

# The 4 and 8 neighbours of a cell as (row step, column step).
_STEPS_4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
_STEPS_8 = _STEPS_4 + ((1, 1), (1, -1), (-1, -1), (-1, 1))
# What a diagonal step saves over two straight ones, per cell, in the estimate.
_DIAGONAL_GAP = math.sqrt(2) - 2


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
        if not costmap.in_bounds(*start.xyt[:2]):
            return no_path(
                "AStar", "the start (%.2f, %.2f) is off the map", *start.xyt[:2]
            )
        if not costmap.in_bounds(*goal.xyt[:2]):
            return no_path(
                "AStar", "the goal (%.2f, %.2f) is off the map", *goal.xyt[:2]
            )
        blocked = costmap.data >= INSCRIBED
        end = goal.xyt[:2]
        if blocked[last]:
            wanted, near = last, self._nearest_free(blocked, goal.xyt[:2])
            if near is None:
                return no_path("AStar", *self._blocked_goal(wanted))
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
            return no_path(
                "AStar",
                "no way from the start %s to the goal %s; one of them is "
                "walled in by blocked cells",
                first,
                last,
            )
        middle = [costmap.cell_to_world(r, c) for r, c in cells[1:-1]]
        points = np.array([start.xyt[:2], *middle, end])
        return with_headings(points, goal.xyt[2])

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
        x, y = costmap.centres(rr, cc)
        distance = np.hypot(x - goal_xy[0], y - goal_xy[1])
        distance[blocked[rr, cc] | (distance > self.tolerance)] = np.inf
        nearest = int(distance.argmin())  # the first of equals in row order
        if not np.isfinite(distance.flat[nearest]):
            return None
        return int(rr.flat[nearest]), int(cc.flat[nearest])

    def _search(self, blocked, price, first, last):  # pylint: disable=too-many-locals
        """The cells from first to last as a list, or None if last cannot be reached.

        The cells are numbered along the rows, so a neighbour is a number added to
        the cell's own, and the grids are plain lists: indexing a numpy array one
        cell at a time is several times slower, and this loop runs once for every
        cell it reaches. That is why its values are all locals.
        """
        rows, cols = blocked.shape
        resolution = self._costmap.resolution
        shut, cost_of = blocked.ravel().tolist(), price.ravel().tolist()
        steps = _STEPS_8 if self.connectivity == 8 else _STEPS_4
        moves = [
            (dr * cols + dc, dr, dc, resolution * math.hypot(dr, dc), bool(dr and dc))
            for dr, dc in steps
        ]
        gap = _DIAGONAL_GAP if self.connectivity == 8 else 0.0
        goal_row, goal_col = last
        goal = goal_row * cols + goal_col
        start = first[0] * cols + first[1]
        best = [math.inf] * (rows * cols)
        came_from = [-1] * (rows * cols)
        best[start] = 0.0
        # (estimate, distance still to go, order found, cell, row, column): a tie goes
        # to the cell nearer the goal, then to the one found first, so a run repeats.
        far = abs(first[0] - goal_row), abs(first[1] - goal_col)
        h0 = resolution * (sum(far) + gap * min(far))
        queue = [(h0, h0, 0, start, first[0], first[1])]
        found = 1
        while queue:
            estimate, remaining, _, cell, row, col = heapq.heappop(queue)
            here = best[cell]
            if estimate - remaining > here + 1e-9:
                continue  # a cheaper way to this cell was found after this entry
            if cell == goal:
                return _trace(came_from, cell, cols)
            for offset, dr, dc, length, corner in moves:
                r, c = row + dr, col + dc
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                ahead = cell + offset
                if shut[ahead]:
                    continue
                if corner and (shut[cell + dr * cols] or shut[cell + dc]):
                    continue  # a diagonal step may not cut a corner
                cost = here + length * cost_of[ahead]
                if cost < best[ahead]:
                    best[ahead] = cost
                    came_from[ahead] = cell
                    far_r, far_c = abs(r - goal_row), abs(c - goal_col)
                    h = resolution * (far_r + far_c + gap * min(far_r, far_c))
                    heapq.heappush(queue, (cost + h, h, found, ahead, r, c))
                    found += 1
        return None


def _trace(came_from, cell, cols):
    """The cells (row, column) from the first to `cell`, by where each came from."""
    path = []
    while cell != -1:
        path.append(divmod(cell, cols))
        cell = came_from[cell]
    return path[::-1]
