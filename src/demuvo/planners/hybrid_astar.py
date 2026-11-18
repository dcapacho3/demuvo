"""The Hybrid A* planner - a search over poses for a car, round obstacles.

A car cannot follow the cell-to-cell path of `AStar`, and `Dubins` and `ReedsShepp`
ignore the costmap, so a car among obstacles needs both: a path that goes round
what the costmap shows and that no bend is tighter than the car can steer.

The search runs over (x, y, heading). From a pose it tries arcs of the turning
radius, left, straight and right, forward and, with `allow_reversing`, backward.
Two poses count as the same when they fall in the same 0.1 m cell with the same
heading bin, and the cheapest way to each is kept. The distance to the goal round
the obstacles guides the search, and now and then it tries to end the path at once
with a Dubins path (a Reeds-Shepp path with `allow_reversing`) to the goal pose,
which, if the costmap shows it clear, gives the exact goal pose and heading.

A pose whose cell has cost `INSCRIBED` or more is blocked, as for `AStar`. An arc
costs its length times `1 + cost_weight * cost / MAX_NON_OBSTACLE`, with the mean
cost along it, so the path keeps off walls.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from robomath import Transform2D, wrap_angle

from ..costmap import INSCRIBED, MAX_NON_OBSTACLE, Costmap
from ..paths import arc_poses
from ..plugins import check_not_negative, check_positive
from .base import no_path
from .dubins import Dubins
from .reeds_shepp import ReedsShepp

# The side of a cell of the search, in metres, and how many heading bins it has.
CELL = 0.1
BINS = 72
# The length of one arc, in metres; it must be more than the cell's diagonal, so a
# step always reaches another cell.
STEP = 0.25
# A pose this near the goal, in turning radii, tries the closing path at each step;
# further off, at every this many steps.
NEAR = 4.0
EVERY = 10
# The extra cost in metres of changing direction, and of changing the way it steers.
CUSP = 2.0
TURN = 0.2
# The 8 neighbours of a cell as (row step, column step, length in cells).
_AROUND = tuple(
    (dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
)


@dataclass
class HybridAStar:
    """The cheapest path of arcs a car can drive, over the costmap, to a goal pose.

    The path ends on the goal pose, heading included, and is empty when there is
    none: the start or goal is off the map, the goal is blocked, or the search ran
    out of poses to open without a way. The start may sit in a blocked cell, as for
    `AStar`, though no arc may then leave it.

    Attributes:
        turning_radius: the tightest bend of the path, in metres. Set it no lower
            than the robot's least turning radius, or the robot cannot track it.
        allow_reversing: lets the path back up, as a Reeds-Shepp path does; off,
            it drives forward only, as a Dubins path does.
        cost_weight: how much a cell's cost raises the price of a step, 0 or more.
        reverse_penalty: how many times dearer a metre backward is than one
            forward, 1 or more.
        spacing: distance between the points of the path, in metres.
        max_expansions: how many poses the search may open before it gives up.
    """

    turning_radius: float = 0.6
    allow_reversing: bool = False
    cost_weight: float = 2.0
    reverse_penalty: float = 2.0
    spacing: float = 0.05
    max_expansions: int = 20000
    _costmap: Costmap | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _order: itertools.count = field(
        default_factory=itertools.count, init=False, repr=False, compare=False
    )

    curvature_bounded: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Check the numbers."""
        check_positive(self, "turning_radius", "spacing")
        check_not_negative(self, "cost_weight")
        if not isinstance(self.reverse_penalty, int | float) or not (
            self.reverse_penalty >= 1.0
        ):
            raise ValueError(
                f"reverse_penalty must be 1 or more, got {self.reverse_penalty!r}"
            )
        if (
            not isinstance(self.max_expansions, int)
            or isinstance(self.max_expansions, bool)
            or self.max_expansions < 1
        ):
            raise ValueError("max_expansions must be a whole number of 1 or more")

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap to search.

        Raises:
            ValueError: if there is none, since this planner cannot work without.
        """
        if costmap is None:
            raise ValueError("HybridAStar needs a costmap, but the world turns it off")
        self._costmap = costmap

    def plan(self, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """The path from start to goal, shape (N, 3); shape (0, 3) if there is none."""
        if self._costmap is None:
            raise ValueError("HybridAStar is not configured; call configure first")
        costmap = self._costmap
        blocked = costmap.data >= INSCRIBED
        for name, pose in (("start", start), ("goal", goal)):
            if not costmap.in_bounds(*pose.xyt[:2]):
                return no_path(
                    "HybridAStar",
                    "the %s (%.2f, %.2f) is off the map",
                    name,
                    *pose.xyt[:2],
                )
        cell = costmap.world_to_cell(*goal.xyt[:2])
        if blocked[cell]:
            return no_path(
                "HybridAStar",
                "the goal cell %s is blocked (cost %d), so the car cannot be there",
                cell,
                costmap.data[cell],
            )
        steps, opened = self._search(start.xyt, goal, blocked)
        if steps is None:
            return no_path(
                "HybridAStar",
                "no path from the start (%.2f, %.2f) to the goal (%.2f, %.2f) that "
                "bends no tighter than %.2f m; the search opened %d poses%s",
                *start.xyt[:2],
                *goal.xyt[:2],
                self.turning_radius,
                opened,
                "" if opened < self.max_expansions else " and stopped there",
            )
        return steps

    def _search(self, start, goal, blocked):
        """The path of arcs ended by a closing path, or None, and the poses opened."""
        costmap = self._costmap
        far = _distance_to(costmap, blocked, goal.xyt[:2])
        shot = (ReedsShepp if self.allow_reversing else Dubins)(
            self.turning_radius, self.spacing
        )
        first = tuple(float(v) for v in start)
        key0 = _key(costmap, first)
        # best[key] = (cost so far, parent key, the pose, the move that led to it)
        best = {key0: (0.0, None, first, None)}
        heap = [(far[key0[:2]] * CELL, 0, key0)]
        opened = 0
        while opened < self.max_expansions:
            if not heap:
                return None, opened
            opened += 1
            key = heapq.heappop(heap)[2]
            near = far[key[:2]] * CELL < NEAR * self.turning_radius
            if opened % EVERY == 1 or near:
                closing = self._close(shot, best[key][2], goal)
                if closing is not None:
                    return self._assemble(best, key, closing, goal), opened
            self._expand(key, best, heap, far)
        return None, opened

    def _expand(self, key, best, heap, far) -> None:
        """Open a pose: try each arc from it and keep the ones that are cheaper."""
        costmap = self._costmap
        kappas = (-1.0 / self.turning_radius, 0.0, 1.0 / self.turning_radius)
        moves = [(+1, k) for k in kappas]
        if self.allow_reversing:
            moves += [(-1, k) for k in kappas]
        price = 1.0 + self.cost_weight * costmap.data / MAX_NON_OBSTACLE
        cost, _, pose, last = best[key]
        for move in moves:
            found = self._arc(pose, move, price)
            if found is None:
                continue
            new, step_cost = found
            new_key = _key(costmap, new)
            ahead = far[new_key[:2]]
            if not math.isfinite(ahead):
                continue
            total = cost + step_cost + _change(last, move)
            if new_key not in best or total < best[new_key][0]:
                best[new_key] = (total, key, new, move)
                heapq.heappush(heap, (total + ahead * CELL, next(self._order), new_key))

    def _clear(self, points: np.ndarray):
        """The cells under some points if all are on the map and free, else None."""
        rows, cols = self._costmap.cells(points[:, :2])
        if not self._costmap.on_grid(rows, cols).all():
            return None
        if (self._costmap.data[rows, cols] >= INSCRIBED).any():
            return None
        return rows, cols

    def _arc(self, pose, move, price):
        """The pose at the end of an arc and its cost, or None if it is blocked."""
        direction, kappa = move
        count = math.ceil(STEP / self._costmap.resolution)
        distance = direction * STEP * np.arange(1, count + 1) / count
        points = arc_poses(pose, kappa, distance)
        cells = self._clear(points)
        if cells is None:
            return None
        length = STEP * float(price[cells].mean())
        if direction < 0:
            length *= self.reverse_penalty
        return tuple(float(v) for v in points[-1]), length

    def _close(self, shot, pose, goal):
        """The closing path from a pose to the goal if it is clear, else None."""
        path = shot.plan(Transform2D.from_xyt(pose), goal)
        return path if self._clear(path) is not None else None

    def _assemble(self, best, key, closing, goal) -> np.ndarray:
        """The arcs from the start to a pose, then the closing path."""
        chain = []
        while key is not None:
            _, parent, pose, move = best[key]
            chain.append((pose, move))
            key = parent
        chain.reverse()
        pieces = [np.array([chain[0][0]])]
        for (before, _), (_, move) in zip(chain, chain[1:]):
            direction, kappa = move
            count = max(1, math.ceil(STEP / self.spacing))
            distance = direction * STEP * np.arange(1, count + 1) / count
            pieces.append(arc_poses(before, kappa, distance))
        pieces.append(closing[1:])
        path = np.concatenate(pieces)
        path[:, 2] = wrap_angle(path[:, 2])
        path[-1] = goal.xyt
        return path


def _key(costmap: Costmap, pose) -> tuple[int, int, int]:
    """The cell and heading bin that a pose falls in."""
    row, col = costmap.world_to_cell(pose[0], pose[1])
    bin_ = round((pose[2] % (2.0 * math.pi)) / (2.0 * math.pi / BINS)) % BINS
    step = max(1, round(CELL / costmap.resolution))
    return row // step, col // step, bin_


def _change(last, move) -> float:
    """The extra cost of this move after the last one."""
    if last is None:
        return 0.0
    extra = CUSP if last[0] != move[0] else 0.0
    return extra + (TURN if last[1] != move[1] else 0.0)


def _coarse(costmap: Costmap, blocked) -> tuple[np.ndarray, int]:
    """The free cells of the search's grid, and how many costmap cells one covers."""
    step = max(1, round(CELL / costmap.resolution))
    rows, cols = blocked.shape
    height, width = -(-rows // step), -(-cols // step)
    padded = np.ones((height * step, width * step), bool)
    padded[:rows, :cols] = blocked
    covered = padded.reshape((height, step, width, step))
    return ~covered.any(axis=(1, 3)), step


def _distance_to(costmap: Costmap, blocked, goal_xy) -> dict:
    """The distance, in cells of the search, round blocked cells, from the goal."""
    free, step = _coarse(costmap, blocked)
    height, width = free.shape
    goal = costmap.world_to_cell(*goal_xy)
    start = (goal[0] // step, goal[1] // step)
    free[start] = True  # the goal cell is free in the costmap
    far = _Far()
    far[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        here, (r, c) = heapq.heappop(heap)
        if here > far[(r, c)]:
            continue
        for dr, dc, length in _AROUND:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width and free[nr, nc]):
                continue
            if dr and dc and not (free[r, nc] and free[nr, c]):
                continue
            there = here + length
            if there < far[(nr, nc)]:
                far[(nr, nc)] = there
                heapq.heappush(heap, (there, (nr, nc)))
    return far


class _Far(dict):
    """Distances by cell that read as infinity for a cell never reached."""

    def __missing__(self, key):
        """A cell the search did not reach is as far as can be."""
        return math.inf
