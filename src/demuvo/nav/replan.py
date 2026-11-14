"""Deciding when a robot that drives should plan again.

`Navigation` asks the `Replanner` at each controller run and plans again when it
gives a reason. The reasons, each a setting of the robot's `ReplanSpec`, are the
ones Nav2's behaviour trees use: a timer, a path the local costmap shows blocked,
a robot too far off its path, and a robot that stands still, and one more: a local
costmap that changed since the plan, as it does when another robot or a moving
obstacle comes into view or goes. The rules that hold
for every trigger are here too: no plan within `period` of the last, none for a
robot within reach of its goal, and a plan that failed is tried again on the
timer alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from robomath import Transform2D

from ..costmap import INSCRIBED, Costmap
from ..evaluation.judge import STOPPED, goal_error
from ..paths import along_path, distance_to_path
from ..world import ReplanSpec, Task

# How many cells a blocked cell may move between two local costmaps before the
# move counts as a change; see `changed_area`.
MARGIN = 2


@dataclass(frozen=True)
class Sight:
    """What the replanner looks at for one robot at one controller run.

    Attributes:
        time: simulated time in seconds.
        pose: the pose the controller sees.
        path: the path the robot follows, shape (N, 3); empty if none was found.
        task: the task the robot is on.
        planned_at: when the path was last planned, or last tried.
        failed: True if that plan found no path.
        velocity: how the robot moved in the last step, (vx, vy, wz).
        costmap: the robot's local costmap, or None.
    """

    time: float
    pose: Transform2D
    path: np.ndarray
    task: Task
    planned_at: float
    failed: bool
    velocity: np.ndarray
    costmap: Costmap | None


class Replanner:
    """Gives the reason a robot should plan again now, or None.

    Args:
        specs: the `ReplanSpec` of each robot by name; None for a robot that
            plans once for each task.
        stopped: speeds in m/s and rad/s below which a robot counts as still.
    """

    def __init__(
        self,
        specs: Mapping[str, ReplanSpec | None],
        stopped: tuple[float, float] = STOPPED,
    ) -> None:
        """Start with no robot standing still and no costmap seen."""
        self.specs = dict(specs)
        self._stopped = stopped
        self._still_since: dict[str, float] = {}
        self._seen: dict[str, Costmap | None] = {}

    def forget(self, name: str) -> None:
        """Forget how long a robot has stood still, when its path is replaced."""
        self._still_since.pop(name, None)

    def planned(self, name: str, costmap: Costmap | None) -> None:
        """Note the local costmap a robot planned on, which `changed` measures from.

        A robot that had none yet measures from the first one it gets.
        """
        self._seen[name] = costmap

    def reason(self, name: str, sight: Sight) -> str | None:
        """Why the robot should plan again now, in plain words, or None.

        A robot within its goal's distance tolerance never plans again: it is
        turning to the goal's angle or holding there. A robot whose last plan
        found nothing tries again every `period`, whatever the triggers say.
        """
        spec = self.specs.get(name)
        if spec is None:
            return None
        task = sight.task
        distance, _ = goal_error(sight.pose, task)
        if distance <= task.goal_tolerance[0]:
            self._still_since.pop(name, None)
            return None
        still_for = self._track_stillness(name, sight)
        if sight.time - sight.planned_at + 1e-9 < spec.period:
            return None
        if sight.failed:
            return "it has no path yet"
        if spec.timer:
            return f"{spec.period:g} s passed"
        if spec.blocked and sight.costmap is not None:
            if path_blocked(sight.path, sight.pose, sight.costmap):
                return "the path ahead is blocked"
        if spec.off_path > 0:
            off = distance_to_path(sight.pose.translation, sight.path)
            if off > spec.off_path:
                return f"it is {off:.2f} m off the path"
        if spec.changed > 0 and sight.costmap is not None:
            if self._seen.get(name) is None:
                self._seen[name] = sight.costmap  # the first costmap: measure from it
            area = changed_area(self._seen[name], sight.costmap)
            if area + 1e-9 >= spec.changed:
                return f"the local costmap changed over {area:.2f} square metres"
        if spec.stuck > 0 and still_for + 1e-9 >= spec.stuck:
            return f"it has stood still for {still_for:.1f} s"
        return None

    def _track_stillness(self, name: str, sight: Sight) -> float:
        """Seconds the robot has stood still, as of this run; 0 if it moves."""
        vx, vy, wz = sight.velocity
        still = math.hypot(vx, vy) <= self._stopped[0] and abs(wz) <= self._stopped[1]
        if not still:
            self._still_since.pop(name, None)
            return 0.0
        return sight.time - self._still_since.setdefault(name, sight.time)


def path_blocked(path: np.ndarray, pose: Transform2D, costmap: Costmap) -> bool:
    """True if the path, from its point nearest the robot on, crosses a blocked cell.

    The path is walked every half cell, so no cell it passes through is missed.
    Only the cells inside the costmap are known; a cell of cost `INSCRIBED` or
    more is blocked, as it is for the planners. An empty path is not blocked.
    """
    if len(path) == 0:
        return False
    points = along_path(path, costmap.resolution / 2)
    nearest = int(np.argmin(np.hypot(*(points - pose.translation).T)))
    rows, cols = costmap.cells(points[nearest:])
    height, width = costmap.data.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return bool((costmap.data[rows[inside], cols[inside]] >= INSCRIBED).any())


def changed_area(before: Costmap, after: Costmap) -> float:
    """Square metres that became blocked or clear between two local costmaps.

    Both are a robot's local costmap, so they share a resolution, and the window
    may have moved with the robot between them. A cell of cost `INSCRIBED` or
    more is blocked. Only the cells both windows cover are compared, so the
    window rolling on is not a change. A cell that became blocked counts only if
    it lies more than `MARGIN` cells from every cell blocked before, and one that
    became clear only if it lies that far from every cell blocked now: a wall
    seen a little differently as the robot moves is not a change, while a body
    that came or went in open ground is. Two windows that do not meet give 0.
    """
    res = after.resolution
    dr = round((after.origin[1] - before.origin[1]) / res)
    dc = round((after.origin[0] - before.origin[0]) / res)
    height, width = after.data.shape
    before_height, before_width = before.data.shape
    r0, r1 = max(0, -dr), min(height, before_height - dr)
    c0, c1 = max(0, -dc), min(width, before_width - dc)
    if r0 >= r1 or c0 >= c1:
        return 0.0
    now = after.data[r0:r1, c0:c1] >= INSCRIBED
    then = before.data[r0 + dr : r1 + dr, c0 + dc : c1 + dc] >= INSCRIBED
    appeared = now & ~_dilate(then, MARGIN)
    gone = then & ~_dilate(now, MARGIN)
    return float((appeared | gone).sum()) * res * res


def _dilate(cells: np.ndarray, count: int) -> np.ndarray:
    """The cells and every cell within `count` steps of one, sides or corners."""
    out = cells.copy()
    for _ in range(count):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= grown[:, :-1].copy()
        grown[:, :-1] |= grown[:, 1:].copy()
        out = grown
    return out
