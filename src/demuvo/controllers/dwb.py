"""DWB: try many velocities, roll each one out, and drive the best that is safe.

The Dynamic Window Approach as Nav2's DWB controller does it. Each run it samples
the speeds and turn rates the robot can reach before the next run, rolls each pair
forward for `sim_time` seconds, and scores every trajectory with critics on the
local costmap, the grid the robot builds round itself from its lidar scans. The
lowest total wins.

- obstacle: a trajectory that comes within a cell of a cell of cost `INSCRIBED`
  or more would put the robot's body on an obstacle, so it is thrown out; the
  others pay for the mean cost they pass through.
- path distance: how far the trajectory's end is from the path.
- goal distance: how much path is left from where the trajectory ends.
- path align and goal align: the same two, for a point `forward_point_distance`
  ahead of the trajectory's end, so the robot faces along the path. They are
  left out once the goal is within reach of a trajectory and its point ahead,
  where they would pull the robot past it.

As in Nav2's map grid critics, the path distance is counted through free cells
only, from the free cells of the path. While the path ahead of the robot runs
through something the lidar sees, it cannot be followed: the path critics are left
out, and the goal critics measure instead the distance through free cells to the
local goal, the last free cell of the path in the window. That distance flows round
the block, so the robot leaves the path to pass it, and follows the path again once
the lidar shows the way ahead clear. Both distances cost more near obstacles, as
`AStar`'s steps do, so the robot keeps its distance.

At the goal it turns on the spot to the goal's angle, as Nav2's RotateToGoal
critic makes it do. There are no recovery behaviours: a robot that no trajectory
can move stays still until one can.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from robomath import wrap_angle

from ..costmap import INSCRIBED, MAX_NON_OBSTACLE, Costmap
from ..models import Twist
from ..plugins import check_not_negative, check_positive
from .base import Observation

LOGGER = logging.getLogger(__name__)
# Extra metres round the path of the blank grid a robot with no local costmap uses.
_BLANK_MARGIN = 2.0
_BLANK_RESOLUTION = 0.05


class _Maps(NamedTuple):
    """What the critics look up, made once for each costmap and path.

    Attributes:
        grid: the local costmap, or a blank grid round the path.
        free: the cells a robot's centre may be in, a cell clear of every
            cell of cost `INSCRIBED` or more.
        to_path: metres to the nearest free cell of the path, round obstacles.
        to_goal: metres to the local goal, the path's last free cell in the grid.
        points: points along the path, half a cell apart.
        arc: the length of path up to each point.
        inside: which points lie in the grid.
        blocked: which points lie in the grid on a cell that is not free.
    """

    grid: Costmap
    free: np.ndarray
    to_path: np.ndarray
    to_goal: np.ndarray
    points: np.ndarray
    arc: np.ndarray
    inside: np.ndarray
    blocked: np.ndarray


@dataclass
class DWB:
    """Choose the best safe velocity from the window of reachable ones.

    The window is what the accelerations allow from the last command over the
    time since the controller last ran. A diff-drive robot does what it is told,
    so the last command is its speed. A robot with no local costmap (no lidar)
    scores on a blank grid round its path, so it only follows the path. A
    trajectory that leaves the window, or ends where the path cannot be reached,
    is thrown out, so the window should hold `max_vel_x * sim_time +
    forward_point_distance` on every side of the robot. If every trajectory that
    moves is thrown out, the robot stops until one is free; the log says so once
    for each path.

    Attributes:
        max_vel_x: largest forward speed in m/s.
        min_vel_x: smallest forward speed in m/s; below 0 lets it back up.
        max_vel_theta: largest turn rate in rad/s, either way.
        acc_lim_x: forward acceleration limit in m/s^2.
        acc_lim_theta: turn acceleration limit in rad/s^2.
        vx_samples: how many forward speeds it tries.
        vtheta_samples: how many turn rates it tries.
        sim_time: how far ahead each trajectory is rolled out, in seconds.
        time_granularity: seconds between two checked points of a trajectory.
        obstacle_scale: weight of the mean cost along a trajectory, 0 to 1.
        path_dist_scale: weight per metre from the trajectory's end to the path.
        goal_dist_scale: weight per metre from the trajectory's end to the
            local goal.
        path_align_scale: weight per metre from the point ahead to the path.
        goal_align_scale: weight per metre from the point ahead to the local goal.
        forward_point_distance: how far ahead of the trajectory's end the align
            critics look, in metres.
        cost_weight: how much a cell's cost raises the price of a step of the
            distance maps, as in `AStar`, so they keep off obstacles; 0 or more.
        rotate_speed: largest turn rate in rad/s when turning to the goal's angle.
        turn_gain: turn rate per radian of error when turning to the goal's angle.
        rotate_to_goal: turn to the goal's angle once at the goal.
    """

    max_vel_x: float = 0.5
    min_vel_x: float = 0.0
    max_vel_theta: float = 1.0
    acc_lim_x: float = 2.5
    acc_lim_theta: float = 3.2
    vx_samples: int = 20
    vtheta_samples: int = 20
    sim_time: float = 1.0
    time_granularity: float = 0.05
    obstacle_scale: float = 2.0
    path_dist_scale: float = 1.0
    goal_dist_scale: float = 1.0
    path_align_scale: float = 1.0
    goal_align_scale: float = 0.75
    forward_point_distance: float = 0.325
    cost_weight: float = 6.0
    rotate_speed: float = 1.0
    turn_gain: float = 2.0
    rotate_to_goal: bool = True
    _last: Twist = field(default_factory=Twist, init=False, repr=False, compare=False)
    _maps: tuple[tuple, _Maps] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _stuck_on: int = field(default=-1, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the numbers."""
        check_positive(
            self,
            "max_vel_x",
            "max_vel_theta",
            "acc_lim_x",
            "acc_lim_theta",
            "sim_time",
            "time_granularity",
            "rotate_speed",
            "turn_gain",
        )
        check_not_negative(
            self,
            "obstacle_scale",
            "path_dist_scale",
            "goal_dist_scale",
            "path_align_scale",
            "goal_align_scale",
            "forward_point_distance",
            "cost_weight",
        )
        if not math.isfinite(self.min_vel_x) or self.min_vel_x > self.max_vel_x:
            raise ValueError(
                f"min_vel_x must be a number no more than max_vel_x, "
                f"got {self.min_vel_x!r}"
            )
        for name in ("vx_samples", "vtheta_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a whole number of 1 or more")

    def command(self, obs: Observation) -> Twist:
        """The best safe command, or a turn to the goal's angle at the goal.

        Raises:
            ValueError: if there is a path but no goal.
        """
        if len(obs.path) == 0:
            return self._keep(Twist())
        if obs.goal is None:
            raise ValueError("DWB needs a goal to follow a path")
        x, y, yaw = obs.pose.xyt
        gx, gy, gyaw = obs.goal.xyt
        xy_tolerance, yaw_tolerance = obs.goal_tolerance
        to_goal = math.hypot(gx - x, gy - y)
        if to_goal <= xy_tolerance:
            return self._keep(self._turn_to_goal(gyaw - yaw, yaw_tolerance))
        vx, wz = self._window(obs.dt)
        poses = rollout(obs.pose.xyt, vx, wz, self.sim_time, self.time_granularity)
        reach = self.max_vel_x * self.sim_time + self.forward_point_distance
        scores = self._score(poses, obs, align=to_goal > reach)
        still = (np.abs(vx) < 1e-9) & (np.abs(wz) < 1e-9)
        scores[still] = np.inf  # standing still gets nowhere
        best = int(np.argmin(scores))
        if not np.isfinite(scores[best]):
            if self._stuck_on != obs.path_changes:
                LOGGER.warning(
                    "DWB: no trajectory from (%.2f, %.2f) is free and leads on "
                    "along the path; the robot stops until one is",
                    x,
                    y,
                )
                self._stuck_on = obs.path_changes
            return self._keep(Twist())
        return self._keep(Twist(vx=float(vx[best]), wz=float(wz[best])))

    def _keep(self, command: Twist) -> Twist:
        """Remember the command as the robot's speed for the next window."""
        self._last = command
        return command

    def _window(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Every (vx, wz) pair reachable in dt from the last command, flat."""
        v0, w0 = self._last.vx, self._last.wz
        v_low = max(self.min_vel_x, v0 - self.acc_lim_x * dt)
        v_high = min(self.max_vel_x, v0 + self.acc_lim_x * dt)
        w_low = max(-self.max_vel_theta, w0 - self.acc_lim_theta * dt)
        w_high = min(self.max_vel_theta, w0 + self.acc_lim_theta * dt)
        vs = np.linspace(v_low, v_high, self.vx_samples)
        ws = np.linspace(w_low, w_high, self.vtheta_samples)
        if self.min_vel_x <= 0.0 <= v_high and v_low <= 0.0:
            vs = np.union1d(vs, [0.0])  # a turn on the spot is always in reach
        vx, wz = np.meshgrid(vs, ws, indexing="ij")
        return vx.ravel(), wz.ravel()

    def _grids(self, obs: Observation) -> _Maps:
        """The grid, its free cells, the path through it and the distance maps.

        They are made again only when the costmap or the path changes, which is
        once every local costmap update.
        """
        if self._maps is not None:
            (costmap, changes), maps = self._maps
            if costmap is obs.costmap and changes == obs.path_changes:
                return maps
        grid = obs.costmap if obs.costmap is not None else blank_grid(obs.path)
        free = ~grow(grid.data >= INSCRIBED)
        points = along_path(obs.path, grid.resolution / 2)
        rows, cols = grid.cells(points)
        height, width = free.shape
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows, cols = np.where(inside, rows, 0), np.where(inside, cols, 0)
        open_ = inside & free[rows, cols]
        seeds = rows[open_], cols[open_]
        price = 1.0 + self.cost_weight * grid.data / MAX_NON_OBSTACLE
        goal = seeds[0][-1:], seeds[1][-1:]
        maps = _Maps(
            grid=grid,
            free=free,
            to_path=wavefront(free, seeds, price) * grid.resolution,
            to_goal=wavefront(free, goal, price) * grid.resolution,
            points=points,
            arc=np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points.T)))]),
            inside=inside,
            blocked=inside & ~open_,
        )
        self._maps = ((obs.costmap, obs.path_changes), maps)
        return maps

    def _score(self, poses: np.ndarray, obs: Observation, align: bool) -> np.ndarray:
        """The total of the critics for each trajectory; inf if it is thrown out."""
        maps = self._grids(obs)
        nearest = int(np.argmin(np.hypot(*(maps.points - obs.pose.translation).T)))
        end = poses[:, -1, :2]
        ahead = end + self.forward_point_distance * np.column_stack(
            [np.cos(poses[:, -1, 2]), np.sin(poses[:, -1, 2])]
        )
        if maps.blocked[nearest:].any():
            # The path ahead runs through what the lidar sees: leave it, and let
            # the distance to the local goal, which flows round, lead.
            goal_end = lookup(maps.grid, maps.to_goal, end)
            total = self.goal_dist_scale * goal_end
            if align:
                total += self.goal_align_scale * _ahead(
                    maps.grid,
                    maps.to_goal,
                    ahead,
                    goal_end,
                    self.forward_point_distance,
                    end,
                )
        else:
            # Follow the path: keep near it and get along it.
            ahead_of_robot = maps.inside.copy()
            ahead_of_robot[:nearest] = False
            path_end = lookup(maps.grid, maps.to_path, end)
            total = self.path_dist_scale * path_end
            total += self.goal_dist_scale * left_along(end, maps, ahead_of_robot)
            if align:
                total += self.path_align_scale * _ahead(
                    maps.grid,
                    maps.to_path,
                    ahead,
                    path_end,
                    self.forward_point_distance,
                    end,
                )
                total += self.goal_align_scale * left_along(ahead, maps, ahead_of_robot)
        total += self.obstacle_scale * obstacle_cost(poses, maps.grid, maps.free)
        return np.nan_to_num(total, nan=np.inf, posinf=np.inf)

    def _turn_to_goal(self, error: float, tolerance: float) -> Twist:
        """Stop, or turn on the spot toward the goal's angle."""
        error = float(wrap_angle(error))
        if not self.rotate_to_goal or abs(error) <= tolerance:
            return Twist()
        limit = min(self.rotate_speed, self.max_vel_theta)
        return Twist(wz=max(-limit, min(limit, self.turn_gain * error)))


def blank_grid(path: np.ndarray) -> Costmap:
    """A grid of free cells round the path, for a robot with no local costmap."""
    low = path[:, :2].min(axis=0) - _BLANK_MARGIN
    high = path[:, :2].max(axis=0) + _BLANK_MARGIN
    cols, rows = np.ceil((high - low) / _BLANK_RESOLUTION).astype(int)
    return Costmap(
        np.zeros((rows, cols), dtype=np.uint8),
        _BLANK_RESOLUTION,
        (float(low[0]), float(low[1])),
    )


def rollout(
    start: np.ndarray, vx: np.ndarray, wz: np.ndarray, sim_time: float, step: float
) -> np.ndarray:
    """The poses of a diff-drive robot under each constant (vx, wz), every `step`.

    Each pose is exact, on the arc of radius vx / wz, as `DiffDrive.step` moves.
    The first pose is one step from the start, and the last is at `sim_time`.

    Returns:
        shape (K, T, 3), (x, y, angle) for each of the K pairs at each of T times.
    """
    count = max(1, math.ceil(round(sim_time / step, 9)))
    times = np.minimum(np.arange(1, count + 1) * step, sim_time)
    x0, y0, yaw0 = start
    v, w = vx[:, None], wz[:, None]
    turned = w * times
    straight = np.abs(w) < 1e-9
    safe_w = np.where(straight, 1.0, w)
    # Along the heading and to its left, in the start's frame.
    forward = np.where(straight, v * times, v * np.sin(turned) / safe_w)
    left = np.where(straight, 0.0, v * (1.0 - np.cos(turned)) / safe_w)
    cos, sin = math.cos(yaw0), math.sin(yaw0)
    return np.stack(
        [
            x0 + cos * forward - sin * left,
            y0 + sin * forward + cos * left,
            yaw0 + np.broadcast_to(turned, forward.shape),
        ],
        axis=-1,
    )


def along_path(path: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the path's polyline, no more than `spacing` apart, in order."""
    xy = path[:, :2]
    if len(xy) < 2:
        return xy.copy()
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    at = np.linspace(0.0, along[-1], max(2, math.ceil(along[-1] / spacing) + 1))
    return np.column_stack(
        [np.interp(at, along, xy[:, 0]), np.interp(at, along, xy[:, 1])]
    )


def grow(blocked: np.ndarray) -> np.ndarray:
    """The blocked cells and every cell that touches one, sides or corners.

    A cell of cost `INSCRIBED` holds a centre less than the robot's radius from
    an obstacle as measured between cell centres, so a centre near a corner can
    be up to half a cell closer. One more cell all round keeps it clear.
    """
    out = blocked.copy()
    out[1:] |= blocked[:-1]
    out[:-1] |= blocked[1:]
    grown = out.copy()
    grown[:, 1:] |= out[:, :-1]
    grown[:, :-1] |= out[:, 1:]
    return grown


# The eight moves to a neighbouring cell as (row step, column step, length).
_MOVES = tuple(
    (dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
)


def wavefront(
    free: np.ndarray,
    seeds: tuple[np.ndarray, np.ndarray],
    price: np.ndarray | None = None,
) -> np.ndarray:
    """Distance in cells from the nearest seed to each cell, through free cells.

    It moves to the eight neighbours, a diagonal costing the square root of 2, and
    never cuts the corner of a cell that is not free. A step's length is times the
    `price` of the cell it enters, 1 everywhere if None. A seed that is not free
    is left out, as Nav2's map grid critics do, so the distance goes round an
    obstacle on the path. A cell no free seed reaches is inf.
    """
    if price is None:
        price = np.ones(free.shape)
    height, width = free.shape
    distance = np.full(free.shape, np.inf)
    heap = []
    for row, col in sorted(set(zip(seeds[0].tolist(), seeds[1].tolist()))):
        if free[row, col]:
            distance[row, col] = 0.0
            heap.append((0.0, row, col))
    heapq.heapify(heap)
    while heap:
        here, row, col = heapq.heappop(heap)
        if here > distance[row, col]:
            continue
        for dr, dc, length in _MOVES:
            r, c = row + dr, col + dc
            if not (0 <= r < height and 0 <= c < width and free[r, c]):
                continue
            if dr and dc and not (free[row + dr, col] and free[row, col + dc]):
                continue
            there = here + length * price[r, c]
            if there < distance[r, c]:
                distance[r, c] = there
                heapq.heappush(heap, (there, r, c))
    return distance


def lookup(grid: Costmap, values: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The map at each point (..., 2), between cell centres; inf outside the grid.

    It interpolates the four cell centres round the point, so a small move gives
    a small change and nearby trajectories do not tie. Next to a cell that is
    inf it takes the value of the cell under the point instead.
    """
    flat = xy.reshape(-1, 2)
    rows, cols = grid.cells(flat)
    height, width = values.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    out = np.full(len(flat), np.inf)
    out[inside] = values[rows[inside], cols[inside]]
    at = (flat - grid.origin) / grid.resolution - 0.5  # in units of cell centres
    low = np.floor(at).astype(int)
    frac = at - low
    c0, r0 = low[:, 0], low[:, 1]
    ok = inside & (r0 >= 0) & (r0 + 1 < height) & (c0 >= 0) & (c0 + 1 < width)
    r0, c0, fx, fy = r0[ok], c0[ok], frac[ok, 0], frac[ok, 1]
    corners = np.stack(
        [
            values[r0, c0],
            values[r0, c0 + 1],
            values[r0 + 1, c0],
            values[r0 + 1, c0 + 1],
        ]
    )
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    known = np.isfinite(corners).all(axis=0)
    blend = (np.where(np.isfinite(corners), corners, 0.0) * weights).sum(axis=0)
    out[np.flatnonzero(ok)[known]] = blend[known]
    return out.reshape(xy.shape[:-1])


def left_along(points: np.ndarray, maps: _Maps, usable: np.ndarray) -> np.ndarray:
    """Path left to the local goal from the usable path point nearest each point."""
    index = np.flatnonzero(usable)
    if index.size == 0:
        return np.full(len(points), np.inf)
    candidates = maps.points[index]
    gaps = np.hypot(
        points[:, None, 0] - candidates[None, :, 0],
        points[:, None, 1] - candidates[None, :, 1],
    )
    return maps.arc[index[-1]] - maps.arc[index[np.argmin(gaps, axis=1)]]


def downhill(grid: Costmap, values: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The heading from each point's cell to its neighbour lowest on the map.

    Every cell a free seed reaches has a lower neighbour, unless it is a seed;
    a point there or outside the grid gets nan.
    """
    rows, cols = grid.cells(xy)
    height, width = values.shape
    best = np.full(len(xy), np.inf)
    heading = np.full(len(xy), np.nan)
    here = np.full(len(xy), np.inf)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    here[inside] = values[rows[inside], cols[inside]]
    for dr, dc, _ in _MOVES:
        r, c = rows + dr, cols + dc
        ok = inside & (r >= 0) & (r < height) & (c >= 0) & (c < width)
        value = np.full(len(xy), np.inf)
        value[ok] = values[r[ok], c[ok]]
        lower = (value < best) & (value < here)
        best = np.where(lower, value, best)
        heading = np.where(lower, math.atan2(dr, dc), heading)
    return heading


def _ahead(grid, values, ahead, at_end, step, end) -> np.ndarray:
    """The map under each point ahead, and a fair price for one that is blocked.

    A point ahead in an obstacle makes a heading bad, not the whole trajectory,
    as in Nav2's align critics, since a robot beside a wall may face it. It
    counts as the end plus `step`, worse than any free point ahead, and up to
    `step` more the farther the heading is from the way the map falls at the
    end, so a robot that faces a wall still knows which way to turn.
    """
    value = lookup(grid, values, ahead)
    heading = np.arctan2(ahead[:, 1] - end[:, 1], ahead[:, 0] - end[:, 0])
    off = 1.0 - np.cos(heading - downhill(grid, values, end))
    blocked = at_end + step * (1.0 + np.nan_to_num(off, nan=1.0) / 2.0)
    return np.where(np.isfinite(value), value, blocked)


def obstacle_cost(poses: np.ndarray, grid: Costmap, free: np.ndarray) -> np.ndarray:
    """The mean cost along each trajectory, from 0 to 1; inf if it leaves `free`.

    A trajectory that leaves the grid is thrown out too, since nothing is known
    of what lies there.
    """
    k, t = poses.shape[:2]
    rows, cols = grid.cells(poses[..., :2].reshape(-1, 2))
    height, width = grid.data.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows, cols = np.where(inside, rows, 0), np.where(inside, cols, 0)
    cost = (grid.data[rows, cols] / MAX_NON_OBSTACLE).reshape(k, t)
    safe = (inside & free[rows, cols]).reshape(k, t).all(axis=1)
    return np.where(safe, np.minimum(cost, 1.0).mean(axis=1), np.inf)
