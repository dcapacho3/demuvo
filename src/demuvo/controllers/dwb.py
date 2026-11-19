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

A robot that cannot turn in place (`Observation.limits` says so, as an ackermann
robot's do) gets a window shaped like its steering: every sampled turn rate is
pulled inside what the model allows at that sample's speed, so the trajectory
that is scored is the one the robot will drive, trajectories slower than
`car_min_vel_x` are thrown out so it turns on its least circle rather than inching,
and the turn on the spot at the goal is skipped. With `allow_reversing` such a
robot may also back up, and the path angle critic then counts facing away from its
aim as good as facing it, so a reverse trajectory toward the path is not punished.
A robot that can turn in place is driven as before.

A car follows a path that changes direction, as a Reeds-Shepp path does, one run at
a time: the path is cut to the run, only speeds in the run's direction are sampled,
and the local goal is the run's end. Near the path a heading critic also keeps the
car facing as the path does, which lands it on the goal's angle.

A robot that can move sideways (`Observation.limits.strafes`, as a mecanum robot's
does) gets a third axis, as in Nav2's DWB for a holonomic base: `vy_samples`
sideways speeds, reachable within `acc_lim_y` and no faster than `max_vel_y` or the
robot's own `max_vy`. Every (vx, vy, wz) is rolled out on its exact arc and pulled
inside the robot's limits as its model would pull it, so the scored trajectory is
the driven one. The critics are the same; the align critics still make it face
along the path, and it strafes to get onto the path and round what blocks it. A
robot that cannot strafe samples vy = 0 alone and is driven exactly as before.
"""

from __future__ import annotations

import dataclasses
import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import numpy as np
from robomath import rplus_se2, se2_to_xyt, wrap_angle, xyt_to_se2

from ..costmap import INSCRIBED, MAX_NON_OBSTACLE, Costmap
from ..models import Limits, Twist
from ..paths import along_path
from ..plugins import check_not_negative, check_positive
from .base import Observation, car_limits, pass_runs, path_runs, strafe_limits

LOGGER = logging.getLogger(__name__)
# How many times the path angle weight the heading critic counts, near the path.
HEADING_WEIGHT = 3.0
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
class DWB:  # pylint: disable=too-many-instance-attributes
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
        allow_reversing: lets it back up, at up to `max_vel_x`, on a path with no
            cusp; a path that backs is followed without it.
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
        path_angle_scale: weight per radian between a trajectory's end heading
            and the direction from there to the path point `max_vel_x * sim_time
            + forward_point_distance` ahead of the robot, beyond the end of any
            trajectory, for a robot that cannot turn in place. The
            distance critics cannot tell a small arc to the left from one to the
            right when the path lies behind; this one can, as MPPI's PathAngle
            critic does for a car, so the robot swings round on its least
            circle toward the path. Not used for a robot that turns in place.
        car_min_vel_x: the least |vx| of a trajectory for a robot that cannot
            turn in place, in m/s; 0 for none. Such a robot can only turn by
            driving, and every move away from a path that lies to its side
            scores worse than a smaller one, so without a floor it inches.
            With one it turns on its least circle at this speed instead. The
            floor is never above the fastest speed the window reaches.
        max_vel_y: largest sideways speed in m/s, either way, for a robot that
            can move sideways; the robot's own `max_vy` caps it too.
        acc_lim_y: sideways acceleration limit in m/s^2.
        vy_samples: how many sideways speeds it tries, for a robot that can move
            sideways; the window holds this many times as many trajectories.
    """

    max_vel_x: float = 0.5
    allow_reversing: bool = False
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
    path_angle_scale: float = 2.0
    car_min_vel_x: float = 0.1
    max_vel_y: float = 0.5
    acc_lim_y: float = 2.5
    vy_samples: int = 5
    # Settings only a robot that cannot turn in place uses, only one that can, and
    # only one that can move sideways; a form shows the ones that apply.
    car_only: ClassVar[tuple[str, ...]] = ("car_min_vel_x", "path_angle_scale")
    spot_only: ClassVar[tuple[str, ...]] = (
        "rotate_speed",
        "turn_gain",
        "rotate_to_goal",
    )
    strafe_only: ClassVar[tuple[str, ...]] = ("max_vel_y", "acc_lim_y", "vy_samples")
    _last: Twist = field(default_factory=Twist, init=False, repr=False, compare=False)
    _maps: tuple[tuple, _Maps] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _stuck_on: int = field(default=-1, init=False, repr=False, compare=False)
    _run: int = field(default=0, init=False, repr=False, compare=False)
    _changes: int = field(default=-1, init=False, repr=False, compare=False)

    @property
    def min_vel_x(self) -> float:
        """The slowest forward speed, in m/s: minus `max_vel_x` if it may back up."""
        return -self.max_vel_x if self.allow_reversing else 0.0

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
            "max_vel_y",
            "acc_lim_y",
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
            "path_angle_scale",
            "car_min_vel_x",
        )
        for name in ("vx_samples", "vtheta_samples", "vy_samples"):
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
        x, y, _ = obs.pose.xyt
        gx, gy, _ = obs.goal.xyt
        to_goal = math.hypot(gx - x, gy - y)
        car = car_limits(obs)
        obs, sign, last = self._run_of(obs, car)
        if to_goal <= obs.goal_tolerance[0] and last:
            arrived = self._arrive(obs, car, sign)
            if arrived is not None:
                return self._keep(arrived)
        vx, wz = self._window(obs.dt, sign)
        if car is not None:
            # Project the window onto the steering cone: a sample the model would
            # clip is scored as what the model would drive instead.
            wz = _within_cone(vx, wz, car)
        vy = None
        speed = self.max_vel_x
        strafe = strafe_limits(obs)
        if strafe is not None:
            vx, vy, wz = self._sideways(vx, wz, obs.dt, strafe)
            # A trajectory reaches farther on a diagonal.
            speed = math.hypot(speed, min(self.max_vel_y, strafe.max_vy))
        poses = rollout(
            obs.pose.xyt, vx, wz, self.sim_time, self.time_granularity, vy=vy
        )
        reach = speed * self.sim_time + self.forward_point_distance
        scores = self._score(
            poses, obs, align=to_goal > reach, car=car is not None, sign=sign
        )
        scores[_standing(vx, vy, wz)] = np.inf  # standing still gets nowhere
        if car is not None:
            # Inching gets nowhere either: a car turns by driving, so it must.
            floor = min(self.car_min_vel_x, float(np.abs(vx).max()))
            scores[np.abs(vx) < floor - 1e-9] = np.inf
        return self._keep(self._choose(scores, obs, (vx, vy, wz)))

    def _arrive(self, obs: Observation, car, sign) -> Twist | None:
        """The command at the goal's place, or None to go on along the path."""
        x, y, yaw = obs.pose.xyt
        gx, gy, gyaw = obs.goal.xyt
        yaw_tolerance = obs.goal_tolerance[1]
        if car is None:
            return self._turn_to_goal(gyaw - yaw, yaw_tolerance)
        # A car arrives at the goal's angle or not at all, so it finishes
        # its path instead of stopping at the first touch of the tolerance,
        # which on an arc would leave the last of the turn undriven. Once
        # the angle is met, or the goal is behind it, it stops where it is.
        behind = (sign or 1.0) * (
            (gx - x) * math.cos(yaw) + (gy - y) * math.sin(yaw)
        ) <= 0.0
        if abs(float(wrap_angle(gyaw - yaw))) <= yaw_tolerance or behind:
            return Twist()
        return None

    def _choose(self, scores: np.ndarray, obs: Observation, speeds) -> Twist:
        """The command of the best-scoring sample, or a stop if none is free."""
        vx, vy, wz = speeds
        best = int(np.argmin(scores))
        if not np.isfinite(scores[best]):
            if self._stuck_on != obs.path_changes:
                LOGGER.warning(
                    "DWB: no trajectory from (%.2f, %.2f) is free and leads on "
                    "along the path; the robot stops until one is",
                    obs.pose.xyt[0],
                    obs.pose.xyt[1],
                )
                self._stuck_on = obs.path_changes
            return Twist()
        if vy is None:
            return Twist(vx=float(vx[best]), wz=float(wz[best]))
        return Twist(vx=float(vx[best]), vy=float(vy[best]), wz=float(wz[best]))

    def _run_of(self, obs: Observation, car) -> tuple[Observation, float | None, bool]:
        """The observation of the run being followed, its direction and if it is last.

        A car follows a path that changes direction one run at a time: the path
        is cut to the run, which the window then drives only forwards or only
        backwards. A path that only goes forward, or any path for a robot that turns in
        place, is followed whole, with no direction, as before.
        """
        if obs.path_changes != self._changes:
            self._run, self._changes = 0, obs.path_changes
        runs = path_runs(obs.path)
        if car is None or runs == [(0, len(obs.path), 1.0)]:
            return obs, None, True
        local = obs.pose.inverse() @ obs.path[:, :2]
        now = pass_runs(runs, self._run, local, self.forward_point_distance)
        if now != self._run:
            self._run = now
            self._last = Twist()  # the car stops at a cusp
        start, stop, sign = runs[now]
        cut = dataclasses.replace(
            obs,
            path=obs.path[start:stop],
            path_changes=obs.path_changes * len(runs) + now,
        )
        return cut, sign, now == len(runs) - 1

    def _keep(self, command: Twist) -> Twist:
        """Remember the command as the robot's speed for the next window."""
        self._last = command
        return command

    def _window(
        self, dt: float, sign: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Every (vx, wz) pair reachable in dt from the last command, flat.

        `sign` is the direction of the run being followed, +1 or -1, which keeps
        the speeds to that side of zero; None leaves both open.
        """
        v0, w0 = self._last.vx, self._last.wz
        # A run backwards is driven backwards whether or not `allow_reversing` is on.
        floor = -self.max_vel_x if sign is not None and sign < 0 else self.min_vel_x
        v_low = max(floor, v0 - self.acc_lim_x * dt)
        v_high = min(self.max_vel_x, v0 + self.acc_lim_x * dt)
        if sign is not None:
            v_low, v_high = (
                (max(v_low, 0.0), v_high) if sign > 0 else (v_low, min(v_high, 0.0))
            )
        w_low = max(-self.max_vel_theta, w0 - self.acc_lim_theta * dt)
        w_high = min(self.max_vel_theta, w0 + self.acc_lim_theta * dt)
        vs = np.linspace(v_low, v_high, self.vx_samples)
        ws = np.linspace(w_low, w_high, self.vtheta_samples)
        if self.min_vel_x <= 0.0 <= v_high and v_low <= 0.0:
            vs = np.union1d(vs, [0.0])  # a turn on the spot is always in reach
        vx, wz = np.meshgrid(vs, ws, indexing="ij")
        return vx.ravel(), wz.ravel()

    def _sideways(
        self, vx: np.ndarray, wz: np.ndarray, dt: float, limits: Limits
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The window with a third axis, sideways speeds, for a robot that strafes.

        It tries `vy_samples` sideways speeds reachable in dt from the last
        command within `acc_lim_y`, no faster than `max_vel_y` or the robot's
        `max_vy`, and 0 when it is in reach, so every (vx, wz) pair of the
        window is still tried on its own. Each triple is then pulled inside the
        robot's limits as `Limits.clip` pulls a command, vx and vy by one factor
        and wz to its bound at that vx, so the trajectory scored is the one the
        robot drives.

        Returns:
            vx, vy and wz, flat, each pair of the window once for each vy.
        """
        bound = min(self.max_vel_y, limits.max_vy)
        y0 = self._last.vy
        low = max(-bound, y0 - self.acc_lim_y * dt)
        high = min(bound, y0 + self.acc_lim_y * dt)
        ys = np.linspace(low, high, self.vy_samples)
        if low <= 0.0 <= high:
            ys = np.union1d(ys, [0.0])
        vy = np.tile(ys, len(vx))
        vx, wz = np.repeat(vx, len(ys)), np.repeat(wz, len(ys))
        over = np.maximum(np.abs(vx) / limits.max_v, np.abs(vy) / limits.max_vy)
        shrink = np.where(over > 1.0, over, 1.0)
        vx, vy = vx / shrink, vy / shrink
        turn = np.array([limits.max_w(float(v)) for v in vx])
        return vx, vy, np.clip(wz, -turn, turn)

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
        inside = grid.on_grid(rows, cols)
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

    def _score(
        self,
        poses: np.ndarray,
        obs: Observation,
        align: bool,
        car: bool = False,
        sign: float | None = None,
    ) -> np.ndarray:
        """The total of the critics for each trajectory; inf if it is thrown out.

        `car` adds the path angle critic, for a robot that cannot turn in place.
        `sign` is the direction of the run being driven, if it has one: the
        critics then face along it, or against it for a run backwards.
        """
        maps = self._grids(obs)
        nearest = int(np.argmin(np.hypot(*(maps.points - obs.pose.translation).T)))
        end = poses[:, -1, :2]
        total = 0.0
        if car and self.path_angle_scale > 0.0:
            total = self._angle_cost(obs, maps, (nearest, end, poses), sign)
        ahead = end + (sign or 1.0) * self.forward_point_distance * np.column_stack(
            [np.cos(poses[:, -1, 2]), np.sin(poses[:, -1, 2])]
        )
        if maps.blocked[nearest:].any():
            # The path ahead runs through what the lidar sees: leave it, and let
            # the distance to the local goal, which flows round, lead.
            goal_end = lookup(maps.grid, maps.to_goal, end)
            total = total + self.goal_dist_scale * goal_end
            if align:
                total += self.goal_align_scale * _ahead(
                    maps.grid,
                    maps.to_goal,
                    ahead,
                    goal_end,
                    step=self.forward_point_distance,
                    end=end,
                )
        else:
            # Follow the path: keep near it and get along it.
            ahead_of_robot = maps.inside.copy()
            ahead_of_robot[:nearest] = False
            path_end = lookup(maps.grid, maps.to_path, end)
            total = total + self.path_dist_scale * path_end
            total += self.goal_dist_scale * left_along(end, maps, ahead_of_robot)
            if align:
                total += self.path_align_scale * _ahead(
                    maps.grid,
                    maps.to_path,
                    ahead,
                    path_end,
                    step=self.forward_point_distance,
                    end=end,
                )
                total += self.goal_align_scale * left_along(ahead, maps, ahead_of_robot)
        total += self.obstacle_scale * obstacle_cost(poses, maps.grid, maps.free)
        return np.nan_to_num(total, nan=np.inf, posinf=np.inf)

    def _angle_cost(self, obs: Observation, maps: _Maps, where, sign) -> np.ndarray:
        """The path angle critic: face the point ahead, and the path's own heading.

        Args:
            obs: the observation.
            maps: the grids of this window.
            where: the path point nearest the robot, the end point of each
                trajectory and the trajectories themselves.
            sign: the direction of the run being driven, or None.
        """
        nearest, end, poses = where
        reach = self.max_vel_x * self.sim_time + self.forward_point_distance
        aim = self._aim(maps, nearest, reach)
        to_aim = np.arctan2(aim[1] - end[:, 1], aim[0] - end[:, 0])
        if sign is not None:
            to_aim = to_aim + (math.pi if sign < 0 else 0.0)
        error = np.abs(wrap_angle(to_aim - poses[:, -1, 2]))
        if sign is None and self.allow_reversing:
            # The robot may back up, so facing away from the aim is as
            # good as facing it, as MPPI's PathAngle critic allows a car.
            error = np.minimum(error, math.pi - error)
        total = self.path_angle_scale * error
        # Near the path, face as it does: its headings end at the goal's angle,
        # so this lands the car there. Where it may back up on a path with no
        # cusp, facing against the path is as good as facing along it.
        apart = end[:, None, :] - obs.path[None, :, :2]
        gap = np.hypot(apart[..., 0], apart[..., 1])
        near = np.argmin(gap, axis=1)
        along = np.abs(wrap_angle(obs.path[near, 2] - poses[:, -1, 2]))
        if sign is None and self.allow_reversing:
            along = np.minimum(along, math.pi - along)
        closeness = np.clip(
            1.0 - gap.min(axis=1) / self.forward_point_distance, 0.0, 1.0
        )
        total = total + HEADING_WEIGHT * self.path_angle_scale * closeness * along
        return total

    def _aim(self, maps: _Maps, nearest: int, reach: float) -> np.ndarray:
        """The path point the path angle critic points at.

        The first point at least `reach` along the path from the one nearest
        the robot, so no trajectory of this window passes it, or the path's
        end; while the path ahead is blocked, the local goal, its last free
        point in the grid.
        """
        if maps.blocked[nearest:].any():
            open_ = np.flatnonzero(maps.inside & ~maps.blocked)
            if open_.size:
                return maps.points[open_[-1]]
        along = maps.arc[nearest:] - maps.arc[nearest]
        ahead = np.flatnonzero(along >= reach)
        return maps.points[nearest + int(ahead[0]) if ahead.size else -1]

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
    start: np.ndarray,
    vx: np.ndarray,
    wz: np.ndarray,
    sim_time: float,
    step: float,
    *,
    vy: np.ndarray | None = None,
) -> np.ndarray:
    """The poses of a robot under each constant body twist, every `step`.

    Each pose is exact, on the arc of the twist, as the models' `step` moves:
    radius vx / wz for a diff drive, hypot(vx, vy) / |wz| with a sideways speed
    vy. None for vy is no sideways speed. The first pose is one step from the
    start, and the last is at `sim_time`.

    Returns:
        shape (K, T, 3), (x, y, angle) for each of the K twists at each of T times.
    """
    count = max(1, math.ceil(round(sim_time / step, 9)))
    times = np.minimum(np.arange(1, count + 1) * step, sim_time)
    side = np.zeros((len(vx), count)) if vy is None else vy[:, None] * times
    # The body twist held for each time, so the pose moves along its exact arc.
    twist = np.stack([vx[:, None] * times, side, wz[:, None] * times], axis=-1)
    return se2_to_xyt(rplus_se2(xyt_to_se2(np.asarray(start, dtype=float)), twist))


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
    # cells are numbered along the rows and the grids are plain lists, since this loop
    # runs once for every cell it reaches and numpy indexing is slow one cell at a time
    open_cell, price = free.ravel().tolist(), price.ravel().tolist()
    distance = [math.inf] * (height * width)
    heap = []
    for row, col in sorted(set(zip(seeds[0].tolist(), seeds[1].tolist()))):
        if free[row, col]:
            distance[row * width + col] = 0.0
            heap.append((0.0, row * width + col))
    heapq.heapify(heap)
    moves = [
        (dr * width + dc, dr, dc, length, bool(dr and dc)) for dr, dc, length in _MOVES
    ]
    while heap:
        here, cell = heapq.heappop(heap)
        if here > distance[cell]:
            continue
        row, col = divmod(cell, width)
        for offset, dr, dc, length, corner in moves:
            if not (0 <= row + dr < height and 0 <= col + dc < width):
                continue
            ahead = cell + offset
            if not open_cell[ahead]:
                continue
            if corner and not (open_cell[cell + dr * width] and open_cell[cell + dc]):
                continue
            there = here + length * price[ahead]
            if there < distance[ahead]:
                distance[ahead] = there
                heapq.heappush(heap, (there, ahead))
    return np.array(distance).reshape(height, width)


def _blend(values, r0, c0, fx, fy):
    """The cell centres round each point mixed by distance, and where all are known."""
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
    return blend, known


def lookup(grid: Costmap, values: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The map at each point (..., 2), between cell centres; inf outside the grid.

    It interpolates the four cell centres round the point, so a small move gives
    a small change and nearby trajectories do not tie. Next to a cell that is
    inf it takes the value of the cell under the point instead.
    """
    flat = xy.reshape(-1, 2)
    rows, cols = grid.cells(flat)
    height, width = values.shape
    inside = grid.on_grid(rows, cols)
    out = np.full(len(flat), np.inf)
    out[inside] = values[rows[inside], cols[inside]]
    at = (flat - grid.origin) / grid.resolution - 0.5  # in units of cell centres
    low = np.floor(at).astype(int)
    frac = at - low
    c0, r0 = low[:, 0], low[:, 1]
    ok = inside & (r0 >= 0) & (r0 + 1 < height) & (c0 >= 0) & (c0 + 1 < width)
    blend, known = _blend(values, r0[ok], c0[ok], frac[ok, 0], frac[ok, 1])
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
    inside = grid.on_grid(rows, cols)
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


def _within_cone(vx, wz, car) -> np.ndarray:
    """The turn rates cut to what the car can steer at each speed."""
    bound = np.array([car.max_w(float(v)) for v in vx])
    return np.clip(wz, -bound, bound)


def _standing(vx, vy, wz) -> np.ndarray:
    """Which samples do not move the robot at all."""
    still = (np.abs(vx) < 1e-9) & (np.abs(wz) < 1e-9)
    if vy is not None:
        still &= np.abs(vy) < 1e-9  # a pure strafe moves
    return still


def _ahead(grid, values, ahead, at_end, *, step, end) -> np.ndarray:
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
    inside = grid.on_grid(rows, cols)
    rows, cols = np.where(inside, rows, 0), np.where(inside, cols, 0)
    cost = (grid.data[rows, cols] / MAX_NON_OBSTACLE).reshape(k, t)
    safe = (inside & free[rows, cols]).reshape(k, t).all(axis=1)
    return np.where(safe, np.minimum(cost, 1.0).mean(axis=1), np.inf)
