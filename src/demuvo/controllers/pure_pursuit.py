"""Pure pursuit: steer toward a point of the path a fixed distance ahead."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from robomath import Transform2D, wrap_angle

from ..models import Limits, Twist
from ..plugins import check_positive
from .base import Observation, car_limits


@dataclass
class PurePursuit:
    """Steer toward a point of the path a fixed distance ahead.

    Near the goal it slows down and stops; then, if `rotate_to_goal` is on, it
    turns on the spot to the goal's angle, as Nav2's Regulated Pure Pursuit does
    with `use_rotate_to_heading`.

    A robot that cannot turn in place (`Observation.limits` says so, as an
    ackermann robot's do) is driven as a car: the turn rate is kept inside what
    the steering allows at the speed, a bearing error too large to steer out of
    is driven out of on full lock instead of turned out of on the spot, and the
    goal's angle is left as the robot arrives, since Nav2 too turns
    `use_rotate_to_heading` off for a car. A robot that can turn in place is
    driven as before, whatever its limits say.

    Such a robot follows a path that changes direction, as a Reeds-Shepp path
    does at its cusps, the way Nav2's Regulated Pure Pursuit does: the path is
    split where the car's heading and its motion disagree into runs driven
    forward or backward, each run is followed in its direction with the point
    ahead taken from that run alone, and the next run starts once the end of
    this one is passed. With `allow_reversing` a path with no cusp whose target
    lies behind the robot is also backed toward, on the arc through the target.

    The point ahead is the first path point at least `lookahead` away from the
    robot, searching from the path point nearest to it. Shortcut: a path that
    crosses itself near the robot can fool the search; upgrade when a planner
    makes such paths.

    Attributes:
        speed: forward speed in m/s on the way.
        lookahead: distance in metres to the point it steers toward.
        approach_gain: forward speed per metre left to the goal, in 1/s; it
            slows the robot near the goal.
        turn_in_place_angle: bearing error in radians above which it turns on
            the spot before it drives.
        turn_gain: turn rate per radian of error when turning on the spot, in 1/s.
        rotate_speed: largest turn rate in rad/s when turning on the spot.
        rotate_to_goal: turn to the goal's angle once at the goal.
        allow_reversing: back toward a target behind the robot on a path with no
            cusp, instead of going round to it forwards, as Nav2's Regulated
            Pure Pursuit does. A path that backs is followed without it. Only a
            robot that cannot turn in place uses it; see the class note.
    """

    speed: float = 0.5
    lookahead: float = 0.6
    approach_gain: float = 1.0
    turn_in_place_angle: float = 0.8
    turn_gain: float = 2.0
    rotate_speed: float = 1.0
    rotate_to_goal: bool = True
    allow_reversing: bool = False
    # Settings only a robot that cannot turn in place uses, and only one that can;
    # a form shows the ones that apply.
    car_only: ClassVar[tuple[str, ...]] = ("allow_reversing",)
    spot_only: ClassVar[tuple[str, ...]] = (
        "rotate_speed",
        "turn_gain",
        "rotate_to_goal",
    )
    _run: int = field(default=0, init=False, repr=False, compare=False)
    _changes: int = field(default=-1, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the numbers."""
        check_positive(
            self,
            "speed",
            "lookahead",
            "approach_gain",
            "turn_in_place_angle",
            "turn_gain",
            "rotate_speed",
        )

    def command(self, obs: Observation) -> Twist:
        """Drive along the path, then stop, then turn to the goal's angle.

        With an empty path there is nothing to follow, so the robot stays still.

        Raises:
            ValueError: if there is a path but no goal.
        """
        if len(obs.path) == 0:
            return Twist()
        if obs.goal is None:
            raise ValueError("PurePursuit needs a goal to follow a path")
        to_robot = obs.pose.inverse()
        distance = float(np.hypot(*(to_robot @ obs.goal.translation)))
        xy_tolerance, yaw_tolerance = obs.goal_tolerance
        car = car_limits(obs)
        reverses = self.allow_reversing or any(s < 0.0 for *_, s in _runs(obs.path))
        if distance <= xy_tolerance:
            if car is None:
                return self._turn_to_goal(obs, yaw_tolerance)
            # A car cannot turn to the goal's angle on the spot: it arrives
            # aligned or not at all, so it finishes its path instead of
            # stopping at the first touch of the tolerance, which on an arc
            # would leave the last of the turn undriven. Once the angle is
            # met, or the goal is behind it, it stops where it is.
            error = float(wrap_angle(obs.goal.xyt[2] - obs.pose.xyt[2]))
            passed = (to_robot @ obs.goal.translation)[0] <= 0.0
            if abs(error) <= yaw_tolerance:
                return Twist()
            # With reversing, whether the goal is passed depends on which way
            # the last run drives; `_follow_runs` judges it.
            if passed and not reverses:
                return Twist()
        if car is not None and reverses:
            return self._follow_runs(obs, car, to_robot, distance)
        x, y = self._target(to_robot @ obs.path[:, :2])
        bearing = math.atan2(y, x)
        vx = min(self.speed, self.approach_gain * distance)
        if abs(bearing) > self.turn_in_place_angle:
            if car is None:
                return Twist(wz=self._turn(bearing))
            return self._swing(car, vx, bearing)
        if x == 0.0 and y == 0.0:
            # On the path's last point but short of the goal - no direction to
            # steer, so stand still instead of dividing by zero.
            return Twist()
        wz = 2.0 * vx * y / (x * x + y * y)
        if car is not None:
            bound = car.max_w(vx)
            wz = max(-bound, min(bound, wz))
        return Twist(vx=vx, wz=wz)

    def _target(self, path: np.ndarray) -> np.ndarray:
        """The point to steer toward, from the path in the robot's frame."""
        distance = np.hypot(path[:, 0], path[:, 1])
        nearest = int(np.argmin(distance))
        ahead = np.flatnonzero(distance[nearest:] >= self.lookahead)
        return path[nearest + int(ahead[0]) if ahead.size else -1]

    def _swing(self, limits: Limits, vx: float, bearing: float) -> Twist:
        """Drive on full lock toward the side the target is on.

        A car cannot turn on the spot, so it turns on its least circle instead:
        forward at vx with the largest turn rate the steering gives at that
        speed. A target straight behind (bearing pi) goes round to the left.
        """
        vx = min(self.speed, max(vx, self.approach_gain * self.lookahead))
        return Twist(vx=vx, wz=math.copysign(limits.max_w(vx), bearing))

    def _follow_runs(
        self, obs: Observation, limits: Limits, to_robot: Transform2D, distance: float
    ) -> Twist:
        """Follow the current run of the path in its direction; see the class note."""
        if obs.path_changes != self._changes:
            self._run, self._changes = 0, obs.path_changes
        runs = _runs(obs.path)
        local = to_robot @ obs.path[:, :2]
        while self._run < len(runs) - 1:
            start, stop, sign = runs[self._run]
            end_x, end_y = local[stop - 1]
            # The run is done once its last point is passed, close by.
            if math.hypot(end_x, end_y) <= self.lookahead and sign * end_x <= 0.0:
                self._run += 1
            else:
                break
        start, stop, sign = runs[self._run]
        if self._run == len(runs) - 1 and distance <= obs.goal_tolerance[0]:
            # On the last run and at the goal: stop once it is passed in the
            # run's own direction; before that, finish the run for its angle.
            if sign * (to_robot @ obs.goal.translation)[0] <= 0.0:
                return Twist()
        x, y = self._target(local[start:stop])
        if x == 0.0 and y == 0.0:
            return Twist()
        if self._run == len(runs) - 1:
            vx = min(self.speed, self.approach_gain * distance)
        else:
            # Slow into the cusp as into the goal, but no slower than at the
            # goal's distance tolerance, so the car passes it instead of
            # creeping up to it forever; the next run then starts on it.
            left = math.hypot(*local[stop - 1])
            floor = self.approach_gain * obs.goal_tolerance[0]
            vx = min(self.speed, max(self.approach_gain * left, floor))
        # A backward run is backed along. So is a target behind the robot on a
        # forward run: a path from a planner that is not for cars has its
        # headings along its motion, so a goal behind reads as a forward run
        # the car would have to turn round for, and backing to it is the point
        # of allowing reverse.
        if sign < 0.0 or x < 0.0:
            return self._reverse(limits, vx, x, y)
        bearing = math.atan2(y, x)
        if abs(bearing) > self.turn_in_place_angle:
            return self._swing(limits, vx, bearing)
        wz = 2.0 * vx * y / (x * x + y * y)
        bound = limits.max_w(vx)
        return Twist(vx=vx, wz=max(-bound, min(bound, wz)))

    def _reverse(self, limits: Limits, vx: float, x: float, y: float) -> Twist:
        """Back toward a target behind the robot, on the arc through it.

        The same circle pure pursuit steers along forwards, driven the other
        way: the curvature is 2y over the target's squared distance, and the
        turn rate is the (negative) speed times it, inside the steering cone.
        """
        vx = -vx
        wz = 2.0 * vx * y / (x * x + y * y)
        bound = limits.max_w(vx)
        return Twist(vx=vx, wz=max(-bound, min(bound, wz)))

    def _turn(self, error: float) -> float:
        """The turn rate for an angle error, limited to `rotate_speed`."""
        return max(-self.rotate_speed, min(self.rotate_speed, self.turn_gain * error))

    def _turn_to_goal(self, obs: Observation, tolerance: float) -> Twist:
        """Stop, or turn on the spot to the goal's angle."""
        if not self.rotate_to_goal:
            return Twist()
        error = float(wrap_angle(obs.goal.xyt[2] - obs.pose.xyt[2]))
        if abs(error) <= tolerance:
            return Twist()
        return Twist(wz=self._turn(error))


def _runs(path: np.ndarray) -> list[tuple[int, int, float]]:
    """The path split where its direction of travel changes.

    A point moves forward if its heading agrees with the step to the next
    point, backward if not; the last point moves as the one before it. Each run
    is (first index, one past the last index, +1.0 or -1.0). A path of one
    point is one forward run.
    """
    if len(path) < 2:
        return [(0, len(path), 1.0)]
    step = np.diff(path[:, :2], axis=0)
    along = np.cos(path[:-1, 2]) * step[:, 0] + np.sin(path[:-1, 2]) * step[:, 1]
    signs = np.where(along < 0.0, -1.0, 1.0)
    runs, start = [], 0
    for i in range(1, len(signs)):
        if signs[i] != signs[start]:
            runs.append((start, i + 1, float(signs[start])))
            start = i
    runs.append((start, len(path), float(signs[start])))
    return runs
