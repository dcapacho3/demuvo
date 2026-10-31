"""Pure pursuit: steer toward a point of the path a fixed distance ahead."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from robomath import wrap_angle

from ..models import Twist
from ..plugins import check_positive
from .base import Observation


@dataclass(frozen=True)
class PurePursuit:
    """Steer toward a point of the path a fixed distance ahead.

    Near the goal it slows down and stops; then, if `rotate_to_goal` is on, it
    turns on the spot to the goal's angle, as Nav2's Regulated Pure Pursuit does
    with `use_rotate_to_heading`.

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
    """

    speed: float = 0.5
    lookahead: float = 0.6
    approach_gain: float = 1.0
    turn_in_place_angle: float = 0.8
    turn_gain: float = 2.0
    rotate_speed: float = 1.0
    rotate_to_goal: bool = True

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

        Raises:
            ValueError: if the robot has no goal or the path is empty.
        """
        if obs.goal is None or len(obs.path) == 0:
            raise ValueError("PurePursuit needs a goal and a path")
        to_robot = obs.pose.inverse()
        distance = float(np.hypot(*(to_robot @ obs.goal.translation)))
        xy_tolerance, yaw_tolerance = obs.goal_tolerance
        if distance <= xy_tolerance:
            return self._turn_to_goal(obs, yaw_tolerance)
        x, y = self._target(to_robot @ obs.path[:, :2])
        bearing = math.atan2(y, x)
        if abs(bearing) > self.turn_in_place_angle:
            return Twist(wz=self._turn(bearing))
        vx = min(self.speed, self.approach_gain * distance)
        return Twist(vx=vx, wz=2.0 * vx * y / (x * x + y * y))

    def _target(self, path: np.ndarray) -> np.ndarray:
        """The point to steer toward, from the path in the robot's frame."""
        distance = np.hypot(path[:, 0], path[:, 1])
        nearest = int(np.argmin(distance))
        ahead = np.flatnonzero(distance[nearest:] >= self.lookahead)
        return path[nearest + int(ahead[0]) if ahead.size else -1]

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
