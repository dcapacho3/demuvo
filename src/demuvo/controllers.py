"""Controllers - the code that decides how a robot follows its path.

A controller is a class with one method, `command(obs)`, that returns the Twist
for this step. It sees only the `Observation`, never the simulation, so the
same class could run in a ROS node that gets /odom, /plan and /scan. The world
file names it as `controller = "module:Class"`; demuvo builds the class once for
each robot that names it, with the robot's `controller_params` as keyword
arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from robomath import Transform2D, wrap_angle

from .models import Twist
from .plugins import check_positive
from .poses import Pose


@dataclass(frozen=True, eq=False)
class Observation:
    """What a controller knows at one step.

    Attributes:
        time: simulated time in seconds.
        dt: length of a step in seconds.
        pose: where the robot believes it is. Today this is the true pose, with
            no noise, because demuvo has no odometry model yet; it will be the
            estimate of the odometry or of a filter, and the true pose stays in
            the simulation for collisions and scoring.
        path: the path to follow, shape (N, 3) of (x, y, angle); empty if the
            robot has no goal.
        goal: the goal as (x, y, angle), or None.
        goal_tolerance: distance in metres and angle in radians within which
            the robot counts as at its goal.
    """

    time: float
    dt: float
    pose: Transform2D
    path: np.ndarray
    goal: Pose | None
    goal_tolerance: tuple[float, float]


class Controller(Protocol):
    """What every controller class offers."""

    def command(self, obs: Observation) -> Twist:
        """The command for this step."""


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
        distance = float(np.hypot(*(to_robot @ np.array(obs.goal[:2]))))
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
        error = float(wrap_angle(obs.goal[2] - obs.pose.xyt[2]))
        if abs(error) <= tolerance:
            return Twist()
        return Twist(wz=self._turn(error))
