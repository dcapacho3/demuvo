"""Omni pursuit: move toward a point of the path ahead in any direction, turn apart.

A path follower for a robot that moves sideways, as a mecanum robot does
(`Observation.limits.strafes`). Such a robot's heading and its direction of
travel are independent, so it does not steer along an arc the way pure pursuit
does: it moves straight at the point ahead and turns to the angle it wants at
the same time.

Each run it:

1. finds the point ahead as pure pursuit does: the first path point at least
   `lookahead` from the robot, searching from the path point nearest to it;
2. aims a world velocity at it, at `speed`, slower near the goal by
   `approach_gain`;
3. turns toward the goal's angle (or, with `face_path`, the path's heading at the
   point ahead, until the goal is within `lookahead`), at `turn_gain` per radian
   of error and no faster than `rotate_speed` or the robot's own turn limit;
4. turns that world velocity into the robot's frame at the heading the robot will
   have halfway through the command, and divides it by sin(h) / h, h half the
   turn the command makes. The command is held for `Observation.dt`, along an
   arc; this puts the arc's chord exactly on the aim, where the heading at the
   start would drift half the turn off course every run;
5. cuts the command into the robot's limits with `Limits.clip`, which scales vx
   and vy by one factor, so the direction of travel is kept and what it commands
   is what the robot does.

Inside the goal's distance tolerance it stops moving and turns on the spot to the
goal's angle, then stops. It needs no turn before it sets off, and seldom one at
the end: it lines up on the way.

A robot that cannot move sideways cannot follow the aim, so it stays still, and
the log says so once for each path; drive one with `PurePursuit` or `DWB`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from robomath import wrap_angle

from ..models import Limits, Twist
from ..plugins import check_positive
from .base import Observation

LOGGER = logging.getLogger(__name__)


@dataclass
class OmniPursuit:
    """Move straight at a point of the path ahead, and turn to the angle apart.

    See the module note. Shortcut, as in `PurePursuit`: a path that crosses
    itself near the robot can fool the search for the point ahead.

    Attributes:
        speed: speed in m/s on the way, in whatever direction it moves.
        lookahead: distance in metres to the point it moves toward; a long one
            cuts corners and can touch a wall, so use less where walls are close.
        approach_gain: speed per metre left to the goal, in 1/s; it slows the
            robot near the goal.
        turn_gain: turn rate per radian of heading error, in 1/s.
        rotate_speed: largest turn rate in rad/s.
        face_path: face along the path on the way, as a robot whose sensors look
            ahead wants to, and turn to the goal's angle only once the goal is
            within `lookahead`; off, it turns to the goal's angle from the start.
    """

    speed: float = 0.5
    lookahead: float = 0.6
    approach_gain: float = 1.0
    turn_gain: float = 2.0
    rotate_speed: float = 1.0
    face_path: bool = False
    _refused_on: int = field(default=-1, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the numbers."""
        check_positive(
            self, "speed", "lookahead", "approach_gain", "turn_gain", "rotate_speed"
        )

    def command(self, obs: Observation) -> Twist:
        """Move toward the point ahead while turning; at the goal, turn and stop.

        With an empty path there is nothing to follow, so the robot stays still.

        Raises:
            ValueError: if there is a path but no goal.
        """
        if len(obs.path) == 0:
            return Twist()
        if obs.goal is None:
            raise ValueError("OmniPursuit needs a goal to follow a path")
        limits = obs.limits
        if limits is not None and not limits.strafes:
            if self._refused_on != obs.path_changes:
                LOGGER.warning(
                    "OmniPursuit: the robot cannot move sideways, so it stays "
                    "still; drive it with PurePursuit or DWB"
                )
                self._refused_on = obs.path_changes
            return Twist()
        x, y, yaw = obs.pose.xyt
        gx, gy, gyaw = obs.goal.xyt
        distance = math.hypot(gx - x, gy - y)
        if distance <= obs.goal_tolerance[0]:
            error = float(wrap_angle(gyaw - yaw))
            if abs(error) <= obs.goal_tolerance[1]:
                return Twist()
            return self._fit(Twist(wz=self._turn(error, limits)), limits)
        index = self._target(obs.path, x, y)
        heading = gyaw
        if self.face_path and distance > self.lookahead:
            heading = float(obs.path[index, 2])
        wz = self._turn(float(wrap_angle(heading - yaw)), limits)
        dx, dy = (float(v) for v in obs.path[index, :2] - (x, y))
        gap = math.hypot(dx, dy)
        if gap == 0.0:
            # On the path's last point but short of the goal: no way to aim, so
            # only turn, instead of dividing by zero.
            return self._fit(Twist(wz=wz), limits)
        scale = min(self.speed, self.approach_gain * distance) / gap
        vx, vy = body_velocity(dx * scale, dy * scale, yaw, wz, obs.dt)
        return self._fit(Twist(vx=vx, vy=vy, wz=wz), limits)

    def _target(self, path: np.ndarray, x: float, y: float) -> int:
        """The index of the point ahead: the first at least `lookahead` away.

        The search starts at the path point nearest the robot; if no point is so
        far, it is the path's last point.
        """
        distance = np.hypot(path[:, 0] - x, path[:, 1] - y)
        nearest = int(np.argmin(distance))
        ahead = np.flatnonzero(distance[nearest:] >= self.lookahead)
        return nearest + int(ahead[0]) if ahead.size else len(path) - 1

    def _turn(self, error: float, limits: Limits | None) -> float:
        """The turn rate for a heading error, inside `rotate_speed` and the robot's."""
        bound = self.rotate_speed
        if limits is not None:
            bound = min(bound, limits.max_w(0.0))
        return max(-bound, min(bound, self.turn_gain * error))

    @staticmethod
    def _fit(command: Twist, limits: Limits | None) -> Twist:
        """The command inside the robot's limits, its direction of travel kept."""
        return command if limits is None else limits.clip(command)


def body_velocity(
    wx: float, wy: float, yaw: float, wz: float, dt: float
) -> tuple[float, float]:
    """The body (vx, vy) that moves the robot at world velocity (wx, wy) over dt.

    Held for dt with turn rate wz, a body velocity sweeps an arc, and the arc's
    chord points along the body velocity turned by the heading at mid-step,
    shortened by s = sin(h) / h with h = wz * dt / 2. So the world velocity turned
    into the frame at the mid-step heading and divided by s moves the robot by
    exactly (wx, wy) * dt. With wz = 0 it is the plain
    turn into the robot's frame.
    """
    half = wz * dt / 2
    s = math.sin(half) / half if half else 1.0
    c, n = math.cos(yaw + half), math.sin(yaw + half)
    return (c * wx + n * wy) / s, (c * wy - n * wx) / s
