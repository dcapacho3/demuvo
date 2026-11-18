"""The Dubins planner: the shortest curvature-bounded path between two poses.

The straight-line planner for a car. A robot that cannot turn in place (an
ackermann robot) cannot follow a path with a corner, and can only meet a goal's
angle by arriving along it. A Dubins path bends no tighter than `turning_radius`
and ends at the goal pose exactly, heading included, so such a robot can track
it and finishes aligned.

The path is the shortest of the six Dubins words, each an arc, a straight or an
arc, and an arc: LSL, RSR, LSR, RSL, RLR and LRL (L a left arc, R a right arc,
S straight), driven forward only. Like the straight-line planner it ignores the
costmap; `HybridAStar` plans a car's path round obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from robomath import Transform2D, wrap_angle

from ..costmap import Costmap
from ..plugins import check_positive

TURN = 2.0 * math.pi


def _mod2pi(angle: float) -> float:
    """The angle as a forward turn, in [0, 2*pi)."""
    return angle % TURN


Word = tuple[float, float, float, str]


def _words(alpha: float, beta: float, d: float) -> list[Word]:
    """Every Dubins word that exists for the normalised problem.

    The start is the origin heading `alpha` backwards, the goal is `d` turning
    radii along +x heading `beta`; see Shkel and Lumelsky. Each word is
    (t, p, q, segments): the lengths of its three segments in turning radii,
    an arc's length being its angle.
    """
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    found = []

    p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sa - sb)
    if p_sq >= 0.0:  # LSL
        tmp = math.atan2(cb - ca, d + sa - sb)
        found.append(
            (_mod2pi(tmp - alpha), math.sqrt(p_sq), _mod2pi(beta - tmp), "LSL")
        )
    p_sq = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sb - sa)
    if p_sq >= 0.0:  # RSR
        tmp = math.atan2(ca - cb, d - sa + sb)
        found.append(
            (_mod2pi(alpha - tmp), math.sqrt(p_sq), _mod2pi(tmp - beta), "RSR")
        )
    p_sq = d * d - 2.0 + 2.0 * c_ab + 2.0 * d * (sa + sb)
    if p_sq >= 0.0:  # LSR
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        found.append((_mod2pi(tmp - alpha), p, _mod2pi(tmp - beta), "LSR"))
    p_sq = d * d - 2.0 + 2.0 * c_ab - 2.0 * d * (sa + sb)
    if p_sq >= 0.0:  # RSL
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        found.append((_mod2pi(alpha - tmp), p, _mod2pi(beta - tmp), "RSL"))
    tmp = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1.0:  # RLR
        p = _mod2pi(TURN - math.acos(tmp))
        t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        found.append((t, p, _mod2pi(alpha - beta - t + p), "RLR"))
    tmp = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sb - sa)) / 8.0
    if abs(tmp) <= 1.0:  # LRL
        p = _mod2pi(TURN - math.acos(tmp))
        t = _mod2pi(-alpha + math.atan2(cb - ca, d + sa - sb) + p / 2.0)
        found.append((t, p, _mod2pi(beta - alpha - t + p), "LRL"))
    return found


@dataclass(frozen=True)
class Dubins:
    """The shortest path that bends no tighter than the turning radius.

    Attributes:
        turning_radius: the tightest bend of the path, in metres. Set it no
            lower than the robot's least turning radius (an ackermann robot's
            is wheelbase / tan(max_steer), 0.55 m with the model defaults), or
            the robot cannot track the path.
        spacing: distance between the points of the path, in metres.
    """

    turning_radius: float = 0.6
    spacing: float = 0.05

    # The navigation warns when a robot that cannot turn in place plans with a
    # planner that does not say this: its paths may have corners a car cannot cut.
    curvature_bounded: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Check the numbers."""
        check_positive(self, "turning_radius", "spacing")

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap and ignore it, as the straight-line planner does."""

    def plan(self, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """The shortest Dubins word, as poses `spacing` apart along it.

        The first pose is the start and the last is the goal, both exact. The
        same two poses always give the same path. A start on the goal gives
        the one-point path of the start, as the straight-line planner does.
        """
        (x0, y0, yaw0), (x1, y1, yaw1) = start.xyt, goal.xyt
        gap = math.hypot(x1 - x0, y1 - y0)
        if gap == 0.0 and abs(float(wrap_angle(yaw1 - yaw0))) < 1e-12:
            return np.array([[x0, y0, yaw0]])
        theta = math.atan2(y1 - y0, x1 - x0) if gap > 0.0 else yaw0
        alpha = _mod2pi(yaw0 - theta)
        beta = _mod2pi(yaw1 - theta)
        t, p, q, word = min(
            _words(alpha, beta, gap / self.turning_radius),
            key=lambda found: found[0] + found[1] + found[2],
        )
        return self._trace((x0, y0, yaw0), zip(word, (t, p, q)), (x1, y1, yaw1))

    def _trace(self, start, segments, goal) -> np.ndarray:
        """Poses along the word's segments, `spacing` apart, ending on the goal."""
        radius = self.turning_radius
        poses, (x, y, yaw) = [np.array(start)], start
        for kind, length in segments:
            if length * radius < 1e-12:
                continue
            steps = max(1, math.ceil(length * radius / self.spacing))
            at = np.linspace(length / steps, length, steps)
            if kind == "S":
                xs = x + np.cos(yaw) * at * radius
                ys = y + np.sin(yaw) * at * radius
                yaws = np.full_like(at, yaw)
            else:
                sign = 1.0 if kind == "L" else -1.0
                yaws = yaw + sign * at
                xs = x + sign * radius * (np.sin(yaws) - math.sin(yaw))
                ys = y - sign * radius * (np.cos(yaws) - math.cos(yaw))
            poses.append(np.column_stack([xs, ys, yaws]))
            x, y, yaw = float(xs[-1]), float(ys[-1]), float(yaws[-1])
        path = np.concatenate([np.atleast_2d(p) for p in poses])
        path[:, 2] = np.arctan2(np.sin(path[:, 2]), np.cos(path[:, 2]))
        path[-1] = goal  # exact, instead of within float rounding
        return path
