"""The Reeds-Shepp planner: Dubins with reverse, for a car that can back up.

A Dubins path drives forward only, so a goal just behind the robot costs a
whole loop. A Reeds-Shepp path may change direction at a cusp: it is the
shortest path between two poses for a car that bends no tighter than
`turning_radius` and drives either way, so a goal behind is reached by backing
up, as a driver would. The path ends at the goal pose exactly, heading included.

The 48 words of Reeds and Shepp fall into 18 segment patterns of arcs (L, R)
and straights (S), each segment driven forward or backward; the formulas are
those of the paper as corrected in OMPL. The heading of each pose is the car's,
whichever way it moves, so a controller that follows the path backwards keeps
the heading continuous.

A robot follows such a path only with a controller that reverses:
`PurePursuit` or `DWB`, which follow it run by run. Like the straight-line and
Dubins planners it ignores the costmap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from robomath import Transform2D, wrap_angle

from ..costmap import Costmap
from ..paths import trace_segments
from ..plugins import check_positive

PI = math.pi
HALF_PI = math.pi / 2
# A length this small is zero: it keeps a test of sign from failing on rounding.
ZERO = 10 * float(np.spacing(1.0))

# The segment patterns, by the number OMPL gives them.
PATTERNS = (
    "LRL",
    "RLR",
    "LRLR",
    "RLRL",
    "LRSL",
    "RLSR",
    "LSRL",
    "RSLR",
    "LRSR",
    "RLSL",
    "RSRL",
    "LSLR",
    "LSR",
    "RSL",
    "LSL",
    "RSR",
    "LRSLR",
    "RLSRL",
)

# A word: its pattern and the signed length of each segment, in turning radii.
Word = tuple[str, tuple[float, ...]]


def _polar(x: float, y: float) -> tuple[float, float]:
    """(radius, angle) of a point."""
    return math.hypot(x, y), math.atan2(y, x)


def _tau_omega(u, v, xi, eta, phi) -> tuple[float, float]:
    """The two outer arcs of a CCCC word; see the paper, formulas 8.7 and 8.8."""
    delta = wrap_angle(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = wrap_angle(t1 + PI) if t2 < 0 else wrap_angle(t1)
    return tau, wrap_angle(tau - u + v - phi)


# Each solver takes the goal (x, y, phi) in the start's frame, in turning radii,
# and gives (t, u, v) for its base word, or None if the word does not exist.


def _lp_sp_lp(x, y, phi):
    """Formula 8.1: L+ S+ L+."""
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= -ZERO:
        v = wrap_angle(phi - t)
        if v >= -ZERO:
            return t, u, v
    return None


def _lp_sp_rp(x, y, phi):
    """Formula 8.2: L+ S+ R+."""
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 *= u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        t = wrap_angle(t1 + math.atan2(2.0, u))
        v = wrap_angle(t - phi)
        if t >= -ZERO and v >= -ZERO:
            return t, u, v
    return None


def _lp_rm_l(x, y, phi):
    """Formulas 8.3 and 8.4: L+ R- L."""
    xi, eta = x - math.sin(phi), y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = wrap_angle(theta + 0.5 * u + PI)
        v = wrap_angle(phi - t + u)
        if t >= -ZERO and u <= ZERO:
            return t, u, v
    return None


def _lp_rup_lum_rm(x, y, phi):
    """Formula 8.7: L+ R+ L- R-."""
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -ZERO and v <= ZERO:
            return t, u, v
    return None


def _lp_rum_lum_rp(x, y, phi):
    """Formula 8.8: L+ R- L- R+."""
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -HALF_PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _lp_rm_sm_lm(x, y, phi):
    """Formula 8.9: L+ R- S- L-."""
    xi, eta = x - math.sin(phi), y - 1.0 + math.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = wrap_angle(theta + math.atan2(r, -2.0))
        v = wrap_angle(phi - HALF_PI - t)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _lp_rm_sm_rm(x, y, phi):
    """Formula 8.10: L+ R- S- R-."""
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = wrap_angle(t + HALF_PI - phi)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _lp_rm_s_lm_rp(x, y, phi):
    """Formula 8.11: L+ R- S- L- R+."""
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= ZERO:
            t = wrap_angle(
                math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta)
            )
            v = wrap_angle(t - phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _words(x: float, y: float, phi: float) -> list[Word]:
    """Every Reeds-Shepp word that exists for the normalised problem.

    Each base word is tried four ways: as is, time-flipped (driven backwards,
    which negates x, phi and every length), reflected (mirrored, which negates
    y, phi and swaps L and R) and both; the CCC and CCSC words also backwards
    (the segments in the other order). That is how 48 words come from 9 formulas.
    """
    found: list[Word] = []
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    def add(solver, shape, pattern, args, *, flip, reverse):
        """Try a solver on transformed inputs and record the word it gives."""
        got = solver(*args)
        if got is None:
            return
        lengths = shape(*got)
        if reverse:
            lengths = tuple(reversed(lengths))
        if flip:
            lengths = tuple(-length for length in lengths)
        found.append((pattern, lengths))

    # (solver, segment lengths from (t, u, v), pattern index, also backwards)
    base = (
        (_lp_sp_lp, lambda t, u, v: (t, u, v), 14, False),
        (_lp_sp_rp, lambda t, u, v: (t, u, v), 12, False),
        (_lp_rm_l, lambda t, u, v: (t, u, v), 0, True),
        (_lp_rup_lum_rm, lambda t, u, v: (t, u, -u, v), 2, False),
        (_lp_rum_lum_rp, lambda t, u, v: (t, u, u, v), 2, False),
        (_lp_rm_sm_lm, lambda t, u, v: (t, -HALF_PI, u, v), 4, True),
        (_lp_rm_sm_rm, lambda t, u, v: (t, -HALF_PI, u, v), 8, True),
        (_lp_rm_s_lm_rp, lambda t, u, v: (t, -HALF_PI, u, -HALF_PI, v), 16, False),
    )
    for solver, shape, pattern, backwards in base:
        for reverse in (False, True) if backwards else (False,):
            point = (xb, yb) if reverse else (x, y)
            # The pattern index of the backwards word: the reversed pattern.
            index = pattern
            if reverse:
                index = {0: 0, 4: 6, 8: 10}[pattern]
            for flip in (False, True):
                for mirror in (0, 1):
                    args = (
                        -point[0] if flip else point[0],
                        -point[1] if mirror else point[1],
                        -phi if flip != bool(mirror) else phi,
                    )
                    add(
                        solver,
                        shape,
                        PATTERNS[index + mirror],
                        args,
                        flip=flip,
                        reverse=reverse,
                    )
    return found


def _length(word: Word) -> float:
    """The length of a word in turning radii."""
    return sum(abs(length) for length in word[1])


@dataclass(frozen=True)
class ReedsShepp:
    """The shortest path for a car that may back up.

    Attributes:
        turning_radius: the tightest bend of the path, in metres. Set it no
            lower than the robot's least turning radius (an ackermann robot's
            is wheelbase / tan(max_steer), 0.55 m with the model defaults), or
            the robot cannot track the path.
        spacing: distance between the points of the path, in metres.
    """

    turning_radius: float = 0.6
    spacing: float = 0.05

    curvature_bounded: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Check the numbers."""
        check_positive(self, "turning_radius", "spacing")

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap and ignore it, as the straight-line planner does."""

    def plan(self, start: Transform2D, goal: Transform2D) -> np.ndarray:
        """The shortest Reeds-Shepp word, as poses `spacing` apart along it.

        The first pose is the start and the last is the goal, both exact. A
        start on the goal gives the one-point path of the start.
        """
        word = self.word(start, goal)
        if word is None:
            return np.array([start.xyt])
        return self._trace(start.xyt, word, goal.xyt)

    def word(self, start: Transform2D, goal: Transform2D) -> Word | None:
        """The shortest word from start to goal, or None if they are one pose."""
        (x0, y0, yaw0), (x1, y1, yaw1) = start.xyt, goal.xyt
        dx, dy = x1 - x0, y1 - y0
        phi = float(wrap_angle(yaw1 - yaw0))
        if dx == 0.0 and dy == 0.0 and abs(phi) < 1e-12:
            return None
        c, s = math.cos(yaw0), math.sin(yaw0)
        x = (c * dx + s * dy) / self.turning_radius
        y = (-s * dx + c * dy) / self.turning_radius
        return min(_words(x, y, phi), key=_length)

    def _trace(self, start, word: Word, goal) -> np.ndarray:
        """Poses along the word's segments, `spacing` apart, ending on the goal."""
        return trace_segments(
            start, zip(*word), goal, self.turning_radius, self.spacing
        )
