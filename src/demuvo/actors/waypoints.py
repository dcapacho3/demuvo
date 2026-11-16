"""The waypoints motion: along a line of points at a speed, once, round or back."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from robomath import Transform2D

from ..plugins import check_not_negative, check_positive

# How the actor goes on after the last point.
MODES = ("once", "loop", "bounce")


@dataclass(frozen=True)
class Waypoints:
    """Straight from the start pose through the points, in order, at one speed.

    The actor leaves where its `pose` puts it, so `points` are the places it
    goes to after that: one point and `mode = "bounce"` is an actor going back
    and forth between two places. It faces the way it moves, and when it stops
    it keeps facing the way it came.

    Attributes:
        points: the places it goes to after the start, in order, each [x, y],
            [x, y, wait] or [x, y, wait, heading]. At least one, and no two in a
            row the same. A point's own `wait` replaces the setting `wait` there,
            and its `heading` (radians) is the way the actor faces while it
            stands there; without one it faces the way it came.
        speed: metres per second along every leg.
        mode: what it does after the last point. "once" stays there; "loop" goes
            straight back to the start and does it all again; "bounce" turns
            round and goes back through the points to the start, and again.
        wait: seconds it stands at every point that has no wait of its own, the
            start included when it comes back to it. 0 passes straight through.
        delay: seconds it stands at the start before it first moves.
        face: True to face the way it moves; False to keep the start angle.

    Attributes of the class:
        CHOICES: the settings that are one of a few words, for a form to show as
            a drop-down; see `demuvo.options.choices_of`.
    """

    CHOICES: ClassVar[dict[str, tuple[str, ...]]] = {"mode": MODES}

    points: list = field(default_factory=list)
    speed: float = 0.5
    mode: str = "loop"
    wait: float = 0.0
    delay: float = 0.0
    face: bool = True

    def __post_init__(self) -> None:
        """Check the settings; the legs are built in `configure`."""
        check_positive(self, "speed")
        check_not_negative(self, "wait", "delay")
        if self.mode not in MODES:
            raise ValueError(
                f"mode must be one of {', '.join(MODES)}, got {self.mode!r}"
            )
        if not isinstance(self.face, bool):
            raise ValueError(f"face must be true or false, got {self.face!r}")
        if not isinstance(self.points, list) or not self.points:
            raise ValueError("points needs at least one [x, y]")
        for point in self.points:
            if (
                not isinstance(point, list | tuple)
                or not 2 <= len(point) <= 4
                or not all(
                    isinstance(v, int | float)
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                    for v in point
                )
            ):
                raise ValueError(
                    "points: each must be [x, y], [x, y, wait] or "
                    f"[x, y, wait, heading], got {point!r}"
                )
            if len(point) > 2 and point[2] < 0:
                raise ValueError(f"points: a wait must be 0 or more, got {point!r}")

    def configure(self, start: Transform2D, rng: np.random.Generator) -> None:
        """Lay out the legs from the start pose; this motion draws no random number."""
        x, y, angle = (float(v) for v in start.xyt)
        # Each stop is its place, how long it holds the actor and which way it
        # faces there (None for the way it came); the start is the first.
        stops = [((x, y), self.wait, None)]
        for point in self.points:
            wait = float(point[2]) if len(point) > 2 else self.wait
            heading = float(point[3]) if len(point) > 3 else None
            stops.append(((float(point[0]), float(point[1])), wait, heading))
        if self.mode == "loop":
            stops.append(stops[0])
        elif self.mode == "bounce":
            stops += stops[-2::-1]
        for (a, _, _), (b, _, _) in zip(stops, stops[1:]):
            if a == b:
                raise ValueError(f"points: two in a row are the same place, {list(a)}")
        # One lap: from leaving the start to standing again where the lap ends.
        # The first entry of `times` is when each leg starts, the stops follow.
        times, legs = [], []
        clock = 0.0
        for (a, _, _), (b, wait, heading) in zip(stops, stops[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            times.append(clock)
            legs.append((a, b, length / self.speed, heading))
            clock += length / self.speed + wait
        if self.mode == "once":
            clock -= stops[-1][1]  # the last stop is for good, not a wait
        object.__setattr__(self, "_angle", angle)
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_legs", legs)
        object.__setattr__(self, "_lap", clock)

    def pose_at(self, time: float) -> Transform2D:
        """The pose at this time; before 0 and before `delay` it is the start pose."""
        legs = getattr(self, "_legs", None)
        if legs is None:
            raise RuntimeError("configure must be called before pose_at")
        t = max(0.0, time - self.delay)
        if self.mode == "once":
            t = min(t, self._lap)
        else:
            t %= self._lap
        k = bisect_right(self._times, t) - 1
        (ax, ay), (bx, by), duration, turn = legs[k]
        if t == 0.0 and time - self.delay <= 0.0:
            # Still waiting to leave: the start pose, as the file gave it.
            return Transform2D.from_xyt(np.array([ax, ay, self._angle]))
        heading = math.atan2(by - ay, bx - ax) if self.face else self._angle
        s = min(1.0, (t - self._times[k]) / duration)
        if turn is not None and s == 1.0:
            heading = turn
        return Transform2D.from_xyt(
            np.array([ax + (bx - ax) * s, ay + (by - ay) * s, heading])
        )
