"""Where to draw a robot between two steps, without pygame."""

from __future__ import annotations

from robomath import Transform2D


def blend(previous: Transform2D, current: Transform2D, fraction: float) -> Transform2D:
    """The pose `fraction` of the way from `previous` to `current`.

    It follows the arc the robot drove, so on a turn it equals a shorter step of
    the same command, and it takes the short way across the angle jump at pi.
    With 0 it is `previous` and with 1 it is `current`.
    """
    return previous.interp(current, fraction)
