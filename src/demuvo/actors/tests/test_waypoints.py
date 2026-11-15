"""Tests for the waypoints motion."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo.actors import Motion, Waypoints

START = Transform2D.from_xyt(np.array([1.0, 1.0, 0.5]))
RNG = np.random.default_rng(0)


def motion(**params) -> Waypoints:
    """A configured motion from the start pose, with these settings."""
    m = Waypoints(**params)
    m.configure(START, RNG)
    return m


def xyt(m: Waypoints, t: float) -> tuple[float, float, float]:
    """The (x, y, angle) at a time, as plain floats."""
    return tuple(float(v) for v in m.pose_at(t).xyt)


def test_it_is_a_motion():
    """The class has the two methods the protocol asks for."""
    m = Waypoints(points=[[2.0, 1.0]])
    assert all(callable(getattr(m, name)) for name in ("configure", "pose_at"))
    assert isinstance(Motion, type)


def test_it_leaves_the_start_at_the_speed_facing_the_way_it_goes():
    """Two metres along x at 0.5 m/s: at 2 s it is halfway, heading 0."""
    m = motion(points=[[3.0, 1.0]], speed=0.5, mode="once")
    assert xyt(m, 0.0) == pytest.approx((1.0, 1.0, 0.5))
    assert xyt(m, 2.0) == pytest.approx((2.0, 1.0, 0.0))
    assert xyt(m, 4.0) == pytest.approx((3.0, 1.0, 0.0))


def test_once_stays_at_the_last_point_for_good():
    """After the last point the pose no longer changes."""
    m = motion(points=[[3.0, 1.0]], speed=0.5, mode="once", wait=1.0)
    assert xyt(m, 4.0) == xyt(m, 100.0) == pytest.approx((3.0, 1.0, 0.0))


def test_loop_goes_back_to_the_start_and_repeats():
    """Start, one point 2 m away and back is a 4 m lap: 8 s at 0.5 m/s."""
    m = motion(points=[[3.0, 1.0]], speed=0.5, mode="loop")
    assert xyt(m, 6.0) == pytest.approx((2.0, 1.0, math.pi))  # on the way back
    assert xyt(m, 8.0)[:2] == pytest.approx((1.0, 1.0))
    assert xyt(m, 10.0) == pytest.approx(xyt(m, 2.0))


def test_bounce_between_two_points_retraces_the_leg():
    """One point and bounce is back and forth: the same lap as a two-point loop."""
    m = motion(points=[[3.0, 1.0]], speed=0.5, mode="bounce")
    assert xyt(m, 2.0) == pytest.approx((2.0, 1.0, 0.0))
    assert xyt(m, 6.0) == pytest.approx((2.0, 1.0, math.pi))
    assert xyt(m, 10.0) == pytest.approx(xyt(m, 2.0))


def test_bounce_along_a_path_turns_round_at_the_end():
    """Three stops: out through the middle, back through it, and again.

    With no wait, the instant it reaches a corner it already faces the next leg.
    """
    m = motion(points=[[2.0, 1.0], [2.0, 2.0]], speed=1.0, mode="bounce")
    assert xyt(m, 1.0) == pytest.approx((2.0, 1.0, math.pi / 2))  # at the middle
    assert xyt(m, 2.0) == pytest.approx((2.0, 2.0, -math.pi / 2))  # turning at the end
    assert xyt(m, 2.5) == pytest.approx((2.0, 1.5, -math.pi / 2))  # on the way back
    assert xyt(m, 4.0)[:2] == pytest.approx((1.0, 1.0))  # at the start
    assert xyt(m, 5.0) == pytest.approx(xyt(m, 1.0))


def test_wait_holds_every_point_it_arrives_at():
    """1 s legs, 2 s waits: at the point from 1 s to 3 s, facing where it came from."""
    m = motion(points=[[2.0, 1.0]], speed=1.0, mode="loop", wait=2.0)
    assert xyt(m, 1.0) == xyt(m, 2.5) == pytest.approx((2.0, 1.0, 0.0))
    assert xyt(m, 3.5) == pytest.approx((1.5, 1.0, math.pi))
    # back at the start at 4 s, held to 6 s, then round again
    assert xyt(m, 5.0) == pytest.approx((1.0, 1.0, math.pi))
    assert xyt(m, 6.5) == pytest.approx((1.5, 1.0, 0.0))


def test_delay_keeps_the_start_pose_then_the_motion_is_the_same_shifted():
    """The actor stands as the file placed it, angle included, until the delay ends."""
    held = motion(points=[[3.0, 1.0]], speed=0.5, mode="loop", delay=3.0)
    free = motion(points=[[3.0, 1.0]], speed=0.5, mode="loop")
    assert xyt(held, 0.0) == xyt(held, 2.9) == pytest.approx((1.0, 1.0, 0.5))
    assert xyt(held, 5.0) == pytest.approx(xyt(free, 2.0))


def test_face_false_keeps_the_start_angle():
    """The body moves without turning."""
    m = motion(points=[[3.0, 1.0]], speed=0.5, face=False)
    assert xyt(m, 2.0) == pytest.approx((2.0, 1.0, 0.5))
    assert xyt(m, 6.0)[2] == pytest.approx(0.5)


def test_a_time_before_zero_is_the_start():
    """Nothing happens before the run begins."""
    m = motion(points=[[3.0, 1.0]])
    assert xyt(m, -1.0) == pytest.approx((1.0, 1.0, 0.5))


def test_the_pose_depends_on_the_time_alone():
    """Asking out of order, or twice, gives the same bits."""
    m = motion(points=[[2.0, 3.0], [4.0, 0.5]], speed=0.7, mode="bounce", wait=0.3)
    first = [m.pose_at(t).xyt.tolist() for t in np.linspace(0, 20, 101)]
    m.pose_at(1234.5)
    second = [m.pose_at(t).xyt.tolist() for t in np.linspace(0, 20, 101)]
    assert first == second


def test_pose_at_before_configure_is_refused():
    """A motion knows nothing of where to go until it is configured."""
    with pytest.raises(RuntimeError, match="configure"):
        Waypoints(points=[[2.0, 1.0]]).pose_at(0.0)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "points needs at least one"),
        ({"points": [[1.0]]}, r"each must be \[x, y\]"),
        ({"points": [[1.0, "a"]]}, r"each must be \[x, y\]"),
        ({"points": [[1.0, 2.0]], "speed": 0}, "speed must be a positive"),
        ({"points": [[1.0, 2.0]], "wait": -1}, "wait must be 0 or more"),
        ({"points": [[1.0, 2.0]], "delay": -1}, "delay must be 0 or more"),
        ({"points": [[1.0, 2.0]], "mode": "ping"}, "mode must be one of once, loop"),
        ({"points": [[1.0, 2.0]], "face": 1}, "face must be true or false"),
    ],
)
def test_a_wrong_setting_is_refused_when_built(params, message):
    """The check is at build time, which is when a world file is loaded."""
    with pytest.raises(ValueError, match=message):
        Waypoints(**params)


def test_two_points_in_a_row_at_the_same_place_are_refused():
    """A leg of no length has no direction to face and no time to take."""
    with pytest.raises(ValueError, match="same place"):
        motion(points=[[1.0, 1.0]])
    with pytest.raises(ValueError, match="same place"):
        motion(points=[[2.0, 1.0], [2.0, 1.0]])
