"""Tests for the kinematic models."""

import math

import pytest
from robomath import Transform2D

from demuvo import MODELS, DiffDrive, Twist

TURN = 2 * math.pi


def drive(pose, command, dt, model=None):
    """The (x, y, angle) after one step from an (x, y, angle) pose."""
    model = model or DiffDrive()
    return tuple(model.step(Transform2D.from_xyt(pose), command, dt).xyt)


def test_the_table_names_every_model():
    """The world file finds a model by its name."""
    assert MODELS["diff_drive"] is DiffDrive


def test_a_straight_line_follows_the_heading():
    """Heading +y and wz = 0: the robot moves along +y."""
    x, y, angle = drive((1.0, 1.0, math.pi / 2), Twist(vx=1.0), 2.0)
    assert (x, y, angle) == pytest.approx((1.0, 3.0, math.pi / 2))


def test_a_spin_in_place_keeps_the_position():
    """Vx = 0: the position stays and the angle grows by wz * dt."""
    x, y, angle = drive((3.0, 2.0, 0.0), Twist(wz=1.0), 0.5)
    assert (x, y, angle) == pytest.approx((3.0, 2.0, 0.5))


def test_the_angle_is_wrapped():
    """A turn past pi comes back in (-pi, pi]."""
    _, _, angle = drive((0.0, 0.0, 3.0), Twist(wz=1.0), 0.5)
    assert angle == pytest.approx(3.5 - TURN)


def test_a_half_circle_has_radius_v_over_w():
    """Vx = 1, wz = 0.5: radius 2. One exact step of half a turn ends at (0, 4)."""
    x, y, angle = drive((0.0, 0.0, 0.0), Twist(vx=1.0, wz=0.5), TURN)
    assert (x, y) == pytest.approx((0.0, 4.0))
    assert abs(angle) == pytest.approx(math.pi)


def test_one_big_step_equals_many_small_ones():
    """The step is exact, so the step length does not change the result."""
    command = Twist(vx=0.7, wz=-0.4)
    pose = Transform2D.from_xyt((1.0, 2.0, 0.3))
    for _ in range(100):
        pose = DiffDrive().step(pose, command, 0.02)
    assert tuple(pose.xyt) == pytest.approx(drive((1.0, 2.0, 0.3), command, 2.0))


def test_the_default_limits():
    """Without parameters a robot reaches 1 m/s and 1 rad/s."""
    assert DiffDrive().clip(Twist(vx=5.0, wz=-5.0)) == Twist(vx=1.0, wz=-1.0)


def test_a_command_is_clipped_to_the_limits():
    """Speeds above the limits are cut, in both directions."""
    model = DiffDrive(max_v=1.0, max_w=0.5)
    assert model.clip(Twist(vx=5.0, wz=-5.0)) == Twist(vx=1.0, wz=-0.5)
    assert model.clip(Twist(vx=-5.0, wz=5.0)) == Twist(vx=-1.0, wz=0.5)
    assert model.clip(Twist(vx=0.2, wz=0.1)) == Twist(vx=0.2, wz=0.1)
    assert drive((0.0, 0.0, 0.0), Twist(vx=5.0), 1.0, model) == pytest.approx(
        (1.0, 0.0, 0.0)
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", ["vx", "vy", "vz", "wx", "wy", "wz"])
def test_a_twist_that_is_not_finite_is_refused(bad, name):
    """Nan or infinity in any field raises, naming the field."""
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        Twist(**{name: bad})


@pytest.mark.parametrize("name", ["vy", "vz", "wx", "wy"])
def test_what_the_robot_cannot_follow_is_ignored_with_a_warning(name):
    """Diff drive moves as if the field were 0, and one warning names it."""
    wanted = drive((0.0, 0.0, 0.0), Twist(vx=1.0), 1.0)
    with pytest.warns(UserWarning, match=name):
        pose = drive((0.0, 0.0, 0.0), Twist(vx=1.0, **{name: 2.0}), 1.0)
    assert pose == pytest.approx(wanted)


def test_a_command_the_robot_can_follow_gives_no_warning(recwarn):
    """The warning is only for a nonzero field the model does not use."""
    drive((0.0, 0.0, 0.0), Twist(vx=1.0, wz=0.5), 1.0)
    assert not recwarn
