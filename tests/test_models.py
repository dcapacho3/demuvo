"""Tests for the kinematic models."""

import math

import pytest

from demuvo import MODELS, DiffDrive

TURN = 2 * math.pi


def test_the_table_names_every_model():
    """The world file finds a model by its name."""
    assert MODELS["diff_drive"] is DiffDrive


def test_a_straight_line_follows_the_heading():
    """Heading +y and w = 0: the robot moves along +y."""
    x, y, angle = DiffDrive().step((1.0, 1.0, math.pi / 2), (1.0, 0.0), 2.0)
    assert (x, y, angle) == pytest.approx((1.0, 3.0, math.pi / 2))


def test_a_spin_in_place_keeps_the_position():
    """V = 0: the position stays and the angle grows by w * dt."""
    x, y, angle = DiffDrive().step((3.0, 2.0, 0.0), (0.0, 1.0), 0.5)
    assert (x, y, angle) == pytest.approx((3.0, 2.0, 0.5))


def test_the_angle_is_wrapped():
    """A turn past pi comes back in (-pi, pi]."""
    _, _, angle = DiffDrive().step((0.0, 0.0, 3.0), (0.0, 1.0), 0.5)
    assert angle == pytest.approx(3.5 - TURN)


def test_a_half_circle_has_radius_v_over_w():
    """V = 1, w = 0.5: radius 2. One exact step of half a turn ends at (0, 4)."""
    x, y, angle = DiffDrive().step((0.0, 0.0, 0.0), (1.0, 0.5), TURN)
    assert (x, y) == pytest.approx((0.0, 4.0))
    assert abs(angle) == pytest.approx(math.pi)


def test_one_big_step_equals_many_small_ones():
    """The step is exact, so the step length does not change the result."""
    model = DiffDrive()
    pose = (1.0, 2.0, 0.3)
    for _ in range(100):
        pose = model.step(pose, (0.7, -0.4), 0.02)
    assert pose == pytest.approx(model.step((1.0, 2.0, 0.3), (0.7, -0.4), 2.0))


def test_the_default_limits():
    """Without parameters a robot reaches 1 m/s and 1 rad/s."""
    assert DiffDrive().clip((5.0, -5.0)) == (1.0, -1.0)


def test_a_command_is_clipped_to_the_limits():
    """Speeds above the limits are cut, in both directions."""
    model = DiffDrive(max_v=1.0, max_w=0.5)
    assert model.clip((5.0, -5.0)) == (1.0, -0.5)
    assert model.clip((-5.0, 5.0)) == (-1.0, 0.5)
    assert model.clip((0.2, 0.1)) == (0.2, 0.1)
    assert model.step((0.0, 0.0, 0.0), (5.0, 0.0), 1.0) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
