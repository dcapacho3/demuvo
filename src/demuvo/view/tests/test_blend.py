"""Tests for the pose between two steps; nothing here draws."""

import pytest
from robomath import Transform2D

from demuvo import Twist
from demuvo.models import DiffDrive
from demuvo.view.blend import blend


def test_fraction_zero_is_the_old_pose_and_one_is_the_new():
    """The ends of the blend are the two poses."""
    old = Transform2D.from_xyt((1.0, 2.0, 0.3))
    new = Transform2D.from_xyt((2.0, 2.5, 0.9))
    assert blend(old, new, 0.0).isclose(old, atol=1e-12)
    assert blend(old, new, 1.0).isclose(new, atol=1e-12)


def test_half_a_step_is_the_exact_half_step_of_the_same_command():
    """On a turn a plain average of x, y and angle is wrong; this is exact."""
    model = DiffDrive()
    start = Transform2D.from_xyt((1.0, 1.0, 0.4))
    command = Twist(vx=0.5, wz=1.0)
    full = model.step(start, command, 0.1)
    half = model.step(start, command, 0.05)
    assert blend(start, full, 0.5).isclose(half, atol=1e-12)


def test_the_blend_takes_the_short_way_across_the_seam():
    """From 3.1 rad to -3.1 rad the middle is near pi, not 0."""
    old = Transform2D.from_xyt((0.0, 0.0, 3.1))
    new = Transform2D.from_xyt((0.0, 0.0, -3.1))
    assert abs(blend(old, new, 0.5).xyt[2]) == pytest.approx(3.14159, abs=1e-4)
