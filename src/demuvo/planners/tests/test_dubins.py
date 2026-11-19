"""Tests for the Dubins planner."""

import math

import numpy as np
import pytest
from robomath import Transform2D, wrap_angle

from demuvo import Ackermann, options
from demuvo.planners import Dubins

from .bends import max_curvature


def plan(start, goal, **params):
    """The Dubins path between two (x, y, angle) poses."""
    return Dubins(**params).plan(
        Transform2D.from_xyt(start), Transform2D.from_xyt(goal)
    )


def test_the_path_starts_and_ends_on_the_poses():
    """First row is the start, last row is the goal, angle included, exactly."""
    path = plan((1.0, 1.0, 0.0), (3.0, 4.0, 2.0))
    assert path[0] == pytest.approx([1.0, 1.0, 0.0])
    assert path[-1] == pytest.approx([3.0, 4.0, 2.0])


def test_a_goal_straight_ahead_is_the_straight_line():
    """Aligned poses on one line: the path is that line, no longer."""
    path = plan((1.0, 1.0, 0.0), (4.0, 1.0, 0.0))
    steps = np.diff(path[:, :2], axis=0)
    assert np.hypot(steps[:, 0], steps[:, 1]).sum() == pytest.approx(3.0)
    assert path[:, 1] == pytest.approx(np.full(len(path), 1.0))
    assert path[:, 2] == pytest.approx(np.zeros(len(path)))


def test_a_start_on_the_goal_is_one_point():
    """Nothing to drive: the path is the start alone."""
    path = plan((2.0, 2.0, 0.5), (2.0, 2.0, 0.5))
    assert path.shape == (1, 3)
    assert path[0] == pytest.approx([2.0, 2.0, 0.5])


@pytest.mark.parametrize("seed", range(20))
def test_the_path_never_bends_tighter_than_the_turning_radius(seed):
    """Heading change per metre stays within 1 / turning_radius, pose to pose."""
    rng = np.random.default_rng(seed)
    start = rng.uniform([-4, -4, -math.pi], [4, 4, math.pi])
    goal = rng.uniform([-4, -4, -math.pi], [4, 4, math.pi])
    path = plan(tuple(start), tuple(goal), turning_radius=0.8, spacing=0.02)
    assert max_curvature(path, 0.02) <= 1 / 0.8 + 0.05
    assert path[-1] == pytest.approx(goal)


def test_a_goal_behind_is_reached_the_long_way_round():
    """A U-turn cannot be shorter than half a turning circle."""
    path = plan((0.0, 0.0, 0.0), (-2.0, 0.0, math.pi), turning_radius=1.0)
    steps = np.diff(path[:, :2], axis=0)
    assert np.hypot(steps[:, 0], steps[:, 1]).sum() >= math.pi


def test_two_poses_nose_to_nose_take_three_arcs():
    """Closer than the radius allows, the path loops out and back (a CCC word)."""
    path = plan((0.0, 0.0, 0.0), (0.3, 0.0, math.pi), turning_radius=1.0)
    steps = np.diff(path[:, :2], axis=0)
    assert np.hypot(steps[:, 0], steps[:, 1]).sum() > 2.0
    assert path[-1] == pytest.approx([0.3, 0.0, math.pi])


def test_an_ackermann_robot_can_hold_every_bend_of_the_path():
    """A radius at the model's least turning circle keeps each bend steerable."""
    model = Ackermann()  # least radius wheelbase / tan(max_steer)
    radius = model.wheelbase / math.tan(model.max_steer)
    path = plan((0.0, 0.0, 0.0), (1.0, 2.5, 2.5), turning_radius=radius, spacing=0.01)
    steps = np.diff(path[:, :2], axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    turned = np.array([float(wrap_angle(d)) for d in np.diff(path[:, 2])])
    cone = model.limits()
    for length, turn in zip(lengths[lengths > 1e-12], turned[lengths > 1e-12]):
        vx = length  # over one second, so wz is the turn
        # A chord is a touch shorter than its arc, so allow that much slack.
        assert abs(turn) <= cone.max_w(vx) * (1 + 1e-3) + 1e-9


@pytest.mark.parametrize("name", ["turning_radius", "spacing"])
@pytest.mark.parametrize("bad", [0.0, -0.5, math.nan])
def test_a_bad_setting_is_refused_by_name(name, bad):
    """Turning radius and spacing must be positive numbers."""
    with pytest.raises(ValueError, match=name):
        Dubins(**{name: bad})


def test_the_options_list_dubins():
    """The inspector offers the planner by its friendly name."""
    assert options.planners()["Dubins"] == "demuvo.planners:Dubins"
