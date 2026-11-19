"""Tests for the Reeds-Shepp planner."""

# pylint: disable=protected-access  # these tests check the module's own helpers

import math

import numpy as np
import pytest
from robomath import Transform2D, wrap_angle

from demuvo import options
from demuvo.planners import ReedsShepp, dubins
from demuvo.planners import reeds_shepp as rs

from .bends import max_curvature


def poses(start, goal):
    """Two (x, y, angle) tuples as transforms."""
    return Transform2D.from_xyt(start), Transform2D.from_xyt(goal)


def plan(start, goal, **params):
    """The Reeds-Shepp path between two (x, y, angle) poses."""
    return ReedsShepp(**params).plan(*poses(start, goal))


def drive(start, word, radius):
    """Where a word ends, driven exactly: each segment is one constant twist."""
    pose = Transform2D.from_xyt(start)
    for kind, length in zip(*word):
        turn = 0.0 if kind == "S" else (length if kind == "L" else -length)
        pose = pose.rplus(np.array([length * radius, 0.0, turn]))
    return np.asarray(pose.xyt, float)


def test_the_path_starts_and_ends_on_the_poses():
    """First row is the start, last row is the goal, angle included, exactly."""
    path = plan((1.0, 1.0, 0.0), (0.4, 1.6, 0.3))
    assert path[0] == pytest.approx([1.0, 1.0, 0.0])
    assert path[-1] == pytest.approx([0.4, 1.6, 0.3])


def test_a_goal_straight_behind_is_backed_to():
    """Aligned poses, goal behind: one reverse straight, no loop."""
    path = plan((1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    steps = np.diff(path[:, :2], axis=0)
    assert np.hypot(steps[:, 0], steps[:, 1]).sum() == pytest.approx(1.0)
    assert path[:, 2] == pytest.approx(np.zeros(len(path)))  # heading kept


def test_a_start_on_the_goal_is_one_point():
    """Nothing to drive: the path is the start alone."""
    path = plan((2.0, 2.0, 0.5), (2.0, 2.0, 0.5))
    assert path.shape == (1, 3)


@pytest.mark.parametrize("seed", range(30))
def test_every_word_that_exists_ends_on_the_goal(seed):
    """Each of the 48 words found, not just the shortest, drives to the goal."""
    rng = np.random.default_rng(seed)
    start = rng.uniform([-3, -3, -math.pi], [3, 3, math.pi])
    goal = rng.uniform([-3, -3, -math.pi], [3, 3, math.pi])
    radius = 0.8
    (x0, y0, yaw0), (x1, y1, yaw1) = start, goal
    c, s = math.cos(yaw0), math.sin(yaw0)
    x = (c * (x1 - x0) + s * (y1 - y0)) / radius
    y = (-s * (x1 - x0) + c * (y1 - y0)) / radius
    words = rs._words(x, y, float(wrap_angle(yaw1 - yaw0)))
    assert words
    for word in words:
        end = drive(start, word, radius)
        assert end[:2] == pytest.approx(goal[:2], abs=1e-6)
        assert float(wrap_angle(end[2] - goal[2])) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(30))
def test_the_path_is_never_longer_than_dubins(seed):
    """Driving forward only is one of the choices, so reversing cannot cost more."""
    rng = np.random.default_rng(seed)
    start = tuple(rng.uniform([-3, -3, -math.pi], [3, 3, math.pi]))
    goal = tuple(rng.uniform([-3, -3, -math.pi], [3, 3, math.pi]))
    word = ReedsShepp(turning_radius=0.8).word(*poses(start, goal))
    (x0, y0, yaw0), (x1, y1, yaw1) = start, goal
    gap = math.hypot(x1 - x0, y1 - y0)
    theta = math.atan2(y1 - y0, x1 - x0)
    shortest = min(
        sum(found[:3])
        for found in dubins._words(
            dubins._mod2pi(yaw0 - theta), dubins._mod2pi(yaw1 - theta), gap / 0.8
        )
    )
    assert rs._length(word) <= shortest + 1e-9


@pytest.mark.parametrize("seed", range(20))
def test_the_path_never_bends_tighter_than_the_turning_radius(seed):
    """Heading change per metre stays within 1 / turning_radius, pose to pose."""
    rng = np.random.default_rng(seed)
    start = tuple(rng.uniform([-3, -3, -math.pi], [3, 3, math.pi]))
    goal = tuple(rng.uniform([-3, -3, -math.pi], [3, 3, math.pi]))
    path = plan(start, goal, turning_radius=0.8, spacing=0.02)
    assert max_curvature(path, 0.02) <= 1 / 0.8 + 0.05


def test_a_sideways_shift_has_a_cusp():
    """Parallel parking: the path changes direction at least once."""
    path = plan((0.0, 0.0, 0.0), (0.0, 0.8, 0.0), turning_radius=1.0)
    steps = np.diff(path[:, :2], axis=0)
    along = np.cos(path[:-1, 2]) * steps[:, 0] + np.sin(path[:-1, 2]) * steps[:, 1]
    assert (along > 0).any() and (along < 0).any()


@pytest.mark.parametrize("name", ["turning_radius", "spacing"])
@pytest.mark.parametrize("bad", [0.0, -0.5, math.nan])
def test_a_bad_setting_is_refused_by_name(name, bad):
    """Turning radius and spacing must be positive numbers."""
    with pytest.raises(ValueError, match=name):
        ReedsShepp(**{name: bad})


def test_the_options_list_reeds_shepp():
    """The inspector offers the planner by its friendly name."""
    assert options.planners()["Reeds-Shepp"] == "demuvo.planners:ReedsShepp"
