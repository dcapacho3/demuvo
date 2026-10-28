"""Tests for the planners."""

import math

import numpy as np
import pytest

from demuvo import StraightLine


def test_the_path_runs_from_the_start_to_the_goal():
    """Points every `spacing` metres; the last one has the goal's angle."""
    path = StraightLine(spacing=0.25).plan(None, (0.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    assert path.shape == (5, 3)
    assert path[:, 0] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert path[:, 1] == pytest.approx(0.0)
    assert path[:, 2] == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])


def test_the_angle_along_the_path_is_the_heading_of_the_line():
    """A diagonal path points along the diagonal."""
    path = StraightLine().plan(None, (1.0, 1.0, 0.0), (3.0, 3.0, 0.5))
    assert path[0, 2] == pytest.approx(math.pi / 4)
    assert path[-1] == pytest.approx([3.0, 3.0, 0.5])


def test_a_goal_at_the_start_gives_two_points():
    """No distance to cover: the path keeps the start's angle, then the goal's."""
    path = StraightLine().plan(None, (2.0, 2.0, 0.3), (2.0, 2.0, 1.0))
    assert path == pytest.approx(np.array([[2.0, 2.0, 0.3], [2.0, 2.0, 1.0]]))


@pytest.mark.parametrize("spacing", [0.0, -0.1, math.nan, math.inf, True, "0.1"])
def test_the_spacing_must_be_a_positive_number(spacing):
    """A bad spacing is refused when the planner is built."""
    with pytest.raises(ValueError, match="spacing"):
        StraightLine(spacing=spacing)
