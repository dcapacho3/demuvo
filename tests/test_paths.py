"""Tests for the path helpers."""

import math

import numpy as np
import pytest

from demuvo.paths import along_path, distance_to_path, with_headings


def test_each_pose_looks_along_its_step_and_the_last_has_the_given_angle():
    """East, then north, then the angle asked for."""
    path = with_headings(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), 2.5)
    assert path[:, :2].tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert path[:, 2] == pytest.approx([0.0, math.pi / 2, 2.5])


def test_one_point_has_only_the_given_angle():
    """No step to look along."""
    assert with_headings(np.array([[3.0, 4.0]]), 0.7).tolist() == [[3.0, 4.0, 0.7]]


def test_along_path_spaces_points_no_farther_than_asked_and_keeps_the_ends():
    """A 1 m path at 0.3 m spacing gives 5 points 0.25 m apart; a point stays one."""
    points = along_path(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 0.3)
    assert points[:, 0].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert along_path(np.array([[2.0, 3.0, 0.0]]), 0.3).tolist() == [[2.0, 3.0]]


def test_distance_to_path_measures_to_the_segments_not_only_the_points():
    """Beside a long segment, past its end, to one point, and to no path at all."""
    path = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0]])
    assert distance_to_path((2.0, 1.0), path) == pytest.approx(1.0)
    assert distance_to_path((-3.0, 4.0), path) == pytest.approx(5.0)
    assert distance_to_path((5.0, 2.0), path) == pytest.approx(1.0)
    assert distance_to_path((1.0, 1.0), path[:1]) == pytest.approx(math.sqrt(2))
    assert distance_to_path((1.0, 1.0), np.empty((0, 3))) == math.inf
