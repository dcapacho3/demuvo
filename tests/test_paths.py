"""Tests for the path helpers."""

import math

import numpy as np
import pytest

from demuvo.paths import with_headings


def test_each_pose_looks_along_its_step_and_the_last_has_the_given_angle():
    """East, then north, then the angle asked for."""
    path = with_headings(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), 2.5)
    assert path[:, :2].tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert path[:, 2] == pytest.approx([0.0, math.pi / 2, 2.5])


def test_one_point_has_only_the_given_angle():
    """No step to look along."""
    assert with_headings(np.array([[3.0, 4.0]]), 0.7).tolist() == [[3.0, 4.0, 0.7]]
