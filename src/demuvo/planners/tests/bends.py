"""Checks that the planner tests share."""

import numpy as np
from robomath import wrap_angle


def max_curvature(path, spacing):
    """The largest heading change per metre along a path; the points must be close.

    Raises:
        AssertionError: if two points are farther apart than `spacing`.
    """
    steps = np.diff(path[:, :2], axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    assert lengths.max() <= spacing + 1e-9  # points are spacing apart at most
    turned = np.abs([float(wrap_angle(d)) for d in np.diff(path[:, 2])])
    driven = lengths > 1e-12
    return float((turned[driven] / lengths[driven]).max())
