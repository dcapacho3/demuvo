"""Paths - helpers for a path of (x, y, angle) poses."""

from __future__ import annotations

import numpy as np


def with_headings(points: np.ndarray, last_angle: float) -> np.ndarray:
    """Poses (N, 3) from points (N, 2).

    Each pose looks along its step to the next point, and the last has `last_angle`.
    """
    path = np.empty((len(points), 3))
    path[:, :2] = points
    step = np.diff(points, axis=0)
    path[:-1, 2] = np.arctan2(step[:, 1], step[:, 0])
    path[-1, 2] = last_angle
    return path
