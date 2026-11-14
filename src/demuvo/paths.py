"""Paths - helpers for a path of (x, y, angle) poses."""

from __future__ import annotations

import math

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


def along_path(path: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the path's polyline, no more than `spacing` apart, in order."""
    xy = path[:, :2]
    if len(xy) < 2:
        return xy.copy()
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    at = np.linspace(0.0, along[-1], max(2, math.ceil(along[-1] / spacing) + 1))
    return np.column_stack(
        [np.interp(at, along, xy[:, 0]), np.interp(at, along, xy[:, 1])]
    )


def distance_to_path(point, path: np.ndarray) -> float:
    """Metres from a point (x, y) to the nearest point of the path's polyline.

    The path's segments count, not only its points, so a sparse path measures
    the same as a dense one. An empty path is infinitely far.
    """
    xy = path[:, :2]
    if len(xy) == 0:
        return math.inf
    point = np.asarray(point, dtype=float)
    if len(xy) == 1:
        return float(math.hypot(*(xy[0] - point)))
    a, b = xy[:-1], xy[1:]
    ab = b - a
    length2 = (ab * ab).sum(axis=1)
    t = ((point - a) * ab).sum(axis=1) / np.where(length2 > 0, length2, 1.0)
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return float(np.hypot(*(closest - point).T).min())
