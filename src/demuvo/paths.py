"""Paths - helpers for a path of (x, y, angle) poses."""

from __future__ import annotations

import math

import numpy as np
from robomath import rplus_se2, se2_to_xyt, wrap_angle, xyt_to_se2


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


def arc_poses(start, curvature: float, distances) -> np.ndarray:
    """The poses reached from `start` by driving signed distances along one arc.

    The arc bends by `curvature` (1 / radius, positive to the left, 0 for a straight
    line); a negative distance drives backward. Shape (len(distances), 3).
    """
    drive = np.asarray(distances, dtype=float)
    twist = np.column_stack([drive, np.zeros_like(drive), curvature * drive])
    return se2_to_xyt(rplus_se2(xyt_to_se2(np.asarray(start, dtype=float)), twist))


def trace_segments(start, segments, goal, radius: float, spacing: float) -> np.ndarray:
    """Poses `spacing` apart along arcs and straights, ending exactly on the goal.

    Args:
        start: (x, y, angle).
        segments: (kind, length) pairs, kind "S" for straight, "L" or "R" for an arc
            to the left or right of `radius`; the length is in radii, and negative
            drives backward.
        goal: (x, y, angle), put as the last pose instead of the float rounding of it.
        radius: the tightest bend, in metres.
        spacing: the most a step may be, in metres.
    """
    poses, pose = [np.array(start, dtype=float)], tuple(start)
    curvatures = {"S": 0.0, "L": 1.0 / radius, "R": -1.0 / radius}
    for kind, length in segments:
        if abs(length) * radius < 1e-12:
            continue
        steps = max(1, math.ceil(abs(length) * radius / spacing))
        at = np.linspace(length / steps, length, steps)  # signed, in radii
        arc = arc_poses(pose, curvatures[kind], at * radius)
        poses.append(arc)
        pose = tuple(arc[-1])
    path = np.concatenate([np.atleast_2d(p) for p in poses])
    path[:, 2] = wrap_angle(path[:, 2])
    path[-1] = goal
    return path
