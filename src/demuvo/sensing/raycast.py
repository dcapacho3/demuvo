"""Ray casting - the distance along a ray to the first thing it meets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# A ray this close to parallel with a segment is taken as missing it.
_PARALLEL = 1e-12


@dataclass(frozen=True, eq=False)
class Scene:
    """What a ray can hit, as arrays.

    Every obstacle, wall and robot body is a segment or a circle. The owner of
    each says whose it is, so a robot's sensor can leave out its own body.

    Attributes:
        segments: shape (S, 4), each row (x0, y0, x1, y1).
        segment_owner: shape (S,), who each segment belongs to; FIXED for the
            walls and obstacles, a robot's index for its body.
        circles: shape (C, 3), each row (x, y, radius).
        circle_owner: shape (C,), as `segment_owner`.
    """

    segments: np.ndarray
    segment_owner: np.ndarray
    circles: np.ndarray
    circle_owner: np.ndarray

    FIXED: ClassVar[int] = -1


def _segment_hits(origin, directions, segments) -> np.ndarray:
    """Distance from the origin along each direction to each segment, inf if none."""
    start = segments[:, :2]
    edge = segments[:, 2:] - start
    to_start = start - origin
    denominator = directions[:, 0, None] * edge[None, :, 1] - (
        directions[:, 1, None] * edge[None, :, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        along_ray = (
            to_start[:, 0] * edge[:, 1] - to_start[:, 1] * edge[:, 0]
        ) / denominator
        along_edge = (
            to_start[None, :, 0] * directions[:, 1, None]
            - to_start[None, :, 1] * directions[:, 0, None]
        ) / denominator
    hit = (
        (np.abs(denominator) > _PARALLEL)
        & (along_ray >= 0)
        & (along_edge >= 0)
        & (along_edge <= 1)
    )
    return np.where(hit, along_ray, np.inf)


def _circle_hits(origin, directions, circles) -> np.ndarray:
    """Distance from the origin along each direction to each circle, inf if none."""
    to_origin = origin - circles[:, :2]
    half = directions @ to_origin.T
    inside = (to_origin**2).sum(axis=1) - circles[:, 2] ** 2
    discriminant = half**2 - inside
    with np.errstate(invalid="ignore"):
        entry = -half - np.sqrt(discriminant)
    entry = np.where(inside < 0, 0.0, entry)  # an origin inside a circle is a hit at 0
    return np.where((discriminant >= 0) & (entry >= 0), entry, np.inf)


def cast(
    origin: np.ndarray,
    angles: np.ndarray,
    scene: Scene,
    max_range: float,
    skip: int | None = None,
) -> np.ndarray:
    """The distance to the first thing each ray meets, shape (B,).

    A ray that meets nothing within `max_range` gives `inf`, as a ROS laser scan
    does.

    Args:
        origin: where every ray starts, (x, y).
        angles: the angle of each ray in radians, shape (B,).
        scene: what the rays can hit.
        max_range: the farthest a ray reaches, in metres.
        skip: the owner whose segments and circles the rays pass through, such as
            the robot that carries the sensor; None leaves nothing out.
    """
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    ranges = np.full(len(angles), np.inf)
    keep = (
        np.ones(len(scene.segment_owner), bool)
        if skip is None
        else (scene.segment_owner != skip)
    )
    if keep.any():
        ranges = np.minimum(
            ranges, _segment_hits(origin, directions, scene.segments[keep]).min(axis=1)
        )
    keep = (
        np.ones(len(scene.circle_owner), bool)
        if skip is None
        else (scene.circle_owner != skip)
    )
    if keep.any():
        ranges = np.minimum(
            ranges, _circle_hits(origin, directions, scene.circles[keep]).min(axis=1)
        )
    ranges[ranges > max_range] = np.inf
    return ranges
