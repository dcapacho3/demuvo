"""Random start and goal pairs, picked out of free space the check has measured.

A map is half of a benchmark; the other half is where the robot starts and where it
must get to. Both are picked from the largest free region of a `Connectivity`
report, which is the region the check holds every anchor to, so a pair is reachable
one from the other by the robot the report was measured for. Nothing here rasterises
a world or searches it: the report carries the regions of every cell, and this reads
them.

The two are measured apart in a straight line and not along a path. A path distance
would need a search, and making the planner do that search is the point of the pair.
"""

from __future__ import annotations

import numpy as np

from ..sensing import stream
from .connectivity import Connectivity, GenerateError

Point = tuple[float, float]


def _points(places: np.ndarray, origin: tuple[float, float], resolution: float):
    """The world positions of the middles of some cells, as x and y arrays.

    Distances are measured between these and not between the cells they came from,
    so that a pair is as far apart as was asked when the caller measures it in the
    world, and not only in the grid. The two differ by the last bit of a float,
    which is enough to fail a test and, worse, to be a promise half kept.
    """
    return (
        origin[0] + (places[:, 1] + 0.5) * resolution,
        origin[1] + (places[:, 0] + 0.5) * resolution,
    )


def _across(xs: np.ndarray, ys: np.ndarray) -> float:
    """How far apart the two furthest corners of a region's box are, in metres."""
    if xs.size == 0:
        return 0.0
    return float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def place_pair(
    report: Connectivity,
    seed: int,
    *,
    min_distance: float,
    max_distance: float | None = None,
    tries: int = 100,
) -> tuple[Point, Point]:
    """Pick a start and a goal in the largest free region of a report.

    Both lie where the robot the report was measured for may stand, and both lie in
    one region, so one is reachable from the other. A start is picked at random and
    then a goal from everywhere that is the right distance from it, which is why a
    pair is found when one exists rather than when the dice allow; `tries` is how
    many starts are tried before the distance is called impossible.

    Two calls with equal arguments give the same pair: the numbers come from
    `stream(seed, "generate", "pair")`, which depends on the seed and nothing else.

    Args:
        report: a free-space report, whose `labels`, `resolution` and `origin` say
            where the robot may stand.
        seed: the seed of the pair.
        min_distance: the least the two may be apart, in metres, in a straight line.
        max_distance: the most they may be apart, or None for no limit.
        tries: how many starts to try before giving up.

    Returns:
        The start and the goal, as (x, y) in the world.

    Raises:
        GenerateError: if the report has no free region, or no two places in it are
            as far apart as asked.
        ValueError: if the distances or the tries cannot be.
    """
    if min_distance < 0:
        raise ValueError(f"min_distance must be 0 or more, got {min_distance!r}")
    if max_distance is not None and max_distance < min_distance:
        raise ValueError(
            f"max_distance must be at least min_distance, got {max_distance!r}"
        )
    if not isinstance(tries, int) or isinstance(tries, bool) or tries < 1:
        raise ValueError(f"tries must be 1 or more, got {tries!r}")
    places = np.argwhere(report.labels == 0)
    if places.size == 0:
        raise GenerateError(
            "pair: this map has nowhere for the robot to stand, so it has no start"
        )
    rng = stream(seed, "generate", "pair")
    xs, ys = _points(places, report.origin, report.resolution)
    for _ in range(tries):
        at = int(rng.integers(len(places)))
        away = np.hypot(xs - xs[at], ys - ys[at])
        allowed = away >= min_distance
        if max_distance is not None:
            allowed &= away <= max_distance
        found = np.flatnonzero(allowed)
        if found.size:
            other = int(found[int(rng.integers(len(found)))])
            return (float(xs[at]), float(ys[at])), (float(xs[other]), float(ys[other]))
    limit = "" if max_distance is None else f" and at most {max_distance:.2f} m"
    raise GenerateError(
        f"pair: no two places the robot can stand are at least {min_distance:.2f} m"
        f"{limit} apart; the free space is about {_across(xs, ys):.2f} m "
        f"across at its widest"
    )
