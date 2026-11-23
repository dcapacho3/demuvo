"""The obstacles of a grid of blocked cells, for every style that draws one.

The tracing itself is `world.regions`, which the Gazebo mesh cut uses too. This adds
what a generated world needs on top of it: real `Obstacle` objects instead of file
tables, a name for each one, the gate every shape of a world file goes through, and
the check that the world they are added to has room for them.
"""

from __future__ import annotations

import re

import numpy as np
from robomath import Transform2D

from ..verify import WorldError
from ..world import Obstacle, World
from ..world.regions import vectorize
from ..world.shapes import parse_shape

# What the obstacles of a generated map are called, before their number.
DEFAULT_PREFIX = "field"


def check_base(
    base: World | None,
    size: tuple[float, float],
    origin: tuple[float, float],
    prefix: str = DEFAULT_PREFIX,
) -> None:
    """Refuse a base world a style cannot add its obstacles to.

    Args:
        base: the world the clutter is added to, or None for a new one.
        size: the size the caller asked for, which must be the base world's.
        origin: the origin the caller asked for, which must be the base world's.
        prefix: what the style will call its own obstacles.

    Raises:
        WorldError: if the size or the origin differs from the base world's, or
            the base world already has an obstacle named the way the style names
            its own, which a world file refuses as a name used twice.
    """
    if base is None:
        return
    if tuple(size) != tuple(base.size) or tuple(origin) != tuple(base.origin):
        raise WorldError(
            f"generate: the base world is {base.size[0]} by {base.size[1]} m at "
            f"{base.origin}, and {tuple(size)} at {tuple(origin)} was asked for; "
            f"leave the size and the origin to the base world"
        )
    taken = sorted(
        o.name for o in base.obstacles if re.fullmatch(rf"{prefix} \d+", o.name)
    )
    if taken:
        raise WorldError(
            f"generate: the base world already has an obstacle called "
            f"'{taken[0]}', and this style names its own that way; rename it"
        )


def anchor_points(
    base: World | None, anchors: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    """Every point a style must leave standable, in the order the check names them.

    The robots of the base world come first, then the goals of its tasks, then the
    extra points the caller named.
    """
    points: list[tuple[float, float]] = []
    if base is not None:
        points += [tuple(r.pose.translation.tolist()) for r in base.robots]
        points += [tuple(t.goal.translation.tolist()) for t in base.tasks]
    return tuple(points) + tuple((float(x), float(y)) for x, y in anchors)


def grid_obstacles(  # pylint: disable=too-many-arguments
    mask: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    *,
    pieces: str = "polygons",
    prefix: str = DEFAULT_PREFIX,
    first: int = 1,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[Obstacle, ...]:
    """The obstacles of a grid of blocked cells, one or more for each region.

    A region becomes one polygon with the staircase of its cells straightened, or a
    box when it is a rectangle; a region a polygon cannot hold is tiled with boxes.
    Every shape passes `parse_shape`, so a generated obstacle can never be one the
    world file would refuse.

    The mask is taken as it is: nothing is dropped as a speck and nothing is
    cleaned, because each style knows which cleaning its grid needs and has done it.

    Args:
        mask: True where the ground is blocked, row 0 at the lowest y.
        resolution: side of a cell in metres.
        origin: world position (x, y) of the corner of cell (0, 0).
        pieces: "polygons" to trace each region, or "boxes" to tile them all.
        prefix: what the obstacles are called, before their number.
        first: the number of the first obstacle.
        bounds: (x_min, y_min, x_max, y_max) that every shape must stay inside,
            or None for no edge to keep. A world's bounds belong here: the loader
            refuses an obstacle that reaches past them, and straightening the
            staircase of a region that the edge cut off flat can send a corner
            well outside. Such a region is tiled with boxes instead.

    Returns:
        The obstacles, in row-major order of each region's first cell, named
        "field 1", "field 2" and so on.
    """
    tables, _ = vectorize(
        mask, origin, resolution, min_cells=1, pieces=pieces, bounds=bounds
    )
    found = []
    for number, item in enumerate(tables, start=first):
        name = f"{prefix} {number}"
        shape = parse_shape(item["shape"], name)
        pose = (
            Transform2D.from_xyt(item["pose"])
            if "pose" in item
            else Transform2D.identity()
        )
        found.append(Obstacle(pose, shape, name))
    return tuple(found)
