"""The loop every style runs: draw a world, check it, keep it or draw another.

A style says how to fill a grid with blocked cells and nothing else. What happens
round that is the same whatever it drew: the arguments are checked, the clearance
learns its robot, the grid becomes obstacles that a world file would accept, the
world is measured by `connectivity`, and a world that fails is thrown away and the
next one drawn from the next stream. Keeping that here is what makes two styles
agree on the contract of section 3 of the design, and what a third will inherit.
"""

from __future__ import annotations

import math
from dataclasses import replace

from ..sensing import stream
from ..world import World
from ..world.regions import drop_thin
from .connectivity import (
    Clearance,
    Connectivity,
    check_connectivity,
    clearance_for,
    refused,
)
from .result import Generated
from .shapes import anchor_points, check_base, grid_obstacles


def draw_world(  # pylint: disable=too-many-arguments,too-many-locals
    style: str,
    size: tuple[float, float],
    seed: int,
    *,
    mask_of,
    resolution: float,
    pieces: str,
    prefix: str,
    origin: tuple[float, float],
    base: World | None,
    anchors: tuple[tuple[float, float], ...],
    attempts: int,
    clearance: Clearance | None,
    advice: str = "",
    thin: int = 2,
) -> Generated:
    """Draw worlds in this style until one passes the check; see the module docstring.

    Args:
        style: the name of the style, which names its random stream and starts
            the message when every attempt failed.
        size: width along x and height along y in metres.
        seed: the seed of the map, and the seed of the world that comes back.
        mask_of: what the style draws, called as
            `mask_of(rng, rows, cols, clearance, anchors)` and giving back the
            blocked cells and the continuous field behind them, or None for a
            style that has no field.
        resolution: side of a cell of that grid in metres.
        pieces: "polygons" or "boxes", how the blocked cells become shapes.
        prefix: what the obstacles of this style are called.
        origin: where the lower-left corner of the world is.
        base: a world to add the obstacles to, or None for a new one.
        anchors: points that must be free and in the largest free region, beyond
            the robots and goals the base world already carries.
        attempts: how many worlds to draw before giving up.
        clearance: what the free space must be like, or None for no check.
        advice: what the caller might change, in a sentence of the style's own.
        thin: obstacles narrower than this many cells are erased from the grid, as
            they are traces of the grid and no wall. A style that draws walls one
            cell thick gives 1.

    Returns:
        The world and the grids it was drawn from; see `Generated`.

    Raises:
        GenerateError: if no world passed the check in `attempts` tries.
        WorldError: if the base world does not fit, or nothing says how big the
            robot is.
        ValueError: for an argument that cannot be, before any work is done.
    """
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError(f"attempts must be 1 or more, got {attempts!r}")
    size = (float(size[0]), float(size[1]))
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"size must be two lengths above 0, got {size!r}")
    check_base(base, size, origin, prefix)
    measure = clearance_for(clearance, base)
    points = anchor_points(base, anchors)
    rows = math.ceil(round(size[1] / resolution, 9))
    cols = math.ceil(round(size[0] / resolution, 9))
    bounds = (origin[0], origin[1], origin[0] + size[0], origin[1] + size[1])
    last = Connectivity(False, (), 0.0, 0, 0.0)
    for attempt in range(attempts):
        rng = stream(seed, "generate", style, str(attempt))
        mask, field = mask_of(rng, rows, cols, measure, points)
        drop_thin(mask, thin)
        obstacles = grid_obstacles(
            mask, resolution, origin, pieces=pieces, prefix=prefix, bounds=bounds
        )
        world = (
            World(size=size, seed=seed, obstacles=obstacles, robots=(), origin=origin)
            if base is None
            else replace(base, seed=seed, obstacles=base.obstacles + obstacles)
        )
        report = check_connectivity(world, measure, anchors)
        if report.ok or clearance is None:
            return Generated(world, field, mask, resolution, report, attempt)
        last = report
    raise refused(style, attempts, last, advice)
