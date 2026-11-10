"""The layers of the global costmap - static (the map) and inflation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..world import World
from ..world.mapfile import read_map
from ..world.shapes import Part, border_parts
from .grid import FREE, INSCRIBED, LETHAL, MAX_NON_OBSTACLE, NO_INFORMATION


def edge_distance(points: np.ndarray, vertices) -> np.ndarray:
    """Distance from each point (m, 2) to the outline of the vertices."""
    v = np.asarray(vertices, dtype=float)
    a, b = (v, np.roll(v, -1, axis=0)) if len(v) > 2 else (v[:1], v[-1:])
    ab = b - a
    ap = points[:, None, :] - a[None]
    length2 = np.maximum((ab * ab).sum(axis=1), 1e-30)
    t = np.clip((ap * ab[None]).sum(axis=2) / length2, 0.0, 1.0)
    return np.linalg.norm(ap - t[..., None] * ab[None], axis=2).min(axis=1)


def distance_to_part(points: np.ndarray, part: Part) -> np.ndarray:
    """Distance from each point (m, 2) to a convex part; 0 inside it."""
    d = edge_distance(points, part.vertices)
    if len(part.vertices) > 2:
        v = np.asarray(part.vertices, dtype=float)
        edge = np.roll(v, -1, axis=0) - v
        rel = points[:, None, :] - v[None]
        cross = edge[None, :, 0] * rel[..., 1] - edge[None, :, 1] * rel[..., 0]
        d = np.where((cross >= 0).all(axis=1), 0.0, d)
    return np.maximum(d - part.radius, 0.0)


def _mark(
    grid: np.ndarray,
    part: Part,
    resolution: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Set to LETHAL every cell whose centre is within half a cell of the part.

    The origin is the world position of the corner of cell (0, 0).
    """
    reach = part.radius + resolution
    x0, y0, x1, y1 = part.box
    ox, oy = origin
    c0 = math.floor((x0 - reach - ox) / resolution)
    c1 = math.ceil((x1 + reach - ox) / resolution)
    r0 = math.floor((y0 - reach - oy) / resolution)
    r1 = math.ceil((y1 + reach - oy) / resolution)
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, grid.shape[1]), min(r1, grid.shape[0])
    if c0 >= c1 or r0 >= r1:
        return
    rows, cols = np.mgrid[r0:r1, c0:c1]
    centres = np.stack([cols.ravel() + 0.5, rows.ravel() + 0.5], axis=1)
    centres = centres * resolution + origin
    # the small margin keeps a wall on a cell edge from slipping between two cells
    near = distance_to_part(centres, part) <= resolution / 2 + 1e-9
    grid[rows.ravel()[near], cols.ravel()[near]] = LETHAL


def generate_static(world: World, resolution: float) -> np.ndarray:
    """The obstacles and the border of the world as cells.

    Cell (0, 0) has its corner at the world's origin.

    A cell is LETHAL when its centre lies within half a cell of a shape, so a
    shape never hides between cells; a cell touching a shape's corner can be
    marked a little early.
    """
    width, height = world.size
    origin = world.origin
    grid = np.zeros(
        (
            math.ceil(round(height / resolution, 9)),
            math.ceil(round(width / resolution, 9)),
        ),
        dtype=np.uint8,
    )
    parts = border_parts(world.bounds)
    for obstacle in world.obstacles:
        parts += obstacle.shape.parts(obstacle.pose)
    for part in parts:
        _mark(grid, part, resolution, origin)
    return grid


def load_static(
    path: Path, unknown: str = "free"
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Read a `map.yaml` and its image as cells, a resolution and an origin.

    A cell is LETHAL when its occupancy reaches `occupied_thresh` (default 0.65)
    and FREE when it is below `free_thresh` (default 0.25). A cell between the
    two is unknown: FREE if `unknown` is "free", NO_INFORMATION if "blocked".

    Raises:
        WorldError: if the file or its image cannot be read.
    """
    found = read_map(path)
    grid = np.full(found.occupancy.shape, FREE, dtype=np.uint8)
    if unknown == "blocked":
        grid[found.occupancy >= found.free_thresh] = NO_INFORMATION
    grid[found.occupancy >= found.occupied_thresh] = LETHAL
    return grid, found.resolution, found.origin


def inflate(
    grid: np.ndarray,
    resolution: float,
    inscribed_radius: float,
    inflation_radius: float,
    cost_scaling_factor: float,
) -> np.ndarray:
    """A copy of the grid with cost spread out from the LETHAL cells.

    A cell closer than the inscribed radius to a LETHAL cell costs INSCRIBED;
    farther out, 252 * exp(-cost_scaling_factor * (distance - inscribed radius)),
    down to the inflation radius (at least the inscribed radius).
    """
    lethal = grid == LETHAL
    reach = math.ceil(round(max(inflation_radius, inscribed_radius) / resolution, 9))
    out = grid.copy()
    rows, cols = grid.shape
    # shortcut: one pass per offset, so rows*cols*reach^2 work; use a distance
    # transform if this is slow
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            distance = math.hypot(dy, dx) * resolution
            if (dy == 0 and dx == 0) or math.hypot(dy, dx) > reach:
                continue
            if distance <= inscribed_radius:
                cost = INSCRIBED
            else:
                cost = int(
                    MAX_NON_OBSTACLE
                    * math.exp(-cost_scaling_factor * (distance - inscribed_radius))
                )
            if cost == 0:
                continue
            dst = (
                slice(max(dy, 0), rows + min(dy, 0)),
                slice(max(dx, 0), cols + min(dx, 0)),
            )
            src = (
                slice(max(-dy, 0), rows + min(-dy, 0)),
                slice(max(-dx, 0), cols + min(-dx, 0)),
            )
            view = out[dst]
            np.maximum(
                view, np.where(lethal[src], np.uint8(cost), np.uint8(0)), out=view
            )
    return out
