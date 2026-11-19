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


def part_cells(
    part: Part, resolution: float, origin: tuple[float, float]
) -> tuple[int, int, int, int]:
    """The cells a part can touch: first row, end row, first column, end column.

    They are counted from the corner of the world and are not clipped to a grid.
    """
    reach = part.radius + resolution
    x0, y0, x1, y1 = part.box
    ox, oy = origin
    return (
        math.floor((y0 - reach - oy) / resolution),
        math.ceil((y1 + reach - oy) / resolution),
        math.floor((x0 - reach - ox) / resolution),
        math.ceil((x1 + reach - ox) / resolution),
    )


def _mark(
    grid: np.ndarray,
    part: Part,
    resolution: float,
    origin: tuple[float, float] = (0.0, 0.0),
    first: tuple[int, int] = (0, 0),
) -> None:
    """Set to LETHAL every cell whose centre is within half a cell of the part.

    The origin is the world position of the corner of cell (0, 0) of the world, and
    `first` the row and column of the world that the grid's own cell (0, 0) is, so a
    piece of the map gets the same cells as the whole of it would.
    """
    r0, r1, c0, c1 = part_cells(part, resolution, origin)
    r0, c0 = max(r0 - first[0], 0), max(c0 - first[1], 0)
    r1, c1 = min(r1 - first[0], grid.shape[0]), min(c1 - first[1], grid.shape[1])
    if c0 >= c1 or r0 >= r1:
        return
    rows, cols = np.mgrid[r0:r1, c0:c1]
    centres = np.stack(
        [cols.ravel() + first[1] + 0.5, rows.ravel() + first[0] + 0.5], axis=1
    )
    centres = centres * resolution + origin
    # the small margin keeps a wall on a cell edge from slipping between two cells
    near = distance_to_part(centres, part) <= resolution / 2 + 1e-9
    grid[rows.ravel()[near], cols.ravel()[near]] = LETHAL


def world_parts(world: World) -> list[Part]:
    """The border and the obstacles of the world as convex parts."""
    parts = border_parts(world.bounds)
    for obstacle in world.obstacles:
        parts += obstacle.shape.parts(obstacle.pose)
    return parts


def generate_static(
    world: World,
    resolution: float,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """The obstacles and the border of the world as cells.

    Cell (0, 0) has its corner at the world's origin.

    A cell is LETHAL when its centre lies within half a cell of a shape, so a
    shape never hides between cells; a cell touching a shape's corner can be
    marked a little early.

    Args:
        world: the world.
        resolution: side of a cell in metres.
        window: (first row, end row, first column, end column) to give only that part
            of the grid; the cells are those the whole grid has.
    """
    width, height = world.size
    rows = math.ceil(round(height / resolution, 9))
    cols = math.ceil(round(width / resolution, 9))
    r0, r1, c0, c1 = window or (0, rows, 0, cols)
    grid = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    for part in world_parts(world):
        _mark(grid, part, resolution, world.origin, (r0, c0))
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


def _vertical_gap(lethal: np.ndarray, limit: int) -> np.ndarray:
    """Rows from each cell to the nearest LETHAL cell of its column, to limit + 1."""
    rows = lethal.shape[0]
    index = np.arange(rows)[:, None]
    far = 4 * rows + limit + 2
    above = np.maximum.accumulate(np.where(lethal, index, -far), axis=0)
    below = np.minimum.accumulate(np.where(lethal, index, far)[::-1], axis=0)[::-1]
    return np.minimum(np.minimum(index - above, below - index), limit + 1)


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

    The distance to the nearest LETHAL cell is found in two passes: along each column,
    then across the few columns within reach, which is the work of one pass for each
    column of the reach and not one for each cell of it.
    """
    lethal = grid == LETHAL
    reach = math.ceil(round(max(inflation_radius, inscribed_radius) / resolution, 9))
    out = grid.copy()
    if not lethal.any():
        return out
    rows, cols = grid.shape
    gap = _vertical_gap(lethal, reach).astype(np.int32)
    padded = np.full((rows, cols + 2 * reach), reach + 1, dtype=np.int32)
    padded[:, reach : reach + cols] = gap
    nearest = np.full((rows, cols), 2 * (reach + 1) ** 2, dtype=np.int32)
    for dx in range(-reach, reach + 1):
        side = padded[:, reach + dx : reach + dx + cols]
        np.minimum(nearest, dx * dx + side * side, out=nearest)
    # the cost of a cell by its squared distance in cells; none beyond the reach
    table = np.zeros(2 * (reach + 1) ** 2 + 1, dtype=np.uint8)
    for dy in range(reach + 1):
        for dx in range(reach + 1):
            if not 0 < dy * dy + dx * dx <= reach * reach:
                continue
            distance = math.hypot(dy, dx) * resolution
            if distance <= inscribed_radius:
                cost = INSCRIBED
            else:
                cost = int(
                    MAX_NON_OBSTACLE
                    * math.exp(-cost_scaling_factor * (distance - inscribed_radius))
                )
            table[dy * dy + dx * dx] = cost
    np.maximum(out, table[nearest], out=out)
    return out
