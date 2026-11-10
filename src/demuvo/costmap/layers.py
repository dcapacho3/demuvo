"""The layers of the global costmap - static (the map) and inflation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..verify import WorldError
from ..world import World
from ..world.shapes import Part, border_parts
from .grid import FREE, INSCRIBED, LETHAL, MAX_NON_OBSTACLE


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


def _mark(grid: np.ndarray, part: Part, resolution: float) -> None:
    """Set to LETHAL every cell whose centre is within half a cell of the part."""
    reach = part.radius + resolution
    x0, y0, x1, y1 = part.box
    c0, c1 = math.floor((x0 - reach) / resolution), math.ceil((x1 + reach) / resolution)
    r0, r1 = math.floor((y0 - reach) / resolution), math.ceil((y1 + reach) / resolution)
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, grid.shape[1]), min(r1, grid.shape[0])
    if c0 >= c1 or r0 >= r1:
        return
    rows, cols = np.mgrid[r0:r1, c0:c1]
    centres = (
        np.stack([(cols.ravel() + 0.5), (rows.ravel() + 0.5)], axis=1) * resolution
    )
    # the small margin keeps a wall on a cell edge from slipping between two cells
    near = distance_to_part(centres, part) <= resolution / 2 + 1e-9
    grid[rows.ravel()[near], cols.ravel()[near]] = LETHAL


def generate_static(world: World, resolution: float) -> np.ndarray:
    """The obstacles and the border of the world as cells.

    A cell is LETHAL when its centre lies within half a cell of a shape, so a
    shape never hides between cells; a cell touching a shape's corner can be
    marked a little early.
    """
    width, height = world.size
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
        _mark(grid, part, resolution)
    return grid


def _read_pgm(path: Path) -> np.ndarray:
    """The brightness of a PGM image (P2 or P5) as floats from 0 to 1, row 0 on top."""
    data = path.read_bytes()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            pos = data.index(b"\n", pos)
            continue
        end = pos
        while end < len(data) and not data[end : end + 1].isspace():
            end += 1
        tokens.append(data[pos:end])
        pos = end
    magic, width, height, maxval = tokens[0], *(int(t) for t in tokens[1:])
    if magic == b"P5":
        dtype = np.uint8 if maxval < 256 else ">u2"
        pixels = np.frombuffer(data, dtype=dtype, count=width * height, offset=pos + 1)
    elif magic == b"P2":
        pixels = np.array(data[pos:].split(), dtype=float)[: width * height]
    else:
        raise WorldError(f"{path}: only PGM images (P2 or P5) can be read")
    return pixels.reshape(height, width) / maxval


def load_static(path: Path) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Read a `map.yaml` and its image as cells, a resolution and an origin.

    A cell is LETHAL when its occupancy, `1 - brightness` (or `brightness` if
    `negate` is set), reaches `occupied_thresh` (default 0.65). Every other cell
    is FREE. The `mode` and `free_thresh` keys do nothing here.
    The origin's angle must be 0.

    Raises:
        WorldError: if the file or its image cannot be read.
    """
    try:
        keys = {}
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                key, _, value = line.partition(":")
                keys[key.strip()] = value.strip().strip("'\"")
        resolution = float(keys["resolution"])
        origin = [float(v) for v in keys["origin"].strip("[]").split(",")]
        image = _read_pgm(path.parent / keys["image"])
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise WorldError(f"costmap: cannot read map '{path}': {error!r}") from error
    if resolution <= 0 or len(origin) != 3 or origin[2] != 0:
        raise WorldError(
            f"costmap: map '{path}' needs resolution > 0 and an origin with angle 0"
        )
    occupancy = image if keys.get("negate", "0") in ("1", "true") else 1.0 - image
    lethal = occupancy >= float(keys.get("occupied_thresh", 0.65))
    grid = np.where(lethal, LETHAL, FREE).astype(np.uint8)
    grid = grid[::-1].copy()  # image row 0 is the top; cell row 0 is the lowest y
    return grid, resolution, (origin[0], origin[1])


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
