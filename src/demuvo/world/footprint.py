"""A mesh seen from above, cut to the heights a robot meets.

The triangles are sampled finely, the samples between two heights are marked on a grid,
small enclosed gaps (the inside of a shelf) are filled, and the marked cells become
boxes. A roof, a ceiling or a floor lies outside the heights, so it adds nothing.

The grid is turned into shapes by `regions`, which every grid of blocked cells goes
through, whatever drew it.
"""

from __future__ import annotations

import math

import numpy as np

from .regions import fill_holes, vectorize

# The most samples held in memory at once.
CHUNK = 2_000_000
# The most steps, as a power of two, along an edge when sampling a triangle.
MAX_POWER = 11
# The most cells a grid may have; a bigger world is refused, not guessed at.
MAX_CELLS = 40_000_000


def _samples(triangles: np.ndarray, spacing: float):
    """Points on the surface of the triangles, no farther apart than `spacing`.

    Yields arrays of shape (n, 3). Triangles are taken in groups of like size, each
    covered by the same lattice, so a thin triangle of a long wall is not oversampled.
    """
    origin = triangles[:, 0]
    one = triangles[:, 1] - origin
    two = triangles[:, 2] - origin
    # A lattice of 2**k steps along each of two edges has no gap wider than spacing.
    # shortcut: k stops at MAX_POWER, so an edge over 2**MAX_POWER * spacing (50 m at
    # 0.025) may leave gaps; cut such a triangle smaller in the file.
    steps_one = np.ceil(np.log2(np.maximum(np.linalg.norm(one, axis=1) / spacing, 1)))
    steps_two = np.ceil(np.log2(np.maximum(np.linalg.norm(two, axis=1) / spacing, 1)))
    steps_one, steps_two = (np.minimum(s, MAX_POWER) for s in (steps_one, steps_two))
    keys = steps_one.astype(int) * 64 + steps_two.astype(int)
    for key in np.unique(keys):
        group = np.flatnonzero(keys == key)
        count_one, count_two = 2 ** (key // 64), 2 ** (key % 64)
        a, b = np.meshgrid(
            np.arange(count_one + 1) / count_one, np.arange(count_two + 1) / count_two
        )
        keep = (a + b) <= 1.0 + 1e-9
        a, b = a[keep], b[keep]
        per_chunk = max(1, CHUNK // len(a))
        for start in range(0, len(group), per_chunk):
            part = group[start : start + per_chunk]
            yield (
                origin[part, None, :]
                + a[None, :, None] * one[part, None, :]
                + b[None, :, None] * two[part, None, :]
            ).reshape(-1, 3)


def _cut(triangles: np.ndarray, height: float) -> np.ndarray:
    """The segments (x, y) where a plane at a height crosses the triangles.

    Shape (S, 2, 2). A triangle that has all corners on one side adds none.
    """
    z = triangles[:, :, 2] - (height + 1e-7)  # off any corner, so none lies on it
    above = z > 0
    crosses = above.any(axis=1) & ~above.all(axis=1)
    tri, z, above = triangles[crosses], z[crosses], above[crosses]
    # the corner that is alone on its side of the plane
    alone = np.where((above.sum(axis=1) == 1)[:, None], above, ~above).argmax(axis=1)
    n = np.arange(len(tri))
    p0, p1, p2 = (tri[n, (alone + k) % 3] for k in range(3))
    z0, z1, z2 = (z[n, (alone + k) % 3] for k in range(3))
    a = p0 + (p1 - p0) * (z0 / (z0 - z1))[:, None]
    b = p0 + (p2 - p0) * (z0 / (z0 - z2))[:, None]
    return np.stack([a[:, :2], b[:, :2]], axis=1)


def _inside(grid: np.ndarray, corner, segments: np.ndarray, resolution: float) -> None:
    """Mark the cells whose centres lie inside closed segments, by the even-odd rule."""
    y_of = segments[:, :, 1]
    low, high = y_of.min(axis=1), y_of.max(axis=1)
    for r in range(grid.shape[0]):
        y = corner[1] + (r + 0.5) * resolution
        hit = segments[(low <= y) & (y < high)]
        if len(hit) < 2:
            continue
        t = (y - hit[:, 0, 1]) / (hit[:, 1, 1] - hit[:, 0, 1])
        x = np.sort(hit[:, 0, 0] + t * (hit[:, 1, 0] - hit[:, 0, 0]))
        for start, stop in zip(x[0::2], x[1::2]):
            first = max(0, math.ceil((start - corner[0]) / resolution - 0.5))
            last = math.floor((stop - corner[0]) / resolution - 0.5)
            grid[r, first : last + 1] = True


def _stamp(
    grid: np.ndarray,
    corner: np.ndarray,
    triangles: np.ndarray,
    band: tuple[float, float],
    resolution: float,
) -> None:
    """Mark the cells that the triangles cover between two heights."""
    for points in _samples(triangles, resolution / 2):
        points = points[(points[:, 2] >= band[0]) & (points[:, 2] <= band[1])]
        col = np.floor((points[:, 0] - corner[0]) / resolution).astype(int)
        row = np.floor((points[:, 1] - corner[1]) / resolution).astype(int)
        grid[row, col] = True


def _frame(triangles: np.ndarray, resolution: float):
    """The lower corner and the size in cells of a grid that holds the triangles."""
    flat = triangles[:, :, :2].reshape(-1, 2)
    corner = np.floor(flat.min(axis=0) / resolution) * resolution - resolution
    cells = np.ceil((flat.max(axis=0) - corner) / resolution).astype(int) + 2
    return corner, cells


def _solid(
    grid: np.ndarray,
    corner: np.ndarray,
    triangles: np.ndarray,
    band: tuple[float, float],
    resolution: float,
) -> None:
    """Mark on a grid all that a closed piece fills between two heights.

    The piece is cut at three heights and filled by the even-odd rule, so what is
    inside it is solid and a hole through it stays open.
    """
    own, size = _frame(triangles, resolution)
    local = np.zeros((size[1], size[0]), dtype=bool)
    _stamp(local, own, triangles, band, resolution)
    for share in (0.1, 0.5, 0.9):
        cut = _cut(triangles, band[0] + (band[1] - band[0]) * share)
        _inside(local, own, cut, resolution)
    col, row = np.rint((own - corner) / resolution).astype(int)
    end_row = min(row + size[1], grid.shape[0])
    end_col = min(col + size[0], grid.shape[1])
    grid[row:end_row, col:end_col] |= local[: end_row - row, : end_col - col]


def slice_shapes(
    shapes: list[tuple[np.ndarray, bool]],
    band: tuple[float, float],
    resolution: float,
    *,
    largest_hole: float,
    narrowest: float,
) -> tuple[list[dict], int]:
    """The obstacles that meshes make between two heights, seen from above.

    Each region of the cut is one polygon with its stairs smoothed away, or a box if
    it is a rectangle; a region that cannot be a polygon is cut into boxes. The
    tables are those of a world file, in world metres.

    Args:
        shapes: the triangles of each piece in world metres, shape (N, 3, 3), and
            whether the piece is closed. A closed piece is a solid, so what lies inside
            it is filled, however large, and a hole through it stays open; an open one
            is only what its surfaces cross.
        band: the lowest and the highest height that count.
        resolution: side of a grid cell in metres; thinner walls than this are lost.
        largest_hole: area in square metres up to which a shut-in gap between open
            pieces is filled, so a shelf drawn as four walls is solid.
        narrowest: width in metres below which such a gap is filled whatever its
            area, so the space between the two faces of a wall is solid; a room is
            larger and wider than both limits, and stays free.

    Returns:
        The obstacle tables, and how many specks `regions.vectorize` dropped.

    Raises:
        ValueError: if the meshes are too wide for a grid of this resolution.
    """
    kept = []
    for triangles, closed in shapes:
        z = triangles[:, :, 2]
        near = triangles[(z.min(axis=1) <= band[1]) & (z.max(axis=1) >= band[0])]
        if len(near):
            kept.append((near, closed))
    if not kept:
        return [], 0
    corner, cells = _frame(np.concatenate([t for t, _ in kept]), resolution)
    if int(cells[0]) * int(cells[1]) > MAX_CELLS:
        raise ValueError(
            f"{cells[0] * resolution:.0f} by {cells[1] * resolution:.0f} m is too "
            f"large for cells of {resolution} m"
        )
    grid = np.zeros((cells[1], cells[0]), dtype=bool)
    loose = [t for t, closed in kept if not closed]
    if loose:
        _stamp(grid, corner, np.concatenate(loose), band, resolution)
        fill_holes(
            grid,
            math.ceil(largest_hole / resolution**2),
            math.ceil(narrowest / 2 / resolution),
        )
    for triangles, closed in kept:
        if closed:
            _solid(grid, corner, triangles, band, resolution)
    return vectorize(grid, corner, resolution)
