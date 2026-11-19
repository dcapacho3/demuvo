"""A mesh seen from above, cut to the heights a robot meets.

The triangles are sampled finely, the samples between two heights are marked on a grid,
small enclosed gaps (the inside of a shelf) are filled, and the marked cells become
boxes. A roof, a ceiling or a floor lies outside the heights, so it adds nothing.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..verify import WorldError
from .shapes import Polygon

# The most samples held in memory at once.
CHUNK = 2_000_000
# The most steps, as a power of two, along an edge when sampling a triangle.
MAX_POWER = 11
# A region of fewer cells than this is noise, and is dropped.
MIN_CELLS = 4
# The most times a region with a hole is cut in two to take the hole out.
MAX_CUTS = 12
# An edge of fewer cells than this, between two longer ones, is a chamfer.
SHORT_EDGE = 10
# The most corners of a polygon; a region that needs more is cut into boxes.
MAX_CORNERS = 120
# The most cells a grid may have; a bigger world is refused, not guessed at.
MAX_CELLS = 40_000_000


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """The (first, end) columns of each run of True in a row of a grid."""
    padded = np.concatenate([[False], row, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


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


def _label(mask: np.ndarray) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
    """Join the runs of True cells that touch along an edge into regions.

    Returns:
        The runs of each row as (first, end, run number), and the region of each run
        number: the number of the first run that reaches it.
    """
    rows = mask.shape[0]
    parent: list[int] = []
    found: list[list[tuple[int, int, int]]] = []
    for r in range(rows):
        row = []
        for first, end in _runs(mask[r]):
            parent.append(len(parent))
            row.append((first, end, len(parent) - 1))
        found.append(row)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(1, rows):
        above, here = found[r - 1], found[r]
        i = j = 0
        while i < len(above) and j < len(here):
            if above[i][0] < here[j][1] and here[j][0] < above[i][1]:
                parent[root(here[j][2])] = root(above[i][2])
            if above[i][1] < here[j][1]:
                i += 1
            else:
                j += 1
    return found, [root(i) for i in range(len(parent))]


def _fill_holes(grid: np.ndarray, largest: int, narrow: int) -> None:
    """Mark every shut-in free region that is small or narrow.

    A region is small if it has at most `largest` cells, and narrow if no cell of it
    is more than `narrow` cells from the edge of the region. A region that reaches
    the edge of the grid is the outside, and a large wide one is a room; neither is
    touched.
    """
    rows, cols = grid.shape
    found, region = _label(~grid)
    size: dict[int, int] = {}
    edge: set[int] = set()
    for r, row in enumerate(found):
        for first, end, i in row:
            top = region[i]
            size[top] = size.get(top, 0) + end - first
            if r in (0, rows - 1) or first == 0 or end == cols:
                edge.add(top)
    wide = _eroded(~grid, narrow)
    thick = set()
    for r, row in enumerate(found):
        for first, end, i in row:
            if wide[r, first:end].any():
                thick.add(region[i])
    for r, row in enumerate(found):
        for first, end, i in row:
            top = region[i]
            if top not in edge and (size[top] <= largest or top not in thick):
                grid[r, first:end] = True


def _eroded(free: np.ndarray, steps: int) -> np.ndarray:
    """The cells of a mask that are more than `steps` cells from outside it."""
    kept = free.copy()
    for _ in range(steps):
        inner = kept.copy()
        inner[1:, :] &= kept[:-1, :]
        inner[:-1, :] &= kept[1:, :]
        inner[:, 1:] &= kept[:, :-1]
        inner[:, :-1] &= kept[:, 1:]
        inner[0, :] = inner[-1, :] = False
        inner[:, 0] = inner[:, -1] = False
        kept = inner
    return kept


def _boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Rectangles (first column, end column, first row, end row) that cover a grid.

    A run of marked cells in a row joins the run above it when they match exactly.
    """
    rows = grid.shape[0]
    open_runs: dict[tuple[int, int], int] = {}
    found = []
    for r in range(rows + 1):
        runs = _runs(grid[r]) if r < rows else []
        for key in [k for k in open_runs if k not in runs]:
            found.append((key[0], key[1], open_runs.pop(key), r))
        for key in runs:
            open_runs.setdefault(key, r)
    return found


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """The regions of a mask, each cropped, with the row and column of its corner."""
    found, region = _label(mask)
    cells: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for r, row in enumerate(found):
        for first, end, i in row:
            cells[region[i]].append((r, first, end))
    out = []
    for runs in cells.values():
        r0, r1 = runs[0][0], runs[-1][0] + 1
        c0, c1 = min(c for _, c, _ in runs), max(e for _, _, e in runs)
        crop = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for r, first, end in runs:
            crop[r - r0, first - c0 : end - c0] = True
        out.append((crop, r0, c0))
    return out


def _edges(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """The boundary edges of a mask as arrays of start x, start y, end x, end y.

    Each edge has its cell on the left of its direction.
    """
    padded = np.pad(mask, 1)
    rows, cols = np.nonzero(mask)
    found: list[tuple[np.ndarray, ...]] = []
    for dr, dc, start, end in (
        (-1, 0, (0, 0), (1, 0)),
        (0, 1, (1, 0), (1, 1)),
        (1, 0, (1, 1), (0, 1)),
        (0, -1, (0, 1), (0, 0)),
    ):
        edge = ~padded[rows + 1 + dr, cols + 1 + dc]
        c, r = cols[edge], rows[edge]
        found.append((c + start[0], r + start[1], c + end[0], r + end[1]))
    return tuple(np.concatenate(part) for part in zip(*found))


def _loops(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """The outline loops of a mask, as corner points (column, row).

    A loop has the cells on its left, so an outer edge runs counter-clockwise and a
    hole clockwise. Where two cells touch only at a corner they are kept apart.
    """
    sx, sy, ex, ey = _edges(mask)
    leaving: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(sx.tolist(), sy.tolist())):
        leaving[key].append(i)

    def left_of(heading, k: int) -> int:
        """How far an edge turns to the left of a heading."""
        return heading[0] * (ey[k] - sy[k]) - heading[1] * (ex[k] - sx[k])

    used = np.zeros(len(sx), dtype=bool)
    loops = []
    for first in range(len(sx)):
        if used[first]:
            continue
        loop, i = [], first
        while True:
            used[i] = True
            loop.append((int(sx[i]), int(sy[i])))
            heading = (ex[i] - sx[i], ey[i] - sy[i])
            options = [
                k
                for k in leaving[(int(ex[i]), int(ey[i]))]
                if not used[k] or k == first
            ]
            i = max(options, key=lambda k, h=heading: left_of(h, k))
            if i == first:
                break
        loops.append(loop)
    return loops


def _area2(loop) -> int:
    """Twice the signed area of a loop of corner points."""
    x, y = np.array(loop).T
    return int(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _chain(points: np.ndarray, tolerance: float) -> list[int]:
    """Douglas-Peucker: the places of an open chain whose corners stay in tolerance."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        chord = points[b] - points[a]
        length = np.hypot(*chord)
        between = points[a + 1 : b] - points[a]
        if length == 0:
            away = np.hypot(between[:, 0], between[:, 1])
        else:
            away = np.abs(chord[0] * between[:, 1] - chord[1] * between[:, 0]) / length
        worst = int(np.argmax(away))
        if away[worst] > tolerance:
            keep.add(a + 1 + worst)
            stack += [(a, a + 1 + worst), (a + 1 + worst, b)]
    return sorted(keep)


def _line(points: np.ndarray):
    """The centre and the direction of the straight line that best fits the points."""
    centre = points.mean(axis=0)
    direction = np.linalg.svd(points - centre)[2][0]
    return centre, direction


def _corners(loop, tolerance: float) -> np.ndarray:
    """The corners of a closed loop, with the staircase of its cells straightened.

    The loop is cut into straight runs, a line is fitted to the points of each run,
    and each corner is where two neighbouring lines meet, so a diagonal wall is one
    straight edge and a sharp corner stays sharp.
    """
    points = np.array(loop, dtype=float)
    far = int(np.argmax(np.hypot(*(points - points[0]).T)))
    one = _chain(points[: far + 1], tolerance)
    two = _chain(np.vstack([points[far:], points[:1]]), tolerance)
    keep = sorted({*one, *(far + i for i in two[:-1]), *([] if two[-1] else [0])})
    keep = [k for k in keep if k < len(points)]
    corners = points[keep]
    runs = []
    for i, start in enumerate(keep):
        stop = keep[(i + 1) % len(keep)]
        run = (
            points[start : stop + 1]
            if stop > start
            else np.vstack([points[start:], points[: stop + 1]])
        )
        runs.append(_line(run[1:-1] if len(run) > 4 else run))
    pts = [corners[i] for i in range(len(keep))]
    for i in range(len(keep)):
        meet = _meet(runs[i - 1], runs[i], corners[i], 2 * tolerance + 2)
        if meet is not None:
            pts[i] = meet
    # An edge of a few cells between two long ones is a chamfer from the cells: the
    # long edges are extended to meet, and the two corners become one.
    i = 0
    while len(pts) > 3 and i < len(pts):
        j = (i + 1) % len(pts)
        short = np.hypot(*(pts[j] - pts[i])) < SHORT_EDGE
        meet = _meet(
            runs[i - 1], runs[(i + 1) % len(runs)], pts[i], SHORT_EDGE + 2 * tolerance
        )
        if short and meet is not None:
            pts[i] = meet
            del pts[j]
            del runs[i]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.array(pts)


def _meet(first, second, near: np.ndarray, reach: float):
    """Where two fitted lines cross, if that is within `reach` of `near`; else None."""
    (c0, d0), (c1, d1) = first, second
    cross = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(cross) < 0.17:  # under about ten degrees: no corner to find
        return None
    t = ((c1 - c0)[0] * d1[1] - (c1 - c0)[1] * d1[0]) / cross
    meet = c0 + t * d0
    return meet if np.hypot(*(meet - near)) <= reach else None


def _hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of some points, counter-clockwise (Andrew's monotone chain)."""
    order = sorted(map(tuple, points.tolist()))
    if len(order) < 3:
        return np.array(order)

    def half(seq):
        chain: list[tuple[float, float]] = []
        for p in seq:
            while len(chain) >= 2 and (
                (chain[-1][0] - chain[-2][0]) * (p[1] - chain[-2][1])
                - (chain[-1][1] - chain[-2][1]) * (p[0] - chain[-2][0])
                <= 0
            ):
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return np.array(half(order) + half(order[::-1]))


def _smallest_box(hull: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """The smallest box with a side along an edge of a hull.

    Returns:
        Its area, the angle of its first side, and its lower and upper corners in
        the frame turned by that angle.
    """
    best = (math.inf, 0.0, np.zeros(2), np.zeros(2))
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(angle), math.sin(angle)
        local = hull @ np.array([[c, -s], [s, c]])
        low, high = local.min(axis=0), local.max(axis=0)
        area = float(np.prod(high - low))
        if area < best[0]:
            best = (area, angle, low, high)
    return best


def _turned_box(corners: np.ndarray, reach: float):
    """The rotated box that a polygon is nearly equal to, or None.

    The box is the smallest that holds the polygon. It stands for the polygon when
    the polygon fills at least 92 percent of it and no corner is farther than `reach`
    inside its sides.

    Returns:
        (centre x, centre y, turn, length, width) of the box, or None.
    """
    hull = _hull(corners)
    if len(hull) < 3:
        return None
    area, angle, low, high = _smallest_box(hull)
    x, y = corners[:, 0], corners[:, 1]
    inside = abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2
    if area == 0 or inside < 0.92 * area:
        return None
    c, s = math.cos(angle), math.sin(angle)
    local = corners @ np.array([[c, -s], [s, c]])
    if np.minimum(local - low, high - local).min(axis=1).max() > reach:
        return None
    middle = (low + high) / 2 @ np.array([[c, s], [-s, c]])
    turn = (
        angle + math.pi / 2
    ) % math.pi - math.pi / 2  # a box turned by pi is the same
    length, width = high - low
    if turn != angle:  # the box's sides swap when its angle is brought into range
        length, width = width, length
    return float(middle[0]), float(middle[1]), turn, float(length), float(width)


def _shape(corners: np.ndarray, corner, resolution: float) -> dict | None:
    """The obstacle table of a loop's corners: a box if it is one, else a polygon.

    Returns None if the corners do not make a simple polygon.
    """
    world = corners * resolution + corner
    box = _turned_box(world, 1.5 * resolution)
    if box is not None:
        x, y, turn, length, width = box
        return {
            "pose": [
                round(x, 3),
                round(y, 3),
                round(0.0 if abs(turn) < 1e-3 else turn, 4),
            ],
            "shape": {"name": "box", "size": [round(length, 3), round(width, 3)]},
        }
    vertices = [[round(float(x), 3), round(float(y), 3)] for x, y in world]
    if len(vertices) < 3 or len(vertices) > MAX_CORNERS:
        return None
    try:
        Polygon.parse({"vertices": vertices}, "mesh")
    except WorldError:
        return None
    return {"shape": {"name": "polygon", "vertices": vertices}}


def _regions(mask: np.ndarray, cuts: int = 0):
    """The regions of a mask that have no hole, as (cropped mask, loop).

    A region with a hole is cut in two through the hole, again until none is left; if
    that does not work the loop is None.
    """
    for crop, r0, c0 in _components(mask):
        loops = _loops(crop)
        if len(loops) == 1:
            yield crop, r0, c0, loops[0]
            continue
        hole = min(loops, key=_area2)
        if cuts >= MAX_CUTS:
            yield crop, r0, c0, None
            continue
        middle = int(np.mean([x for x, _ in hole]))
        left, right = crop.copy(), crop.copy()
        left[:, middle:] = False
        right[:, :middle] = False
        for part in (left, right):
            for sub, sr, sc, loop in _regions(part, cuts + 1):
                yield sub, r0 + sr, c0 + sc, loop


def _vectors(grid: np.ndarray, corner, resolution: float) -> tuple[list[dict], int]:
    """The obstacles of a grid of marked cells, and how many specks were dropped."""
    shapes: list[dict] = []
    dropped = 0
    for crop, r0, c0, loop in _regions(grid):
        if int(crop.sum()) < MIN_CELLS:
            dropped += 1
            continue
        found = None
        if loop is not None:
            for tolerance in (1.5, 3.0, 6.0):
                found = _shape(_corners(loop, tolerance) + (c0, r0), corner, resolution)
                if found is not None:
                    break
        if found is None:
            shapes += [
                {
                    "pose": [
                        round(corner[0] + (c0 + (a + b) / 2) * resolution, 3),
                        round(corner[1] + (r0 + (t + u) / 2) * resolution, 3),
                        0.0,
                    ],
                    "shape": {
                        "name": "box",
                        "size": [
                            round((b - a) * resolution, 3),
                            round((u - t) * resolution, 3),
                        ],
                    },
                }
                for a, b, t, u in _boxes(crop)
            ]
        else:
            shapes.append(found)
    return shapes, dropped


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
        The obstacle tables, and how many specks of fewer than `MIN_CELLS` cells were
        dropped.

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
        _fill_holes(
            grid,
            math.ceil(largest_hole / resolution**2),
            math.ceil(narrowest / 2 / resolution),
        )
    for triangles, closed in kept:
        if closed:
            _solid(grid, corner, triangles, band, resolution)
    return _vectors(grid, corner, resolution)
