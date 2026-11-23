"""Grids of marked cells, and the obstacles they make.

A boolean grid comes from a mesh cut at a robot's height, from a map file, or from a
generated field: the cells it marks are blocked, the others free. This module is the
one way from such a grid to shapes a world file accepts. The cells are joined into
regions, small shut-in gaps are filled, each region's outline is traced and its
staircase straightened, and the outline becomes a box when it is a rectangle and a
polygon otherwise; a region no polygon can hold is tiled with boxes.

Row 0 of every grid is the lowest y and column 0 the lowest x, as in `Costmap` and
`MapFile`. A region is a set of cells joined along their edges, never at a corner
alone: two cells that touch only at a corner would pinch a polygon to a point, and
no polygon of a world file may do that.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from robomath import Transform2D

from ..verify import WorldError, inside
from .shapes import Polygon, parse_shape

# A region of fewer cells than this is noise, and is dropped.
MIN_CELLS = 4
# The most times a region with a hole is cut in two to take the hole out.
MAX_CUTS = 12
# An edge of fewer cells than this, between two longer ones, is a chamfer.
SHORT_EDGE = 10
# The most corners of a polygon; a region that needs more is cut into boxes.
MAX_CORNERS = 120


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    """The (first, end) columns of each run of True in a row of a grid."""
    padded = np.concatenate([[False], row, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


def label_regions(
    mask: np.ndarray,
) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
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
        for first, end in runs(mask[r]):
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


def fill_holes(grid: np.ndarray, largest: int, narrow: int) -> None:
    """Mark every shut-in free region that is small or narrow.

    A region is small if it has at most `largest` cells, and narrow if no cell of it
    is more than `narrow` cells from the edge of the region. A region that reaches
    the edge of the grid is the outside, and a large wide one is a room; neither is
    touched.
    """
    rows, cols = grid.shape
    found, region = label_regions(~grid)
    size: dict[int, int] = {}
    edge: set[int] = set()
    for r, row in enumerate(found):
        for first, end, i in row:
            top = region[i]
            size[top] = size.get(top, 0) + end - first
            if r in (0, rows - 1) or first == 0 or end == cols:
                edge.add(top)
    wide = eroded(~grid, narrow)
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


def eroded(free: np.ndarray, steps: int) -> np.ndarray:
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


def boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Rectangles (first column, end column, first row, end row) that cover a grid.

    A run of marked cells in a row joins the run above it when they match exactly.
    """
    rows = grid.shape[0]
    open_runs: dict[tuple[int, int], int] = {}
    found = []
    for r in range(rows + 1):
        row_runs = runs(grid[r]) if r < rows else []
        for key in [k for k in open_runs if k not in row_runs]:
            found.append((key[0], key[1], open_runs.pop(key), r))
        for key in row_runs:
            open_runs.setdefault(key, r)
    return found


def components(mask: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """The regions of a mask, each cropped, with the row and column of its corner."""
    found, region = label_regions(mask)
    cells: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for r, row in enumerate(found):
        for first, end, i in row:
            cells[region[i]].append((r, first, end))
    out = []
    for region_runs in cells.values():
        r0, r1 = region_runs[0][0], region_runs[-1][0] + 1
        c0, c1 = min(c for _, c, _ in region_runs), max(e for _, _, e in region_runs)
        crop = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for r, first, end in region_runs:
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

    The loop is cut into straight edges, a line is fitted to the points of each run,
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
    edges = []
    for i, start in enumerate(keep):
        stop = keep[(i + 1) % len(keep)]
        run = (
            points[start : stop + 1]
            if stop > start
            else np.vstack([points[start:], points[: stop + 1]])
        )
        edges.append(_line(run[1:-1] if len(run) > 4 else run))
    pts = [corners[i] for i in range(len(keep))]
    for i in range(len(keep)):
        meet = _meet(edges[i - 1], edges[i], corners[i], 2 * tolerance + 2)
        if meet is not None:
            pts[i] = meet
    # An edge of a few cells between two long ones is a chamfer from the cells: the
    # long edges are extended to meet, and the two corners become one.
    i = 0
    while len(pts) > 3 and i < len(pts):
        j = (i + 1) % len(pts)
        short = np.hypot(*(pts[j] - pts[i])) < SHORT_EDGE
        meet = _meet(
            edges[i - 1],
            edges[(i + 1) % len(edges)],
            pts[i],
            SHORT_EDGE + 2 * tolerance,
        )
        if short and meet is not None:
            pts[i] = meet
            del pts[j]
            del edges[i]
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
    filled = abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2
    if area == 0 or filled < 0.92 * area:
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


def fits(table: dict, bounds) -> bool:
    """True if a world file would take this obstacle table, inside the bounds.

    The table is read back exactly as the loader reads it, rounding and all, so
    this asks the question the loader will ask of the saved file and no other one.
    A shape it refuses is no shape: a region thin enough that its box rounds to a
    side of zero comes through here, and the caller falls back to boxes of cells.
    With no `bounds` only the shape itself is in question, not where it lies.
    """
    try:
        shape = parse_shape(table["shape"], "region")
    except WorldError:
        return False
    if bounds is None:
        return True
    pose = (
        Transform2D.from_xyt(table["pose"])
        if "pose" in table
        else Transform2D.identity()
    )
    return inside(shape.points(pose), bounds)


def _shape(corners: np.ndarray, corner, resolution: float, bounds=None) -> dict | None:
    """The obstacle table of a loop's corners: a box if it is one, else a polygon.

    Returns None if the corners do not make a simple polygon, or if neither the box
    nor the polygon stays inside the bounds. Straightening the staircase of a region
    moves a corner outwards, and a region cut off flat along the edge of its grid
    can send one a long way out, so a caller that has an edge to keep gives its
    bounds and gets boxes for that region instead.
    """
    world = corners * resolution + corner
    box = _turned_box(world, 1.5 * resolution)
    if box is not None:
        x, y, turn, length, width = box
        table = {
            "pose": [
                round(x, 3),
                round(y, 3),
                round(0.0 if abs(turn) < 1e-3 else turn, 4),
            ],
            "shape": {"name": "box", "size": [round(length, 3), round(width, 3)]},
        }
        if fits(table, bounds):
            return table
    vertices = [[round(float(x), 3), round(float(y), 3)] for x, y in world]
    if len(vertices) < 3 or len(vertices) > MAX_CORNERS:
        return None
    try:
        Polygon.parse({"vertices": vertices}, "mesh")
    except WorldError:
        return None
    table = {"shape": {"name": "polygon", "vertices": vertices}}
    return table if fits(table, bounds) else None


def _regions(mask: np.ndarray, cuts: int = 0):
    """The regions of a mask that have no hole, as (cropped mask, loop).

    A region with a hole is cut in two through the hole, again until none is left; if
    that does not work the loop is None.
    """
    for crop, r0, c0 in components(mask):
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


def vectorize(
    grid: np.ndarray,
    corner,
    resolution: float,
    *,
    min_cells: int = MIN_CELLS,
    pieces: str = "polygons",
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list[dict], int]:
    """The obstacles of a grid of marked cells, and how many specks were dropped.

    Each region becomes one polygon with the staircase of its cells straightened, or
    a box when it is a rectangle; a region a polygon cannot hold, one with a hole
    that will not come apart, is tiled with boxes. The tables are those of a world
    file, in world metres, with every number rounded to three decimals.

    Args:
        grid: the marked cells, row 0 at the lowest y.
        corner: the world position (x, y) of the corner of cell (0, 0).
        resolution: side of a cell in metres.
        min_cells: a region of fewer cells than this is a speck, and is dropped. A
            caller that has cleaned its grid already passes 1 to keep every region.
        pieces: "polygons" to trace each region's outline, or "boxes" to tile every
            region with axis-aligned boxes and trace none.
        bounds: (x_min, y_min, x_max, y_max) that every shape must stay inside, or
            None for no edge to keep. A region whose traced outline would reach
            past them is tiled with boxes, which are cells and cannot.

    Returns:
        The obstacle tables, in row-major order of each region's first cell, and how
        many specks were dropped.
    """
    shapes: list[dict] = []
    dropped = 0
    for crop, r0, c0, loop in _regions(grid):
        if int(crop.sum()) < min_cells:
            dropped += 1
            continue
        found = None
        if loop is not None and pieces != "boxes":
            for tolerance in (1.5, 3.0, 6.0):
                found = _shape(
                    _corners(loop, tolerance) + (c0, r0), corner, resolution, bounds
                )
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
                for a, b, t, u in boxes(crop)
            ]
        else:
            shapes.append(found)
    return shapes, dropped


def drop_small(mask: np.ndarray, smallest: int) -> int:
    """Clear every region of the mask with fewer than `smallest` cells.

    The mask is changed in place. This is the mirror of `fill_holes`, which fills
    the small shut-in gaps between the marked cells; here the specks themselves go.

    Returns:
        How many regions were cleared.
    """
    found, region = label_regions(mask)
    size: dict[int, int] = {}
    for row in found:
        for first, end, i in row:
            size[region[i]] = size.get(region[i], 0) + end - first
    gone = {root for root, cells in size.items() if cells < smallest}
    for r, row in enumerate(found):
        for first, end, i in row:
            if region[i] in gone:
                mask[r, first:end] = False
    return len(gone)
