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
from functools import partial

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
# An edge this close to an axis, as a share of its length, is made to lie on it.
SQUARE_SLOPE = 0.05
# A corner sharper than this cosine of the angle between its edges (30 degrees) has
# its tip cut off, by this many cells along each edge at most.
SPIKE_COS = 0.866
SPIKE_CUT = 3.0
# A loop with no edge shorter than this many cells is a wall and is traced exactly.
SHORT_WALL = 2
# A traced outline may cover this share of its region's cells again as cells that
# belong to no region, plus a few; one that strays further is traced at a looser
# tolerance, or tiled with boxes.
STRAY_SHARE = 0.06
STRAY_CELLS = 6
# A traced outline may hold this many cells of other regions, which is a sliver
# of overlap at an edge and not a body lying across another.
FOREIGN_CELLS = 2
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


def _keep(points: np.ndarray, tolerance: float) -> list[int]:
    """The places of a closed loop of points that stay when it is cut into edges."""
    far = int(np.argmax(np.hypot(*(points - points[0]).T)))
    one = _chain(points[: far + 1], tolerance)
    two = _chain(np.vstack([points[far:], points[:1]]), tolerance)
    keep = sorted({*one, *(far + i for i in two[:-1]), *([] if two[-1] else [0])})
    return [k for k in keep if k < len(points)]


def _corners(loop, tolerance: float) -> np.ndarray:
    """The corners of a closed loop, with the staircase of its cells straightened.

    The loop is cut into straight edges, a line is fitted to the points of each run,
    and each corner is where two neighbouring lines meet, so a diagonal wall is one
    straight edge and a sharp corner stays sharp.
    """
    points = np.array(loop, dtype=float)
    keep = _keep(points, tolerance)
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
    return _blunt(_square_up(pts))


def _root(parent: list[int], i: int) -> int:
    """The first of the places joined to place `i`, by path halving."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _square_up(pts: list) -> np.ndarray:
    """Make every edge that is nearly level or nearly upright exactly so.

    The fitted lines leave a wall meant to be straight a few hundredths of a metre
    off true. The corners at the ends of a run of such edges take their mean, so a
    long wall stays one wall and its neighbours move with it.
    """
    out = np.array(pts, dtype=float)
    count = len(out)
    for axis in (0, 1):  # 0 - level edges share y, 1 - upright edges share x
        along, across = (0, 1) if axis == 0 else (1, 0)
        parent = list(range(count))
        true = [False] * count  # on an edge that already lies on the axis

        for i in range(count):
            j = (i + 1) % count
            step = np.abs(out[j] - out[i])
            if step[along] > 0 and step[across] <= SQUARE_SLOPE * step[along]:
                parent[_root(parent, i)] = _root(parent, j)
                true[i] = true[j] = true[i] or step[across] < 1e-9
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(count):
            groups[_root(parent, i)].append(i)
        for members in groups.values():
            if len(members) > 1:
                # A line that is straight already, such as the cut that splits a
                # region round a hole, keeps its place so both sides of it agree.
                fixed = [m for m in members if true[m]] or members
                # The wall is cells, so it lies on the cell boundary nearest its fit.
                out[members, across] = round(float(out[fixed, across].mean()))
    return out


def _blunt(pts: np.ndarray) -> np.ndarray:
    """Cut the tip off every corner sharper than 30 degrees that points outwards.

    A tip this thin is an accident of the noise and no robot reaches into it. The
    cut only takes away from the obstacle, so free space never shrinks.
    """
    count = len(pts)
    area = np.sum(
        pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]
    )
    sign = 1.0 if area > 0 else -1.0
    out = []
    for i in range(count):
        here = pts[i]
        back, ahead = pts[i - 1] - here, pts[(i + 1) % count] - here
        lb, la = np.hypot(*back), np.hypot(*ahead)
        if lb == 0 or la == 0:
            out.append(here)
            continue
        cos = float(back @ ahead) / (lb * la)
        turn = sign * (ahead[0] * back[1] - ahead[1] * back[0])
        if cos > SPIKE_COS and turn > 0:
            cut = min(SPIKE_CUT, 0.4 * lb, 0.4 * la)
            out += [here + back / lb * cut, here + ahead / la * cut]
        else:
            out.append(here)
    return np.array(out)


def _exact(loop) -> np.ndarray | None:
    """The corners of a loop that is long straight edges only, or None.

    A loop of cell edges is a wall of a building when no two short edges (under two
    cells) lie within one edge of each other: a lone step is a nick, while a staircase
    is a slope.
    A wall is kept as it is, since fitting lines to it could only move a corner off
    its cell.
    """
    points = np.array(loop, dtype=float)
    ahead, back = (
        np.roll(points, -1, axis=0) - points,
        points - np.roll(points, 1, axis=0),
    )
    corners = points[np.abs(ahead[:, 0] * back[:, 1] - ahead[:, 1] * back[:, 0]) > 0]
    steps = np.hypot(*(np.roll(corners, -1, axis=0) - corners).T)
    short = steps < SHORT_WALL
    near = (
        np.roll(short, 1) | np.roll(short, -1) | np.roll(short, 2) | np.roll(short, -2)
    )
    if len(corners) < 4 or (short & near).any():
        return None
    return corners


def _plain(loop, tolerance: float) -> np.ndarray:
    """The corners of a loop with the staircase cut by tolerance and no line fitting.

    The corners stay on the cells' own outline, so a region that wraps round a thin
    neck is not pulled into touching itself the way fitted lines can.
    """
    points = np.array(loop, dtype=float)
    return points[_keep(points, tolerance)]


def _traces(loop):
    """The outlines to try for a loop, best first; each is made only when asked for."""
    exact = _exact(loop)
    if exact is not None:
        yield exact
    yield _corners(loop, 1.5)
    for tolerance in (1.0, 0.6):  # finer, for a shape the first cut strayed from
        yield _plain(loop, tolerance)
    for tolerance in (3.0, 6.0):  # coarser, for one with too many corners
        yield _corners(loop, tolerance)


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
    # The angle moved by a half turn at most, so the first side is still the length.
    length, width = high - low
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


def _cells_in(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows and columns of the cells with their centre inside an outline."""
    low = np.floor(corners.min(axis=0)).astype(int)
    high = np.ceil(corners.max(axis=0)).astype(int)
    xs, ys = np.meshgrid(
        np.arange(low[0], high[0]) + 0.5, np.arange(low[1], high[1]) + 0.5
    )
    covered = np.zeros(xs.shape, dtype=bool)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        if a[1] == b[1]:
            continue
        crosses = (a[1] > ys) != (b[1] > ys)
        at = a[0] + (ys - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
        covered ^= crosses & (xs < at)
    return ys[covered].astype(int), xs[covered].astype(int)


def _strays(corners: np.ndarray, crop: np.ndarray, r0: int, c0: int) -> int:
    """How many cells whose centre lies in the outline are two or more from the region.

    The outline is in cells, the region cropped at row `r0` and column `c0`. A cell
    next to the region is the region's own edge, which a straight outline cannot
    follow cell for cell.
    """
    wide = np.pad(crop, 1)
    near = np.zeros_like(wide)
    for dr in range(3):
        for dc in range(3):
            near |= np.roll(np.roll(wide, dr - 1, axis=0), dc - 1, axis=1)
    rows, cols = _cells_in(corners)
    rows, cols = rows - (r0 - 1), cols - (c0 - 1)
    held = (rows >= 0) & (rows < near.shape[0]) & (cols >= 0) & (cols < near.shape[1])
    return int(len(rows) - near[rows[held], cols[held]].sum())


def _holds(cells: np.ndarray, *, crop, r0, c0, limit, others) -> bool:
    """True if an outline stays near its region and off the cells of other regions."""
    return (
        _strays(cells, crop, r0, c0) <= limit
        and _foreign(cells, others) <= FOREIGN_CELLS
    )


def _foreign(corners: np.ndarray, others: np.ndarray) -> int:
    """How many cells of other regions have their centre inside the outline."""
    rows, cols = _cells_in(corners)
    held = (
        (rows >= 0) & (rows < others.shape[0]) & (cols >= 0) & (cols < others.shape[1])
    )
    return int(others[rows[held], cols[held]].sum())


def _shape(
    corners: np.ndarray, corner, resolution: float, bounds=None, *, accept=None
) -> dict | None:
    """The obstacle table of a loop's corners: a box if it is one, else a polygon.

    Returns None if the corners do not make a simple polygon, or if neither the box
    nor the polygon stays inside the bounds. Straightening the staircase of a region
    moves a corner outwards, and a region cut off flat along the edge of its grid
    can send one a long way out, so a caller that has an edge to keep gives its
    bounds and gets boxes for that region instead. A caller that gives `accept`, a
    test of the outline in cells, has a shape that fails it refused the same way.
    """
    world = corners * resolution + corner
    if bounds is not None:  # a corner pushed past the edge by straightening lies on it
        world = np.clip(world, bounds[:2], bounds[2:])
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
        around = (_outline(table) - corner) / resolution
        if fits(table, bounds) and (accept is None or accept(around)):
            return table
    vertices: list[list[float]] = []
    for x, y in world:
        point = [round(float(x), 3), round(float(y), 3)]
        if not vertices or point != vertices[-1]:
            vertices.append(point)
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    if len(vertices) < 3 or len(vertices) > MAX_CORNERS:
        return None
    try:
        Polygon.parse({"vertices": vertices}, "mesh")
    except WorldError:
        return None
    table = {"shape": {"name": "polygon", "vertices": vertices}}
    if accept is not None and not accept((np.array(vertices) - corner) / resolution):
        return None
    return table if fits(table, bounds) else None


def _regions(mask: np.ndarray, cuts: int = 0, family: int | None = None):
    """The regions of a mask that have no hole, as (cropped mask, loop, family).

    A region with a hole is cut in two through the hole, again until none is left; if
    that does not work the loop is None. The family is the number of the region of
    the whole mask that a piece was cut from, so pieces that were cut apart know each
    other.
    """
    for number, (crop, r0, c0) in enumerate(components(mask)):
        kin = number if family is None else family
        loops = _loops(crop)
        if len(loops) == 1:
            yield crop, r0, c0, loops[0], kin
            continue
        hole = min(loops, key=_area2)
        if cuts >= MAX_CUTS:
            yield crop, r0, c0, None, kin
            continue
        middle = int(np.mean([x for x, _ in hole]))
        left, right = crop.copy(), crop.copy()
        left[:, middle:] = False
        right[:, :middle] = False
        for part in (left, right):
            for sub, sr, sc, loop, _ in _regions(part, cuts + 1, kin):
                yield sub, r0 + sr, c0 + sc, loop, kin


def _outline(table: dict) -> np.ndarray:
    """The corners, in world metres, of an obstacle table that is a box or polygon."""
    shape = parse_shape(table["shape"], "region")
    pose = (
        Transform2D.from_xyt(table["pose"])
        if "pose" in table
        else Transform2D.identity()
    )
    return np.array(shape.points(pose), dtype=float)


def _nearest(point: np.ndarray, outline: np.ndarray) -> tuple[float, np.ndarray]:
    """The distance from a point to a closed outline, and the nearest point of it."""
    start, step = outline, np.roll(outline, -1, axis=0) - outline
    length = np.einsum("ij,ij->i", step, step)
    along = np.clip(
        np.einsum("ij,ij->i", point - start, step) / np.where(length == 0, 1, length),
        0,
        1,
    )
    near = start + along[:, None] * step
    gap = np.hypot(*(near - point).T)
    best = int(np.argmin(gap))
    return float(gap[best]), near[best]


def _weld(tables: list[dict], kin: list[int], reach: float) -> None:
    """Move the corners of pieces cut from one region onto the edges beside them.

    Pieces cut apart along a hole meet along the cut, and each was traced on its own,
    so the two sides of the cut can end a few centimetres off one another: a step
    in the wall, or a sliver where they overlap. A corner within `reach` of a
    sister piece's edge, and not on it, is put on it. A move that would spoil the
    polygon is not made.
    """
    for i, table in enumerate(tables):
        if table["shape"]["name"] != "polygon":
            continue
        sisters = [
            _outline(tables[j])
            for j in range(len(tables))
            if j != i and kin[j] == kin[i]
        ]
        vertices = np.array(table["shape"]["vertices"], dtype=float)
        moved = vertices.copy()
        for k, vertex in enumerate(vertices):
            found = [_nearest(vertex, outline) for outline in sisters]
            gap, point = min(found, key=lambda f: f[0], default=(math.inf, vertex))
            if 1e-6 < gap <= reach:
                moved[k] = point
        if np.array_equal(moved, vertices):
            continue
        listed = [[round(float(x), 3), round(float(y), 3)] for x, y in moved]
        try:
            Polygon.parse({"vertices": listed}, "mesh")
        except WorldError:
            continue
        table["shape"]["vertices"] = listed


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
    kin: list[int] = []
    for crop, r0, c0, loop, family in _regions(grid):
        if int(crop.sum()) < min_cells:
            dropped += 1
            continue
        found = None
        if loop is not None and pieces != "boxes":
            limit = STRAY_SHARE * crop.sum() + STRAY_CELLS

            others = grid.copy()
            others[r0 : r0 + crop.shape[0], c0 : c0 + crop.shape[1]] &= ~crop

            near = partial(_holds, crop=crop, r0=r0, c0=c0, limit=limit, others=others)

            for traced in _traces(loop):
                found = _shape(
                    traced + (c0, r0), corner, resolution, bounds, accept=near
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
        kin += [family] * (len(shapes) - len(kin))
    _weld(shapes, kin, 0.8 * resolution)
    return shapes, dropped


def drop_thin(mask: np.ndarray, width: int) -> None:
    """Clear every part of the mask that is narrower than `width` cells.

    The mask is changed in place. A line of cells one cell wide is a trace of the
    grid and no wall; what stands wider than `width` keeps its shape.
    """
    if width < 2:
        return
    solid = mask.copy()
    for _ in range(width - 1):
        solid = solid[:-1, :-1] & solid[1:, :-1] & solid[:-1, 1:] & solid[1:, 1:]
    kept = np.zeros_like(mask)
    for dr in range(width):
        for dc in range(width):
            kept[dr : dr + solid.shape[0], dc : dc + solid.shape[1]] |= solid
    mask &= kept


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
