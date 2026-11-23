"""Editing a grid of cells: what the styles do before the cells become shapes.

A style draws by setting cells. Clearing a ring, filling one, clearing a disc round
a place a robot must stand, cutting a lane from here to there, rounding an outline
with the majority rule: all three styles want these, so they are written once. Every
one of them changes the grid in place, except the majority rule, which cannot.

A cell is True where the ground is blocked. Row 0 is the lowest y and column 0 the
lowest x, as everywhere else in the package.
"""

from __future__ import annotations

import math

import numpy as np


def clear_border(mask: np.ndarray, rings: int) -> None:
    """Clear the outermost rings of cells, so the edge of the world is free."""
    mask[:rings, :] = False
    mask[-rings:, :] = False
    mask[:, :rings] = False
    mask[:, -rings:] = False


def fill_border(mask: np.ndarray, rings: int) -> None:
    """Block the outermost rings of cells, so the world is shut in by its edge."""
    mask[:rings, :] = True
    mask[-rings:, :] = True
    mask[:, :rings] = True
    mask[:, -rings:] = True


def clear_disc(
    mask: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    point: tuple[float, float],
    radius: float,
) -> None:
    """Clear every cell whose centre is within a radius of a point in the world."""
    rows, cols = mask.shape
    first_row = max(math.floor((point[1] - radius - origin[1]) / resolution), 0)
    last_row = min(math.ceil((point[1] + radius - origin[1]) / resolution), rows)
    first_col = max(math.floor((point[0] - radius - origin[0]) / resolution), 0)
    last_col = min(math.ceil((point[0] + radius - origin[0]) / resolution), cols)
    if first_row >= last_row or first_col >= last_col:
        return
    y = origin[1] + (np.arange(first_row, last_row) + 0.5) * resolution
    x = origin[0] + (np.arange(first_col, last_col) + 0.5) * resolution
    near = (x[None, :] - point[0]) ** 2 + (y[:, None] - point[1]) ** 2 <= radius**2
    window = mask[first_row:last_row, first_col:last_col]
    mask[first_row:last_row, first_col:last_col] = window & ~near


def carve_box(mask: np.ndarray, rect, inside: tuple[int, int, int, int]) -> None:
    """Free the cells of a rectangle of cells, clipped to what is inside the edge."""
    r0 = max(rect[0], inside[0])
    c0 = max(rect[1], inside[1])
    r1 = min(rect[2], inside[2])
    c1 = min(rect[3], inside[3])
    if r0 < r1 and c0 < c1:
        mask[r0:r1, c0:c1] = False


def carve_lane(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    inside: tuple[int, int, int, int],
    along_rows: bool,
) -> None:
    """Free an L of two lanes of this width, from one cell to another.

    One lane runs along the rows and the other along the columns; `along_rows`
    says which comes first, which is what turns the corner one way or the other.
    """
    low, high = width // 2, width - width // 2
    (r0, c0), (r1, c1) = start, end
    turn = (r0, c1) if along_rows else (r1, c0)
    for (ra, ca), (rb, cb) in ((start, turn), (turn, end)):
        carve_box(
            mask,
            (
                min(ra, rb) - low,
                min(ca, cb) - low,
                max(ra, rb) + high,
                max(ca, cb) + high,
            ),
            inside,
        )


def majority(mask: np.ndarray) -> np.ndarray:
    """One pass of the 3 by 3 majority rule; a cell off the grid counts as free.

    A cell takes the state of most of the nine cells around it, itself included.
    It rounds an outline, closes a one-cell gap and rubs out a one-cell neck, and
    run enough times over random cells it is what turns noise into caves.
    """
    padded = np.pad(mask, 1).astype(np.int8)
    rows, cols = mask.shape
    count = sum(padded[r : r + rows, c : c + cols] for r in range(3) for c in range(3))
    return count >= 5


def cells_of(metres: float, resolution: float) -> int:
    """How many whole cells a length covers, at least one."""
    return max(1, math.ceil(round(metres / resolution, 9)))


def cell_at(
    point: tuple[float, float], origin: tuple[float, float], resolution: float
) -> tuple[int, int]:
    """The (row, column) of the cell a world point lies in."""
    return (
        math.floor(round((point[1] - origin[1]) / resolution, 9)),
        math.floor(round((point[0] - origin[0]) / resolution, 9)),
    )
