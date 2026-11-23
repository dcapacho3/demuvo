"""The maze style: a lattice of square cells, joined by passages through the walls.

The world starts solid. A grid of cells, each as wide as a corridor, is laid in the
middle of it with a wall between neighbours. A walk that never goes back on itself
opens one passage after another until every cell has been reached, which leaves
exactly one route between any two cells. A share of the walls that are left is then
opened too, and each gives a second route: the share is `loops`, and at 0 the maze
has no loop at all.

The free space is one piece by construction, since the walk reaches every cell.
The check of `connectivity` still runs, because a passage wide enough to draw is not
always wide enough to drive, and because the walls are traced from cells.

Every length comes from the robot the map is drawn for, as in the rooms style: a
corridor is its body plus three cells of slack on each side.
"""

# The call of draw_world below reads like the other styles', because every style
# hands the same loop the same things; that is the point of sharing it.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..plugins import check_positive
from ..verify import WorldError
from ..world import World
from .carve import carve_box, carve_lane, cell_at, cells_of
from .connectivity import Clearance, clearance_for, lane
from .draw import draw_world
from .result import Generated

# What this style calls its obstacles, which are the walls it leaves standing.
PREFIX = "wall"
PIECES = ("polygons", "boxes")
# The four ways out of a cell, as (rows, columns).
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# A maze turns at right angles every cell, and a robot that follows a path cuts each
# corner. Three cells of slack on each side of its body, and not the one the other
# styles leave, kept the demo robot off the walls in every maze that was tried.
SLACK_CELLS = 3


@dataclass(frozen=True)
class MazeStyle:
    """How the maze style lays out a maze.

    Attributes:
        resolution: side of a cell of the grid in metres. The walls and the
            corridors are whole numbers of these.
        corridor: how wide a corridor is in metres, or None to take it from the
            robot the map is drawn for: its body across, and three cells of slack
            on each side, since a maze turns at every cell. A corridor narrower
            than the robot is refused.
        wall: how thick the walls are in metres, outer wall included.
        loops: the share of the inner walls that are opened after the maze is
            made, from 0 to 1. At 0 there is one route between any two places.
        pieces: "polygons" to trace each wall as one polygon, or "boxes" to tile
            the walls with axis-aligned boxes.
    """

    resolution: float = 0.1
    corridor: float | None = None
    wall: float = 0.2
    loops: float = 0.0
    pieces: str = "polygons"

    def __post_init__(self) -> None:
        """Check the settings."""
        check_positive(self, "resolution", "wall")
        if self.corridor is not None:
            check_positive(self, "corridor")
        if not 0.0 <= self.loops <= 1.0:
            raise ValueError(f"loops must be from 0 to 1, got {self.loops!r}")
        if self.pieces not in PIECES:
            raise ValueError(
                f"pieces must be one of {', '.join(PIECES)}, got {self.pieces!r}"
            )


def maze_corridor(style: MazeStyle, reach: float) -> float:
    """How wide a corridor is: what the style says, or the robot and its slack.

    Args:
        style: the settings; its `corridor` wins if it names one.
        reach: how far from a wall the robot's centre must stay, in metres.
    """
    if style.corridor is not None:
        return style.corridor
    return lane(reach, style.resolution, SLACK_CELLS)


def _walk(rng, shape: tuple[int, int]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    """The passages of a maze that has one route between any two cells.

    The walk starts at a random cell and always moves to a random cell next to it
    that it has not seen, going back along its own steps when there is none.
    """
    seen = np.zeros(shape, dtype=bool)
    start = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
    seen[start] = True
    stack, passages = [start], set()
    while stack:
        here = stack[-1]
        ahead = [
            (here[0] + dr, here[1] + dc)
            for dr, dc in STEPS
            if 0 <= here[0] + dr < shape[0]
            and 0 <= here[1] + dc < shape[1]
            and not seen[here[0] + dr, here[1] + dc]
        ]
        if not ahead:
            stack.pop()
            continue
        there = ahead[int(rng.integers(len(ahead)))]
        seen[there] = True
        passages.add((min(here, there), max(here, there)))
        stack.append(there)
    return passages


def _loops(rng, shape: tuple[int, int], passages: set, share: float) -> None:
    """Open a share of the walls that are left, in the same order every time."""
    for row in range(shape[0]):
        for col in range(shape[1]):
            for there in ((row + 1, col), (row, col + 1)):
                if there[0] >= shape[0] or there[1] >= shape[1]:
                    continue
                edge = ((row, col), there)
                if edge not in passages and rng.random() < share:
                    passages.add(edge)


def _nearest(place: tuple[int, int], middles) -> tuple[int, int]:
    """The middle of the maze cell nearest a cell of the grid."""
    return min(middles, key=lambda m: (m[0] - place[0]) ** 2 + (m[1] - place[1]) ** 2)


def maze(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    rng,
    rows: int,
    cols: int,
    style: MazeStyle,
    reach: float,
    origin: tuple[float, float] = (0.0, 0.0),
    anchors: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """The walls of a maze as blocked cells; see the module docstring.

    Args:
        rng: the random stream the maze comes from.
        rows: cells along y.
        cols: cells along x.
        style: how the maze is laid out.
        reach: how far from a wall the robot's centre must stay, in metres, which
            is what the corridors are sized by.
        origin: world position (x, y) of the corner of cell (0, 0), for the anchors.
        anchors: points that must be standable, each joined to the nearest maze
            cell by a corridor of its own, so a robot never starts inside a wall.

    Returns:
        True where a wall is. Everything is a wall but the maze cells, the
        passages between them and the way in to each anchor. The maze is centred,
        and the margin it leaves is wall.
    """
    wall = cells_of(style.wall, style.resolution)
    corridor = cells_of(maze_corridor(style, reach), style.resolution)
    pitch = corridor + wall
    mask = np.ones((rows, cols), dtype=bool)
    inside = (wall, wall, rows - wall, cols - wall)
    shape = ((rows - 2 * wall + wall) // pitch, (cols - 2 * wall + wall) // pitch)
    if shape[0] < 1 or shape[1] < 1:
        return mask  # not one cell fits; the check will say so
    first = (
        wall + (rows - 2 * wall - (shape[0] * pitch - wall)) // 2,
        wall + (cols - 2 * wall - (shape[1] * pitch - wall)) // 2,
    )

    def corner(cell):
        return first[0] + cell[0] * pitch, first[1] + cell[1] * pitch

    passages = _walk(rng, shape)
    _loops(rng, shape, passages, style.loops)
    for row in range(shape[0]):
        for col in range(shape[1]):
            r, c = corner((row, col))
            carve_box(mask, (r, c, r + corridor, c + corridor), inside)
    for one, two in passages:
        r, c = corner(one)
        r2, c2 = corner(two)
        carve_box(mask, (r, c, r2 + corridor, c2 + corridor), inside)
    middles = [
        (corner((row, col))[0] + corridor // 2, corner((row, col))[1] + corridor // 2)
        for row in range(shape[0])
        for col in range(shape[1])
    ]
    for point in anchors:
        place = cell_at(point, origin, style.resolution)
        carve_lane(
            mask,
            place,
            _nearest(place, middles),
            corridor,
            inside,
            bool(rng.integers(2)),
        )
    return mask


def generate_maze(  # pylint: disable=too-many-arguments
    size: tuple[float, float],
    seed: int,
    style: MazeStyle = MazeStyle(),
    clearance: Clearance | None = Clearance(),
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    base: World | None = None,
    anchors: tuple[tuple[float, float], ...] = (),
    attempts: int = 20,
) -> Generated:
    """Draw a maze from a seed; see the module docstring.

    Two calls with equal arguments give equal worlds, wall for wall. Every random
    number comes from `stream(seed, "generate", "maze", attempt)`, so the maze does
    not depend on what was drawn before it.

    The corridors are as wide as the robot needs, which the `clearance` says or the
    robots of the `base` world do. Nothing is drawn for a robot nobody named.

    Args:
        size: width along x and height along y in metres; the base world's, if
            there is one.
        seed: the seed of the maze, and the seed of the world that comes back.
        style: how the maze is laid out.
        clearance: what the free space must be like, and the robot it must fit. A
            clearance with no radius of its own takes one from the robots of the
            `base` world. None asks for no check at all: the first maze is kept,
            and its report is measured for a point robot.
        origin: where the lower-left corner of the world is; the base world's, if
            there is one.
        base: a world to put the maze in. Its obstacles, robots, tasks, actors and
            settings are kept, its robots and goals are anchors, and the world that
            comes back is it with the walls after its own obstacles and `seed` as
            its seed.
        anchors: more points that must be free and in the largest free region.
        attempts: how many mazes to draw before giving up.

    Returns:
        The world and the grid it was drawn from; see `Generated`. A maze has no
        continuous field behind it, so `Generated.field` is None.

    Raises:
        GenerateError: if no maze passed the check in `attempts` tries.
        WorldError: if the base world does not fit the size, origin or names, if
            nothing says how big the robot is, or if the corridors asked for are
            too narrow for it.
        ValueError: for a setting that cannot be, before any work is done.
    """
    reach = clearance_for(clearance, base).reach
    width = maze_corridor(style, reach)
    if width <= 2.0 * reach:
        raise WorldError(
            f"maze: a corridor of {width:.2f} m has no room for a robot that "
            f"needs {reach:.2f} m on each side of its centre; ask for a wider "
            f"corridor, or leave it to the robot"
        )

    def mask_of(rng, rows, cols, measure, points):
        """One maze, laid out for the robot being measured for."""
        return maze(rng, rows, cols, style, measure.reach, origin, points), None

    return draw_world(
        "maze",
        size,
        seed,
        mask_of=mask_of,
        resolution=style.resolution,
        thin=max(1, min(2, round(style.wall / style.resolution))),
        pieces=style.pieces,
        prefix=PREFIX,
        origin=origin,
        base=base,
        anchors=anchors,
        attempts=attempts,
        clearance=clearance,
        advice="Give the maze a thinner wall, a narrower corridor, or a smaller robot.",
    )
