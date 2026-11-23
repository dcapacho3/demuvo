"""The caves style: noise worn down into caverns, and one cave kept out of them.

Every cell starts as rock or as air, by a toss of its own, and then the majority
rule is run over the grid a few times: a cell takes the state of most of the nine
cells around it. Single cells and ragged edges disappear in the first pass, and what
is left after four grows into round caverns with winding ways between them, which is
what cellular automata are used for. The edge of the world is rock, so a cave is
shut in.

A cell of that grid is one lane of the robot wide, and this is what makes the style
work. Tossed on the grid the map is drawn at, the rule gives caverns joined by
cracks: measured, 64% of such a world is open and only 22% of it is ground this
robot could stand on, in 153 patches it could not pass between. Tossed on a grid
whose cell is the robot's lane, the same rule gives one cave in one patch, because
the narrowest thing it can draw is a way the robot fits through. The cave is worn
at the robot's scale and then drawn at the map's, with a last pass of the rule over
the fine cells to rub the corners off the blocks.

What is left over is still filtered, since a rounded corner can pinch and the
cracks of the first toss are not the only way to strand a cavern. The cells the
robot's centre may stand on are found the way the costmap finds them, the largest
patch of those is the cave, and everything its body cannot cover from there is
filled in. What is left is one cave for this robot, which is the same promise the
rooms style keeps by construction and the field style keeps by drawing again.

Nothing here is measured in metres that a robot could be measured in instead. How
much room a cave must have to be a cave, how wide a tunnel to an anchor is, and what
counts as reachable at all, all come from the robot the map is drawn for. The only
lengths a person sets are the grid, the rock round the edge, and how much of the
world starts as rock.
"""

# The call of draw_world below reads like the other styles', because every style
# hands the same loop the same things; that is the point of sharing it.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..plugins import check_not_negative, check_positive, check_share
from ..world import World
from .carve import carve_lane, cell_at, cells_of, fill_border, majority
from .connectivity import Clearance, lane, numbered, within
from .draw import draw_world
from .result import Generated

# What this style calls its obstacles, which are the rock it leaves standing.
PREFIX = "rock"
PIECES = ("polygons", "boxes")


@dataclass(frozen=True)
class CaveStyle:
    """How the caves style wears a world down.

    Attributes:
        resolution: side of a cell in metres. It is the grain of the rock: the
            caverns and the ways between them are whole numbers of cells, and a
            finer grid gives a more ragged wall.
        fill: the share of cave cells that are rock before the first pass of
            the majority rule, 0 to 1. A cave cell is one lane of the robot
            across, so this is a share and not a length, and it means the same
            whatever the robot. What comes out is not this share: the rule moves
            it, and filling in what the robot cannot reach moves it further.
            Measured over eight seeds, 0.45 leaves about three quarters of the
            world open, 0.5 about three fifths, and 0.55 about a third but as
            little as a fifth on a seed, which is where the check starts
            refusing them.
        smooth: how many times the majority rule is run over the cave cells. One
            pass takes the noise out, four leaves round caverns, more rounds
            them further.
        border: metres of solid rock round the edge of the world, so a cave is
            shut in. At least one cell of it, whatever this says.
        pieces: "polygons" to trace each body of rock as one polygon, or "boxes"
            to tile them with axis-aligned boxes. Rock worn into caves is rarely
            a rectangle, so polygons are the shape of it.
    """

    resolution: float = 0.1
    fill: float = 0.5
    smooth: int = 4
    border: float = 0.2
    pieces: str = "polygons"

    def __post_init__(self) -> None:
        """Check the settings."""
        check_positive(self, "resolution", "border")
        check_share(self, "fill")
        if not isinstance(self.smooth, int) or isinstance(self.smooth, bool):
            raise ValueError(f"smooth must be a whole number, got {self.smooth!r}")
        check_not_negative(self, "smooth")
        if self.pieces not in PIECES:
            raise ValueError(
                f"pieces must be one of {', '.join(PIECES)}, got {self.pieces!r}"
            )


def one_cave(mask: np.ndarray, resolution: float, reach: float) -> np.ndarray:
    """Fill in everything but the one cave a robot of this reach can move about in.

    The cells the robot's centre may stand on are the cells not within its reach of
    rock, which is how the costmap decides it too. The largest patch of those is the
    cave; its body can cover whatever lies within its reach of that patch, and the
    rest is rock, whether it was before or not. A world where the robot can stand
    nowhere fills in entirely, and the check will say so in plain numbers.

    Args:
        mask: the rock, changed in place.
        resolution: side of a cell in metres.
        reach: how far from rock the robot's centre must stay, in metres.

    Returns:
        The cells the robot's centre may stand on in the cave that was kept.
    """
    stand = ~within(mask, resolution, reach)
    labels, sizes = numbered(stand)
    main = labels == 0 if sizes else np.zeros(mask.shape, dtype=bool)
    mask |= ~within(main, resolution, reach)
    return main


def _tunnel(
    rng, mask: np.ndarray, place: tuple[int, int], main: np.ndarray, width: int
) -> None:
    """Cut a tunnel from a place to the nearest cell of the cave, in place.

    This is how an anchor joins the cave: a robot of the base world is not left
    inside the rock because the dice put a cavern somewhere else.
    """
    standing = np.argwhere(main)
    away = (standing[:, 0] - place[0]) ** 2 + (standing[:, 1] - place[1]) ** 2
    nearest = standing[int(np.argmin(away))]
    rows, cols = mask.shape
    carve_lane(
        mask,
        place,
        (int(nearest[0]), int(nearest[1])),
        width,
        (0, 0, rows, cols),
        bool(rng.integers(2)),
    )


def cavern(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rng,
    rows: int,
    cols: int,
    style: CaveStyle,
    reach: float,
    origin: tuple[float, float] = (0.0, 0.0),
    anchors: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """The rock of a cave world as blocked cells; see the module docstring.

    Args:
        rng: the random stream the first toss of each cell comes from.
        rows: cells along y.
        cols: cells along x.
        style: how the world is worn down.
        reach: how far from rock the robot's centre must stay, in metres, which is
            what the cave is kept and the tunnels are cut by.
        origin: world position (x, y) of the corner of cell (0, 0), for the anchors.
        anchors: points that must be standable, each joined to the cave by a tunnel
            of its own.

    Returns:
        True where rock is.
    """
    resolution = style.resolution
    rings = cells_of(style.border, resolution)
    # one cave cell is one lane of the robot, so the narrowest thing the rule can
    # draw is a way the robot fits through
    step = cells_of(lane(reach, resolution), resolution)
    rock = rng.random((-(-rows // step), -(-cols // step))) < style.fill
    for _ in range(style.smooth):
        rock = majority(rock)
    mask = np.repeat(np.repeat(rock, step, axis=0), step, axis=1)[:rows, :cols]
    mask = majority(mask)  # rub the corners off the blocks
    fill_border(mask, rings)
    main = one_cave(mask, resolution, reach)
    if main.any():
        for point in anchors:
            _tunnel(rng, mask, cell_at(point, origin, resolution), main, step)
        fill_border(mask, rings)
    return mask


def generate_caves(  # pylint: disable=too-many-arguments
    size: tuple[float, float],
    seed: int,
    style: CaveStyle = CaveStyle(),
    clearance: Clearance | None = Clearance(),
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    base: World | None = None,
    anchors: tuple[tuple[float, float], ...] = (),
    attempts: int = 20,
) -> Generated:
    """Wear a world of caves out of noise from a seed; see the module docstring.

    Two calls with equal arguments give equal worlds, rock for rock. Every random
    number comes from `stream(seed, "generate", "caves", attempt)`, so the cave does
    not depend on what was drawn before it.

    The cave that is kept is the one this robot can move about in, which the
    `clearance` says or the robots of the `base` world do. Nothing is drawn for a
    robot nobody named.

    Args:
        size: width along x and height along y in metres; the base world's, if
            there is one.
        seed: the seed of the cave, and the seed of the world that comes back.
        style: how the world is worn down.
        clearance: what the free space must be like, and the robot it must fit. A
            clearance with no radius of its own takes one from the robots of the
            `base` world. None asks for no check at all: the first cave is kept,
            and its report is measured for a point robot.
        origin: where the lower-left corner of the world is; the base world's, if
            there is one.
        base: a world to wear the caves into. Its obstacles, robots, tasks, actors
            and settings are kept, its robots and goals are anchors, and the world
            that comes back is it with the rock after its own obstacles and `seed`
            as its seed.
        anchors: more points that must be free and in the largest free region.
        attempts: how many caves to wear before giving up.

    Returns:
        The world and the grid it was drawn from; see `Generated`. The first toss
        of each cell is noise and not a field to be cut, so `Generated.field` is
        None.

    Raises:
        GenerateError: if no cave passed the check in `attempts` tries.
        WorldError: if the base world does not fit the size, origin or names, or
            if nothing says how big the robot is.
        ValueError: for a setting that cannot be, before any work is done.
    """

    def mask_of(rng, rows, cols, measure, points):
        """One cave, worn down and kept for the robot being measured for."""
        return cavern(rng, rows, cols, style, measure.reach, origin, points), None

    return draw_world(
        "caves",
        size,
        seed,
        mask_of=mask_of,
        resolution=style.resolution,
        pieces=style.pieces,
        prefix=PREFIX,
        origin=origin,
        base=base,
        anchors=anchors,
        attempts=attempts,
        clearance=clearance,
        advice="Lower fill, smooth fewer times, or ask for less clearance.",
    )
