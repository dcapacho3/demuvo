"""The field style: a smooth random field, cut at a level, becomes blobs of clutter.

A field `f(x, y)` is drawn over the world from lattices of random values, each
lattice half as coarse as the one before. Where `f` is above a cut the ground is
blocked, below it the ground is free, so the world fills with rounded blobs about
`scale` metres across with free channels winding between them. The cut is a quantile
of the field, so `fill` blocks the share of the world it says, on every seed, and not
a level that would mean something different on each one.

What the cut leaves is cleaned before it becomes shapes, in this order: a ring along
the border is cleared, the outlines are rounded, the specks go, the pockets a robot
could never enter are filled, and a disc round every anchor is cleared. Then
`world.regions` traces each blob, and the free space is measured by the shared check.
A field that the check refuses is thrown away and the next one is drawn, so a map that
comes back always has the fill that was asked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..plugins import check_not_negative, check_positive, check_share
from ..world import World
from ..world.regions import drop_small, fill_holes
from .carve import clear_border, clear_disc, majority
from .connectivity import Clearance, room
from .draw import draw_world
from .result import Generated
from .shapes import DEFAULT_PREFIX

# The rings of cells along the border that are always free, so no vertex of a traced
# blob can land on or beyond the edge of the world, which the loader refuses.
BORDER_CELLS = 2
PIECES = ("polygons", "boxes")


@dataclass(frozen=True)
class FieldStyle:
    """How the field style draws its clutter.

    Attributes:
        resolution: side of a field cell in metres. Smaller gives rounder outlines
            and more corners to each polygon.
        scale: the size of the largest features in metres, the spacing of the
            coarsest lattice. Blobs come out about this wide.
        octaves: how many finer lattices are added on top, each at half the spacing
            of the one before. 1 gives smooth ovals, 4 gives ragged coastlines.
        persistence: the weight of each octave against the one before, 0 to 1.
        fill: the share of the world's cells the cut blocks, 0 to 1, before any
            cleaning. 0 gives an empty world.
        smooth: passes of the 3 by 3 majority rule after the cut. Each one rounds
            the outline and takes away the necks of a single cell.
        min_area: blobs of fewer square metres than this are dropped. A speck is a
            shape no planner cares about and a cost to the lidar at every step.
        min_pocket: free pockets of fewer square metres than this, shut in by
            blocked cells, are filled. A pocket a robot cannot enter would only be
            a second free region that fails the check for no reason. None asks for
            the room the robot itself needs, so what counts as a pocket follows
            the robot the map is drawn for and is not a number written here.
        border: metres along the edge of the world kept free. Two field cells are
            always kept free, whatever this says, so every vertex lies inside the
            world.
        pieces: "polygons" to trace each blob as one polygon (a box when it is
            one), or "boxes" to tile every blob with axis-aligned boxes. Polygons
            give a handful of obstacles and exact outlines; boxes give many more
            obstacles but only boxes.
    """

    resolution: float = 0.1
    scale: float = 2.0
    octaves: int = 3
    persistence: float = 0.5
    fill: float = 0.35
    smooth: int = 1
    min_area: float = 0.05
    min_pocket: float | None = None
    border: float = 0.0
    pieces: str = "polygons"

    def __post_init__(self) -> None:
        """Check the settings."""
        check_positive(self, "resolution", "scale")
        check_share(self, "persistence", "fill")
        check_not_negative(self, "min_area", "border")
        if self.min_pocket is not None:
            check_not_negative(self, "min_pocket")
        for name in ("octaves", "smooth"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a whole number, got {value!r}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be 1 or more, got {self.octaves!r}")
        if self.pieces not in PIECES:
            raise ValueError(
                f"pieces must be one of {', '.join(PIECES)}, got {self.pieces!r}"
            )


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """The weight that eases from 0 to 1 with no kink at either end."""
    return t * t * (3.0 - 2.0 * t)


def _lattice(rng, x: np.ndarray, y: np.ndarray, spacing: float) -> np.ndarray:
    """One octave: random values on a lattice, eased out to the cells between them.

    The lattice starts at the world's origin, so the field does not depend on where
    the border happens to fall between two lattice points.
    """
    fx, fy = x / spacing, y / spacing
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    values = rng.uniform(-1.0, 1.0, (int(iy[-1]) + 2, int(ix[-1]) + 2))
    tx = _smoothstep(fx - ix)[None, :]
    ty = _smoothstep(fy - iy)[:, None]
    lower = values[np.ix_(iy, ix)] * (1 - tx) + values[np.ix_(iy, ix + 1)] * tx
    upper = values[np.ix_(iy + 1, ix)] * (1 - tx) + values[np.ix_(iy + 1, ix + 1)] * tx
    return lower * (1 - ty) + upper * ty


def field_of(rng, rows: int, cols: int, style: FieldStyle) -> np.ndarray:
    """The field over the cells of a grid, from `octaves` lattices, as -1 to 1.

    Args:
        rng: the random stream; it draws one lattice for each octave.
        rows: cells along y.
        cols: cells along x.
        style: the settings; `scale`, `octaves` and `persistence` are used.
    """
    x = (np.arange(cols) + 0.5) * style.resolution
    y = (np.arange(rows) + 0.5) * style.resolution
    found = np.zeros((rows, cols))
    weight, total = 1.0, 0.0
    for octave in range(style.octaves):
        found += weight * _lattice(rng, x, y, style.scale / 2**octave)
        total += weight
        weight *= style.persistence
    return found / total


def cut_mask(
    values: np.ndarray,
    style: FieldStyle,
    origin: tuple[float, float],
    anchors: tuple[tuple[float, float], ...] = (),
    reach: float = 0.0,
) -> np.ndarray:
    """Cut a field at its `fill` quantile and clean what is left; see the docstring.

    Args:
        values: the field, row 0 at the lowest y.
        style: the settings of the cut and the cleaning.
        origin: world position (x, y) of the corner of cell (0, 0).
        anchors: points a robot must be able to stand on, which are cleared.
        reach: how far from an obstacle the robot's centre must stay, in metres;
            the cleared disc round an anchor is this and two cells wider. The
            outline traced round a blob is straightened, which moves its corners
            out beyond the cells they came from, so a disc of just the reach can
            leave a robot standing with millimetres to spare. The two cells pay
            for that. A caller that wants real elbow room sets `Clearance.margin`,
            which the check enforces and this follows.

    Returns:
        The blocked cells.
    """
    resolution = style.resolution
    if style.fill <= 0:
        mask = np.zeros(values.shape, dtype=bool)
    else:
        mask = values >= np.quantile(values, 1.0 - style.fill)
    # The border is cleared before the blobs are smoothed, so that a blob the edge
    # cut off flat has its sliver of a cut edge rounded away like any other, and
    # again after each pass, since a pass can push a blob back into the ring. The
    # anchors are cleared last: the disc round one is small enough that filling the
    # pockets would take it back if it were cleared any earlier.
    rings = max(BORDER_CELLS, math.ceil(round(style.border / resolution, 9)))
    clear_border(mask, rings)
    for _ in range(style.smooth):
        mask = majority(mask)
        clear_border(mask, rings)
    pocket = room(reach) if style.min_pocket is None else style.min_pocket
    drop_small(mask, math.ceil(round(style.min_area / resolution**2, 9)))
    fill_holes(mask, math.ceil(round(pocket / resolution**2, 9)), 0)
    for point in anchors:
        clear_disc(mask, origin, resolution, point, reach + 2 * resolution)
    return mask


def generate_field(  # pylint: disable=too-many-arguments
    size: tuple[float, float],
    seed: int,
    style: FieldStyle = FieldStyle(),
    clearance: Clearance | None = Clearance(),
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    base: World | None = None,
    anchors: tuple[tuple[float, float], ...] = (),
    attempts: int = 20,
) -> Generated:
    """Draw a world of random clutter from a seed; see the module docstring.

    Two calls with equal arguments give equal worlds, field for field and vertex
    for vertex. Every random number comes from `stream(seed, "generate", "field",
    attempt)`, so the field does not depend on what was drawn before it.

    Args:
        size: width along x and height along y in metres; the base world's, if there
            is one.
        seed: the seed of the field, and the seed of the world that comes back. The
            seeds of a run (`--seeds`) change the noise of the sensors and the
            actors, not the map, which is written down as obstacles.
        style: how the field is drawn and cut.
        clearance: what the free space must be like, and the robot it must fit.
            A clearance with no radius of its own takes one from the robots of the
            `base` world, since what is drawn is cleaned and measured by the robot
            that will drive it; with no robot either, the call is refused rather
            than drawn for a guessed body. None asks for no check at all: the first
            field is kept whatever its free space is like, and its report is
            measured for a point robot, so a caller can still see what it got.
        origin: where the lower-left corner of the world is; the base world's, if
            there is one.
        base: a world to add the clutter to. Its obstacles, robots, tasks, actors
            and settings are kept, its robots and goals are anchors, and the world
            that comes back is it with the new obstacles after its own and `seed`
            as its seed.
        anchors: more points that must be free and in the largest free region, such
            as a start and a goal a caller has in mind but has not placed yet.
        attempts: how many fields to draw before giving up. Each one is a new
            stream; the settings never change, so a map that comes back has the
            fill that was asked for.

    Returns:
        The world and the grids it was drawn from; see `Generated`.

    Raises:
        GenerateError: if no field passed the check in `attempts` tries.
        WorldError: if the base world does not fit the size, origin or names, or
            if nothing says how big the robot is.
        ValueError: for a setting that cannot be, before any work is done.
    """

    def mask_of(rng, rows, cols, measure, points):
        """One field, cut and cleaned for the robot being measured for."""
        values = field_of(rng, rows, cols, style)
        return cut_mask(values, style, origin, points, measure.reach), values

    return draw_world(
        "field",
        size,
        seed,
        mask_of=mask_of,
        resolution=style.resolution,
        pieces=style.pieces,
        prefix=DEFAULT_PREFIX,
        origin=origin,
        base=base,
        anchors=anchors,
        attempts=attempts,
        clearance=clearance,
        advice="Lower fill, raise scale, or ask for less clearance.",
    )
