"""The rooms-and-corridors style: a building cut into rooms, joined by corridors.

The world starts solid. A rectangle inside the outer wall is cut in two, again and
again, each cut along the longer side of what it cuts, until a piece is too small to
cut or the depth runs out. Every piece that is left is a leaf, and a room is carved
out of it, set in from its edges by the wall and by a random margin, so the rooms are
of different sizes and are not in a row. Each cut then has a corridor carved along
it, from a room on one side to a room on the other.

The free space is one piece by construction and not by luck: the cuts make a tree,
each corridor joins the two subtrees of one cut, and so every room reaches every
other. The check of `connectivity` still has to pass, because carving a corridor
wide enough to draw is not the same as carving one wide enough to drive, and because
the obstacles the walls become are traced from cells.

Nothing here is measured in metres that a robot could be measured in instead. The
corridors, the doors they make and the smallest room all come from the robot the map
is drawn for: a corridor is its body plus a cell of slack on each side. Give the
generator a bigger robot and the building opens up around it.
"""

# The call of draw_world below reads like the field style's, because both styles
# hand the same loop the same things; that is the point of sharing it.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..plugins import check_not_negative, check_positive
from ..verify import WorldError
from ..world import World
from .carve import carve_box, carve_lane, cell_at, cells_of
from .connectivity import Clearance, clearance_for, lane
from .draw import draw_world
from .result import Generated

# What this style calls its obstacles, which are the walls it leaves standing.
PREFIX = "wall"
PIECES = ("polygons", "boxes")
# The smallest room is this many corridors across. A room the width of a corridor is
# a corridor; at three, a robot can pass another and still have a room.
ROOM_CORRIDORS = 3


@dataclass(frozen=True)
class RoomsStyle:
    """How the rooms-and-corridors style lays out a building.

    Attributes:
        resolution: side of a cell in metres. The walls, the rooms and the
            corridors are whole numbers of cells, so this is the grid everything
            is laid out on.
        corridor: how wide a corridor is in metres, or None to take it from the
            robot the map is drawn for: its body across, and a cell of slack on
            each side. A corridor narrower than the robot is refused, since no
            map of it could pass the check.
        min_room: the shortest side a room may have in metres, or None for three
            corridors. It decides how finely the building can be cut: a piece
            that cannot hold two of these side by side is not cut again.
        wall: how thick the walls are in metres, outer wall included. It is a
            thickness, not a clearance, so it is not taken from the robot.
        depth: how many times the building may be cut in two. At most 2**depth
            rooms come of it, fewer where a piece was too small to cut.
        pieces: "polygons" to trace each wall as one polygon, or "boxes" to tile
            the walls with axis-aligned boxes. The walls of a building are one
            piece with the rooms as holes in it, which a polygon cannot be, so
            most of a building is tiled either way; see `world.regions`.
    """

    resolution: float = 0.1
    corridor: float | None = None
    min_room: float | None = None
    wall: float = 0.2
    depth: int = 4
    pieces: str = "polygons"

    def __post_init__(self) -> None:
        """Check the settings."""
        check_positive(self, "resolution", "wall")
        for name in ("corridor", "min_room"):
            if getattr(self, name) is not None:
                check_positive(self, name)
        if not isinstance(self.depth, int) or isinstance(self.depth, bool):
            raise ValueError(f"depth must be a whole number, got {self.depth!r}")
        check_not_negative(self, "depth")
        if self.pieces not in PIECES:
            raise ValueError(
                f"pieces must be one of {', '.join(PIECES)}, got {self.pieces!r}"
            )


def corridor_width(style: RoomsStyle, reach: float) -> float:
    """How wide a corridor is: what the style says, or the robot and its slack.

    Args:
        style: the settings; its `corridor` wins if it names one.
        reach: how far from a wall the robot's centre must stay, in metres.
    """
    if style.corridor is not None:
        return style.corridor
    return lane(reach, style.resolution)


@dataclass(frozen=True)
class _Plan:
    """The lengths of a building in cells, worked out once from the style.

    Attributes:
        wall: thickness of a wall.
        corridor: width of a corridor and of the doors it makes.
        room: the shortest side a room may have.
        least: the smallest piece the cutting will leave, a room and its walls.
    """

    wall: int
    corridor: int
    room: int
    least: int


def _plan(style: RoomsStyle, reach: float) -> _Plan:
    """The lengths of the building in cells; see `_Plan`."""
    resolution = style.resolution
    wall = cells_of(style.wall, resolution)
    corridor = cells_of(corridor_width(style, reach), resolution)
    smallest = (
        ROOM_CORRIDORS * corridor * resolution
        if style.min_room is None
        else style.min_room
    )
    room = max(cells_of(smallest, resolution), corridor)
    return _Plan(wall, corridor, room, room + 2 * wall)


def _cut(rng, rect: tuple[int, int, int, int], depth: int, least: int):
    """Cut a rectangle of cells in two until it is too small, and join the pieces.

    Each cut goes across the longer side, at a place that leaves both halves big
    enough to hold a room. The corridors come back as pairs of room places, one
    from each side of a cut, which is what makes the free space one piece: the
    cuts are a tree and each corridor joins the two subtrees of one cut.

    Args:
        rng: the random stream.
        rect: (first row, first column, end row, end column) of the piece.
        depth: how many more times this piece may be cut.
        least: the smallest piece a cut may leave.

    Returns:
        The leaves of this piece, and the pairs of leaves a corridor joins.
    """
    r0, c0, r1, c1 = rect
    tall, wide = r1 - r0, c1 - c0
    if depth <= 0 or max(tall, wide) < 2 * least:
        return [rect], []
    if tall >= wide:
        at = int(rng.integers(r0 + least, r1 - least + 1))
        halves = ((r0, c0, at, c1), (at, c0, r1, c1))
    else:
        at = int(rng.integers(c0 + least, c1 - least + 1))
        halves = ((r0, c0, r1, at), (r0, at, r1, c1))
    first, links = _cut(rng, halves[0], depth - 1, least)
    second, more = _cut(rng, halves[1], depth - 1, least)
    links = links + more
    here = (
        first[int(rng.integers(len(first)))],
        second[int(rng.integers(len(second)))],
    )
    return first + second, links + [here]


def _room(rng, leaf: tuple[int, int, int, int], plan: _Plan):
    """A room inside a leaf: set in by the wall, then by a random margin.

    The margin never takes the room below the smallest the plan allows, so every
    room holds the robot the building was laid out for.
    """
    r0, c0, r1, c1 = leaf
    r0, c0, r1, c1 = r0 + plan.wall, c0 + plan.wall, r1 - plan.wall, c1 - plan.wall
    spare = (max(r1 - r0 - plan.room, 0), max(c1 - c0 - plan.room, 0))
    take = [int(rng.integers(0, s + 1)) for s in spare]
    before = [int(rng.integers(0, t + 1)) for t in take]
    return (
        r0 + before[0],
        c0 + before[1],
        r1 - (take[0] - before[0]),
        c1 - (take[1] - before[1]),
    )


def _middle(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    """The cell at the middle of a rectangle."""
    r0, c0, r1, c1 = rect
    return (r0 + r1) // 2, (c0 + c1) // 2


def _nearest(place: tuple[int, int], rooms) -> tuple[int, int]:
    """The middle of the room whose middle is nearest a cell."""
    return min(
        (_middle(room) for room in rooms),
        key=lambda m: (m[0] - place[0]) ** 2 + (m[1] - place[1]) ** 2,
    )


def building(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rng,
    rows: int,
    cols: int,
    style: RoomsStyle,
    reach: float,
    origin: tuple[float, float] = (0.0, 0.0),
    anchors: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """The walls of a building as blocked cells; see the module docstring.

    Args:
        rng: the random stream the cuts and the rooms come from.
        rows: cells along y.
        cols: cells along x.
        style: how the building is laid out.
        reach: how far from a wall the robot's centre must stay, in metres, which
            is what the corridors are sized by.
        origin: world position (x, y) of the corner of cell (0, 0), for the anchors.
        anchors: points that must be standable, each joined to the nearest room by
            a corridor of its own, so a robot never starts inside a wall.

    Returns:
        True where a wall is. Everything is a wall but the rooms, the corridors
        and the way in to each anchor.
    """
    plan = _plan(style, reach)
    mask = np.ones((rows, cols), dtype=bool)
    inside = (plan.wall, plan.wall, rows - plan.wall, cols - plan.wall)
    if inside[0] >= inside[2] or inside[1] >= inside[3]:
        return mask  # the walls alone fill the world; the check will say so
    leaves, links = _cut(rng, inside, style.depth, plan.least)
    rooms = [_room(rng, leaf, plan) for leaf in leaves]
    places = dict(zip(leaves, rooms))
    for room in rooms:
        carve_box(mask, room, inside)
    for one, two in links:
        carve_lane(
            mask,
            _middle(places[one]),
            _middle(places[two]),
            plan.corridor,
            inside,
            bool(rng.integers(2)),
        )
    for point in anchors:
        place = cell_at(point, origin, style.resolution)
        carve_lane(
            mask,
            place,
            _nearest(place, rooms),
            plan.corridor,
            inside,
            bool(rng.integers(2)),
        )
    return mask


def generate_rooms(  # pylint: disable=too-many-arguments
    size: tuple[float, float],
    seed: int,
    style: RoomsStyle = RoomsStyle(),
    clearance: Clearance | None = Clearance(),
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    base: World | None = None,
    anchors: tuple[tuple[float, float], ...] = (),
    attempts: int = 20,
) -> Generated:
    """Draw a building of rooms and corridors from a seed; see the module docstring.

    Two calls with equal arguments give equal worlds, wall for wall. Every random
    number comes from `stream(seed, "generate", "rooms", attempt)`, so the building
    does not depend on what was drawn before it.

    The corridors are as wide as the robot needs, which the `clearance` says or the
    robots of the `base` world do. Nothing is drawn for a robot nobody named.

    Args:
        size: width along x and height along y in metres; the base world's, if
            there is one.
        seed: the seed of the building, and the seed of the world that comes back.
        style: how the building is laid out.
        clearance: what the free space must be like, and the robot it must fit. A
            clearance with no radius of its own takes one from the robots of the
            `base` world. None asks for no check at all: the first building is
            kept, and its report is measured for a point robot.
        origin: where the lower-left corner of the world is; the base world's, if
            there is one.
        base: a world to put the building in. Its obstacles, robots, tasks, actors
            and settings are kept, its robots and goals are anchors, and the world
            that comes back is it with the walls after its own obstacles and
            `seed` as its seed.
        anchors: more points that must be free and in the largest free region.
        attempts: how many buildings to draw before giving up.

    Returns:
        The world and the grid it was drawn from; see `Generated`. A building has
        no continuous field behind it, so `Generated.field` is None.

    Raises:
        GenerateError: if no building passed the check in `attempts` tries.
        WorldError: if the base world does not fit the size, origin or names, if
            nothing says how big the robot is, or if the corridors asked for are
            too narrow for it.
        ValueError: for a setting that cannot be, before any work is done.
    """
    reach = clearance_for(clearance, base).reach
    width = corridor_width(style, reach)
    if width <= 2.0 * reach:
        raise WorldError(
            f"rooms: a corridor of {width:.2f} m has no room for a robot that "
            f"needs {reach:.2f} m on each side of its centre; ask for a wider "
            f"corridor, or leave it to the robot"
        )

    def mask_of(rng, rows, cols, measure, points):
        """One building, laid out for the robot being measured for."""
        return building(rng, rows, cols, style, measure.reach, origin, points), None

    return draw_world(
        "rooms",
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
        advice="Give the building a smaller depth, a thinner wall, or a smaller robot.",
    )
