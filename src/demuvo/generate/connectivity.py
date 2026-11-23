"""Is the free space of a world one piece, and are the places that matter in it?

A generator that only drew clutter could draw a wall across the world, or a map whose
start sits in a sealed room. This check refuses both, and it measures with the
planner's own ruler: the world is rasterised as `generate_static` does, the cells
within a robot's reach of an obstacle are taken away as `inflate` does, and what is
left is joined into regions along cell edges, never at a corner alone. A* with eight
neighbours takes a diagonal only when both cells beside it are free, so two free cells
that touch at a corner are no passage for it either, and "connected" here and
"plannable" there are the same thing.

Every style calls this, whatever it drew, so it lives apart from any one of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..costmap.build import inscribed_radius
from ..costmap.grid import FREE, INSCRIBED, LETHAL
from ..costmap.layers import generate_static, inflate
from ..plugins import check_not_negative, check_positive, check_share
from ..verify import WorldError
from ..world import World
from ..world.regions import label_regions


class GenerateError(WorldError):
    """No world a generator drew passed the free-space check."""


# A way through is the robot's body and this many cells of slack on each side, so
# its centre keeps a lane of its own once the walls are inflated.
SLACK_CELLS = 1


def lane(reach: float, resolution: float, slack: int = SLACK_CELLS) -> float:
    """How wide a way through must be for a robot of this reach, on this grid.

    A corridor between rooms, a tunnel into a cave and the way out of an anchor are
    all this wide, so that a robot of this reach has somewhere to put its centre
    after the costmap has inflated what is on either side.
    """
    return 2.0 * reach + 2.0 * slack * resolution


def within(mask: np.ndarray, resolution: float, reach: float) -> np.ndarray:
    """The cells no further than `reach` from a cell of the mask.

    This is the one place in the package that says how far a thing spreads over a
    grid, and it spreads it the way the costmap does, so a style that cleans with
    it is cleaning by the rule the check will hold it to. Called on the walls it
    gives the cells a robot's centre may not be in; called on the places a robot
    may stand it gives everything its body can cover from there.
    """
    if reach <= 0:
        return mask.copy()
    grid = np.where(mask, LETHAL, FREE).astype(np.uint8)
    return inflate(grid, resolution, reach, reach, 1.0) >= INSCRIBED


def room(reach: float) -> float:
    """The square metres a robot of this reach needs to stand clear of everything.

    The square round its clearance disc. It is the one measure of "big enough for
    this robot" in the package: a free region smaller than this is no room, and a
    gap smaller than this is no gap. Sizing by the robot and not by a number
    written here is why a style never says how wide a corridor is in metres.
    """
    return (2.0 * reach) ** 2


@dataclass(frozen=True)
class Clearance:
    """What the free space of a world must be like for the check to pass.

    Attributes:
        radius: the inscribed radius of the robot the map is for, in metres, or
            None to take it from the robots of the world being measured; see
            `for_robots`. A cell nearer an obstacle than this is not free, since
            the robot's centre cannot be there. 0 asks for a point robot, which
            measures the bare space of a map and promises nothing about a body.
        margin: metres added to the radius, for a map that should not be tight.
        resolution: side of a check cell in metres, or None for the world's own
            costmap resolution. A map is only as connected as the grid it is
            measured on, so this should be the resolution the planner will use.
        min_main: the least share of the free cells the largest region may hold,
            0 to 1. 0.9 lets a few small closed pockets stand and refuses a map cut
            in two.
        min_free: the least share of the world's cells that must be free for the
            robot's centre, 0 to 1. It refuses a map that is one solid block with a
            lane through it, which `min_main` alone would pass.
        min_component: free regions of fewer square metres than this are not
            counted as regions at all; a two-cell gap between two blobs is not a
            room. None asks for the room the robot itself needs, which is what
            "too small to be a region" means when a robot is what will use it.
    """

    radius: float | None = None
    margin: float = 0.0
    resolution: float | None = None
    min_main: float = 0.9
    min_free: float = 0.2
    min_component: float | None = None

    def __post_init__(self) -> None:
        """Check the settings."""
        check_not_negative(self, "margin")
        check_share(self, "min_main", "min_free")
        for name in ("radius", "resolution", "min_component"):
            if getattr(self, name) is not None:
                check_not_negative(self, name)
        if self.resolution is not None:
            check_positive(self, "resolution")

    def for_robots(self, shapes) -> Clearance:
        """This clearance with its radius taken from some robot shapes, if it had none.

        The widest robot decides: a map the car can cross, the small robot can
        cross too. A clearance that already names a radius keeps it, so a caller
        may always say what it means. With no radius and no robot there is no body
        to measure for, and the radius is 0.

        Args:
            shapes: the shapes of the robots of the world being measured.
        """
        if self.radius is not None:
            return self
        return replace(
            self, radius=max((inscribed_radius(s) for s in shapes), default=0.0)
        )

    @property
    def reach(self) -> float:
        """How far from an obstacle the robot's centre must stay, in metres.

        Raises:
            ValueError: if the radius is still None, so the clearance has not met
                its robot yet. `for_robots` is what introduces them.
        """
        if self.radius is None:
            raise ValueError(
                "this clearance has no radius yet: call for_robots with the robots "
                "of the world, or give the Clearance a radius of its own"
            )
        return self.radius + self.margin

    @property
    def smallest_region(self) -> float:
        """The square metres a free region needs before it counts as one."""
        return room(self.reach) if self.min_component is None else self.min_component


@dataclass(frozen=True)
class Connectivity:
    """What the free-space check found.

    Attributes:
        ok: True if every condition of the `Clearance` held.
        reasons: one sentence for each condition that failed; empty when ok.
        free_fraction: share of the world's cells the robot's centre may stand on.
        components: how many free regions of at least `min_component` there are.
        main_fraction: share of the free cells that lie in the largest region.
        anchors: for each anchor, in order, the number of its region, 0 for the
            largest, or None if it is blocked or off the grid.
        labels: the region number of every cell, -1 where the robot cannot stand.
            It is left out of comparisons and repr, so two reports still compare by
            their numbers. A caller that places a start and a goal reads it.
        resolution: side of a cell of `labels` in metres.
        origin: world position (x, y) of the corner of cell (0, 0) of `labels`.
    """

    ok: bool
    reasons: tuple[str, ...]
    free_fraction: float
    components: int
    main_fraction: float
    anchors: tuple[int | None, ...] = ()
    labels: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int32),
        compare=False,
        repr=False,
    )
    resolution: float = 0.05
    origin: tuple[float, float] = (0.0, 0.0)


def numbered(free: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """The region of every free cell, the largest region 0, and the region sizes.

    A blocked cell is -1. Regions of equal size keep the order of their first cell,
    so the numbering is the same on every machine and every run.
    """
    found, region = label_regions(free)
    size: dict[int, int] = {}
    for row in found:
        for first, end, i in row:
            size[region[i]] = size.get(region[i], 0) + end - first
    order = sorted(size, key=lambda root: (-size[root], root))
    number = {root: n for n, root in enumerate(order)}
    labels = np.full(free.shape, -1, dtype=np.int32)
    for r, row in enumerate(found):
        for first, end, i in row:
            labels[r, first:end] = number[region[i]]
    return labels, [size[root] for root in order]


def _region_at(labels: np.ndarray, cell: tuple[int, int]) -> int | None:
    """The region of a cell, or None if it is blocked or off the grid."""
    row, col = cell
    rows, cols = labels.shape
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    found = int(labels[row, col])
    return None if found < 0 else found


def _name(names: tuple[str, ...], i: int) -> str:
    """What to call the anchor at a place, for a message."""
    return names[i] if i < len(names) else f"anchor {i + 1}"


def check_grid(
    blocked: np.ndarray,
    resolution: float,
    clearance: Clearance,
    anchors: tuple[tuple[int, int], ...] = (),
    *,
    names: tuple[str, ...] = (),
    origin: tuple[float, float] = (0.0, 0.0),
) -> Connectivity:
    """Measure the free space of a grid of blocked cells; see the module docstring.

    Args:
        blocked: True where an obstacle is, row 0 at the lowest y.
        resolution: side of a cell in metres.
        clearance: what the free space must be like. It must know its robot
            already, since there is no world here to take one from; see
            `Clearance.for_robots`.
        anchors: the (row, column) of each place that must be free and in the
            largest region.
        names: what to call each anchor in a message, such as "robot 'r1'"; an
            anchor with no name is called "anchor 3".
        origin: world position (x, y) of the corner of cell (0, 0), for the report.

    Returns:
        The report, whose `ok` says whether every condition held.
    """
    reach = clearance.reach
    free = ~within(blocked, resolution, reach)
    labels, sizes = numbered(free)
    cells = int(free.sum())
    free_fraction = cells / labels.size if labels.size else 0.0
    main_fraction = sizes[0] / cells if cells else 0.0
    least = math.ceil(round(clearance.smallest_region / resolution**2, 9))
    components = sum(1 for size in sizes if size >= least)
    found = tuple(_region_at(labels, cell) for cell in anchors)
    reasons = []
    if free_fraction < clearance.min_free:
        reasons.append(
            f"a robot of {reach:.2f} m can stand on {free_fraction:.0%} of the "
            f"world, and {clearance.min_free:.0%} is asked for"
        )
    if main_fraction < clearance.min_main:
        reasons.append(
            f"the largest of {components} free regions holds {main_fraction:.0%} "
            f"of the free space, and {clearance.min_main:.0%} is asked for"
        )
    for i, region in enumerate(found):
        if region is None:
            reasons.append(f"{_name(names, i)} has no room for the robot")
        elif region != 0:
            reasons.append(f"{_name(names, i)} is not in the largest free region")
    return Connectivity(
        ok=not reasons,
        reasons=tuple(reasons),
        free_fraction=free_fraction,
        components=components,
        main_fraction=main_fraction,
        anchors=found,
        labels=labels,
        resolution=resolution,
        origin=origin,
    )


def _places(world: World, anchors) -> tuple[list[tuple[float, float]], list[str]]:
    """Every point that must be free: the robots, the goals, then the extra ones."""
    points, names = [], []
    for robot in world.robots:
        points.append(tuple(robot.pose.translation.tolist()))
        names.append(f"robot '{robot.name}'")
    for i, task in enumerate(world.tasks, start=1):
        points.append(tuple(task.goal.translation.tolist()))
        names.append(f"the goal of task {i} ('{task.robot}')")
    for i, point in enumerate(anchors, start=1):
        points.append((float(point[0]), float(point[1])))
        names.append(f"anchor {i}")
    return points, [f"{n} at ({p[0]:.2f}, {p[1]:.2f})" for n, p in zip(names, points)]


def check_connectivity(
    world: World,
    clearance: Clearance = Clearance(),
    anchors: tuple[tuple[float, float], ...] = (),
) -> Connectivity:
    """Measure the free space of a world; see the module docstring.

    The robots and the goals of the tasks are anchors already, and `anchors` adds
    the places a caller has in mind but has not written into the world yet.

    The obstacles and the border of the world are what is rasterised, whatever the
    world's costmap says its source is: a map file is a picture of obstacles, and
    this check is about the ones the world holds.

    A clearance with no radius of its own takes one from the widest robot of the
    world, so a world is measured for the robot that will drive it. A world with no
    robots is measured for a point robot.

    Args:
        world: the world to measure.
        clearance: what the free space must be like, and at which resolution.
        anchors: more points that must be free and in the largest region.

    Returns:
        The report, whose `ok` says whether every condition held.
    """
    clearance = clearance.for_robots([r.shape for r in world.robots])
    resolution = clearance.resolution or world.costmap.resolution
    blocked = generate_static(world, resolution) == LETHAL
    points, names = _places(world, anchors)
    origin = world.origin
    cells = tuple(
        (
            math.floor(round((y - origin[1]) / resolution, 9)),
            math.floor(round((x - origin[0]) / resolution, 9)),
        )
        for x, y in points
    )
    return check_grid(
        blocked,
        resolution,
        clearance,
        cells,
        names=tuple(names),
        origin=origin,
    )


def clearance_for(clearance: Clearance | None, base: World | None) -> Clearance:
    """The clearance a generator works to, with the robot of its base world in it.

    A generator sizes what it draws by the robot that will drive it: how wide a
    corridor must be, how much room an anchor needs, which gap is too narrow to
    count. So it must know the robot, and there are two places to learn it from:
    the `clearance`, if it names a radius, or the widest robot of the `base` world.
    With neither there is nothing to size by, and that is refused rather than
    guessed at, since a guess would quietly draw a map for the wrong robot.

    Args:
        clearance: what the caller asked for, or None for a map that need not pass
            the check at all. Nothing is promised then, so a world with no robot
            is measured for a point robot instead of being refused.
        base: the world the clutter is added to, if there is one.

    Raises:
        WorldError: if a map must pass the check and nothing says how big the
            robot is.
    """
    shapes = [r.shape for r in base.robots] if base is not None else []
    if clearance is None:
        return Clearance().for_robots(shapes)
    if clearance.radius is None and not shapes:
        raise WorldError(
            "generate: nothing says how big the robot is, and what is drawn is "
            "sized by it. Give a base world with a robot in it, or a Clearance "
            "with a radius of its own."
        )
    return clearance.for_robots(shapes)


def refused(
    style: str, attempts: int, last: Connectivity, advice: str = ""
) -> GenerateError:
    """The error a generator raises when no world it drew passed the check.

    Args:
        style: the name of the style, which starts the message.
        attempts: how many worlds were drawn and thrown away.
        last: the report of the last of them, whose reasons are quoted.
        advice: what the caller might change, in a sentence of the style's own.
    """
    times = "attempt" if attempts == 1 else "attempts"
    end = f" {advice}" if advice else ""
    return GenerateError(
        f"{style}: no map passed the free-space check in {attempts} {times}. "
        f"The last one: {'; '.join(last.reasons)}.{end}"
    )
