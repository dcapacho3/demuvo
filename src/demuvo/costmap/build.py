"""Building the global costmap of each robot from the world file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from robomath import Transform2D

from ..world import World
from ..world.shapes import Circle
from .grid import Costmap
from .layers import edge_distance, generate_static, inflate, load_static


def inscribed_radius(shape) -> float:
    """The radius of the largest circle about the robot's origin that fits in its body.

    A circle gives its radius. Any other body gives the distance from the origin to
    the nearest edge of its outline, or 0 if the origin is not inside the outline.
    """
    if isinstance(shape, Circle):
        return shape.radius
    outline = np.array(shape.points(Transform2D.identity()), dtype=float)
    x, y = outline.T
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    crossing = (y > 0) != (y2 > 0)
    at = x + (0 - y) * (x2 - x) / np.where(crossing, y2 - y, 1.0)
    if (crossing & (at > 0)).sum() % 2 == 0:
        return 0.0
    return float(edge_distance(np.zeros((1, 2)), outline)[0])


# The last global map that was built, kept because an edit that moves a robot or
# sets a goal leaves the obstacles alone, and a drag makes dozens of them a second.
# Its costs are read-only, so any number of worlds may share it.
_LAST: dict = {}


def _map_stamp(spec) -> tuple[float, int] | None:
    """When the map file of a costmap was last written, so a changed file is seen."""
    if spec.source == "generate":
        return None
    try:
        found = Path(spec.source).stat()
    except OSError:
        return None  # the build below reports a map that cannot be read
    return found.st_mtime, found.st_size


def _base(world: World):
    """The cells of the static layer, their size and origin; built once per change.

    The same obstacles, size, origin and settings give the same grid, so the last one
    is kept and used again.
    """
    spec = world.costmap
    inputs = (world.size, world.origin, spec, _map_stamp(spec))
    if _LAST.get("obstacles") is world.obstacles and _LAST.get("inputs") == inputs:
        return _LAST["base"]
    origin = world.origin
    resolution = spec.resolution
    if "static" not in spec.plugins:
        base = np.zeros(generate_static(world, resolution).shape, dtype=np.uint8)
    elif spec.source == "generate":
        base = generate_static(world, resolution)
    else:
        base, resolution, origin = load_static(Path(spec.source), spec.unknown)
    base.flags.writeable = False
    # the holder keeps the obstacles alive, so their identity is not reused
    _LAST.update(
        obstacles=world.obstacles,
        inputs=inputs,
        base=(base, resolution, origin),
        shared={},
    )
    return _LAST["base"]


def build_costmaps(world: World) -> dict[str, Costmap | None]:
    """The global costmap of each robot, by name, all None if the world turns it off.

    The static layer is built once and shared. Inflation depends on the robot's
    inscribed radius, so robots with the same radius share one costmap. A world with
    the same obstacles, size and costmap settings as the last one gets the same
    costmaps back, so editing a robot does not build them again.

    Raises:
        WorldError: if a map file named in the world file cannot be read.
    """
    spec = world.costmap
    if not spec.enabled:
        return {r.name: None for r in world.robots}
    base, resolution, origin = _base(world)
    shared: dict[float | None, Costmap] = _LAST["shared"]
    result = {}
    for robot in world.robots:
        radius = None
        if "inflation" in spec.plugins:
            radius = round(inscribed_radius(robot.shape), 6)
        if radius not in shared:
            data = base
            if radius is not None:
                data = inflate(
                    base,
                    resolution,
                    radius,
                    spec.inflation_radius,
                    spec.cost_scaling_factor,
                )
                data.flags.writeable = False
            shared[radius] = Costmap(data, resolution, origin)
        result[robot.name] = shared[radius]
    return result
