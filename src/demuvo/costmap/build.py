"""Building the global costmap of each robot from the world file."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from robomath import Transform2D

from ..world import World
from ..world.shapes import Circle
from .grid import Costmap
from .layers import (
    edge_distance,
    generate_static,
    inflate,
    load_static,
    part_cells,
)


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


# The most shapes that may differ from the last map for it to be patched, not built
# again; a world that differs in more is a new one.
MAX_PATCH = 4


def forget() -> None:
    """Drop the kept map, so the next build starts from nothing."""
    _LAST.clear()


def _window(changed, shape, resolution: float, origin) -> tuple[int, int, int, int]:
    """The cells that changed shapes can touch, clipped to the grid."""
    boxes = [
        part_cells(part, resolution, origin)
        for o in changed
        for part in o.shape.parts(o.pose)
    ]
    return (
        max(min(b[0] for b in boxes), 0),
        min(max(b[1] for b in boxes), shape[0]),
        max(min(b[2] for b in boxes), 0),
        min(max(b[3] for b in boxes), shape[1]),
    )


def _changed(old: tuple, new: tuple) -> list:
    """The obstacles that are in one tuple and not in the other, by identity."""
    kept = {id(o) for o in old} & {id(o) for o in new}
    return [o for o in (*old, *new) if id(o) not in kept]


def _patch(world: World, inputs: tuple):
    """The last map changed for the shapes that differ, or None if it cannot be.

    Moving, adding or removing a shape changes the cells round it and no others, so
    only that window of the static layer is made again, and in each inflated map only
    the window and what its inflation reaches. The result is what a build from
    nothing gives.
    """
    spec = world.costmap
    if _LAST["inputs"] != inputs or spec.source != "generate":
        return None
    if "static" not in spec.plugins:
        return None
    changed = _changed(_LAST["obstacles"], world.obstacles)
    if not changed or len(changed) > MAX_PATCH:
        return None
    base, resolution, origin = _LAST["base"]
    window = _window(changed, base.shape, resolution, origin)
    patched = base.copy()
    on_grid = window[0] < window[1] and window[2] < window[3]
    if on_grid:
        patched[window[0] : window[1], window[2] : window[3]] = generate_static(
            world, resolution, window
        )
    patched.flags.writeable = False
    shared = {}
    for radius, costmap in _LAST["shared"].items():
        data = patched if radius is None else costmap.data
        if radius is not None and on_grid:
            data = _inflated_window(data, patched, window, radius, spec)
        shared[radius] = Costmap(data, resolution, origin)
    return patched, shared


def _grown(window, by: int, shape) -> tuple[int, int, int, int]:
    """A window of cells made `by` cells larger on every side, kept on the grid."""
    return (
        max(window[0] - by, 0),
        min(window[1] + by, shape[0]),
        max(window[2] - by, 0),
        min(window[3] + by, shape[1]),
    )


def _inflated_window(inflated, base, window, radius: float, spec) -> np.ndarray:
    """An inflated map with the cells near a window of the base made again."""
    resolution = spec.resolution
    reach = math.ceil(round(max(spec.inflation_radius, radius) / resolution, 9))
    # the cells whose cost the window can change, and the cells that decide them
    changes = _grown(window, reach, base.shape)
    decides = _grown(changes, reach, base.shape)
    piece = inflate(
        base[decides[0] : decides[1], decides[2] : decides[3]],
        resolution,
        radius,
        spec.inflation_radius,
        spec.cost_scaling_factor,
    )
    data = inflated.copy()
    data[changes[0] : changes[1], changes[2] : changes[3]] = piece[
        changes[0] - decides[0] : changes[1] - decides[0],
        changes[2] - decides[2] : changes[3] - decides[2],
    ]
    data.flags.writeable = False
    return data


def _base(world: World):
    """The cells of the static layer, their size and origin; built once per change.

    The same obstacles, size, origin and settings give the same grid, so the last one
    is kept and used again. Obstacles that differ by a shape or two patch it.
    """
    spec = world.costmap
    inputs = (world.size, world.origin, spec, _map_stamp(spec))
    if _LAST.get("obstacles") is world.obstacles and _LAST.get("inputs") == inputs:
        return _LAST["base"]
    patch = _patch(world, inputs) if _LAST else None
    if patch is not None:
        base, shared = patch
        _LAST.update(
            obstacles=world.obstacles,
            base=(base, _LAST["base"][1], _LAST["base"][2]),
            shared=shared,
        )
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
