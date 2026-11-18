"""Reading a Gazebo world - the boxes, cylinders and spheres it stands on the floor.

A `.world` or `.sdf` file gives the obstacles of a demuvo world: each collision shape
that stands on the floor, seen from above. Floors and planes, shapes too high to touch
a robot, tilted shapes and meshes are left out, since demuvo's world is flat. There
are no robots, as in an empty Gazebo world; the sandbox adds them.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from robomath import Transform2D

from ..verify import WorldError

SUFFIXES = (".world", ".sdf")
# A shape whose top is below this is floor, and one whose bottom is above the next
# height is overhead; both are left out. Both in metres.
FLOOR = 0.1
REACH = 1.0
# A shape tilted more than this, in radians, is not a flat obstacle.
TILT = 0.2
# Metres of free floor left round the shapes.
MARGIN = 1.0


def _pose(element) -> tuple[float, float, float, float, float]:
    """The (x, y, z, tilt, yaw) of a `<pose>` child; the origin if it has none."""
    text = element.findtext("pose") if element is not None else None
    values = [float(v) for v in text.split()] if text else []
    x, y, z, roll, pitch, yaw = [*values, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:6]
    return x, y, z, max(abs(roll), abs(pitch)), yaw


def _compose(outer, inner):
    """The pose of something placed at `inner` in the frame placed at `outer`."""
    flat = Transform2D.from_xyt((outer[0], outer[1], outer[4])) @ Transform2D.from_xyt(
        (inner[0], inner[1], inner[4])
    )
    x, y, yaw = (float(v) for v in flat.xyt)
    return x, y, outer[2] + inner[2], max(outer[3], inner[3]), yaw


def _shape(geometry, pose):
    """The obstacle of one geometry at a pose, or None if it is not one."""
    x, y, z, tilt, yaw = pose
    if geometry.find("box") is not None:
        sx, sy, sz = (float(v) for v in geometry.findtext("box/size").split())
        shape = {"name": "box", "size": [sx, sy]}
    elif geometry.find("cylinder") is not None:
        radius = float(geometry.findtext("cylinder/radius"))
        sz = float(geometry.findtext("cylinder/length"))
        shape = {"name": "circle", "radius": radius}
    elif geometry.find("sphere") is not None:
        sz = radius = float(geometry.findtext("sphere/radius")) * 2
        shape = {"name": "circle", "radius": radius / 2}
    else:
        return None
    flat = tilt <= TILT and z + sz / 2 > FLOOR and z - sz / 2 < REACH
    return {"pose": [x, y, yaw], "shape": shape} if flat else None


def _extent(obstacle) -> tuple[float, float, float, float]:
    """The lowest and highest x and y a shape reaches."""
    x, y, yaw = obstacle["pose"]
    shape = obstacle["shape"]
    if shape["name"] == "circle":
        half_x = half_y = shape["radius"]
    else:
        width, depth = shape["size"]
        half_x = (abs(width * math.cos(yaw)) + abs(depth * math.sin(yaw))) / 2
        half_y = (abs(width * math.sin(yaw)) + abs(depth * math.cos(yaw))) / 2
    return x - half_x, y - half_y, x + half_x, y + half_y


def _models(root):
    """Every model of the file, wherever it sits."""
    yield from root.iter("model")


def read_sdf(path: str | Path) -> dict:
    """The data of a demuvo world file for the shapes of a Gazebo world.

    Args:
        path: a `.world` or `.sdf` file.

    Returns:
        The same dictionary that `tomllib` gives for a world file: its version,
        a `world` table sized to hold the shapes and the `obstacle` list. A model
        nested in another is placed as if it stood alone.

    Raises:
        WorldError: if the file is not valid XML or holds no shape to stand on.
        OSError: if the file cannot be read.
    """
    try:
        root = ET.fromstring(Path(path).read_text(encoding="utf-8").lstrip())
    except ET.ParseError as error:
        raise WorldError(f"file: not valid XML: {error}") from error
    obstacles, left_out = [], 0
    for model in _models(root):
        placed = _pose(model)
        for link in model.findall("link"):
            link_pose = _compose(placed, _pose(link))
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                where = _compose(link_pose, _pose(collision))
                found = _shape(geometry, where) if geometry is not None else None
                if found is not None:
                    obstacles.append(found)
                elif geometry is not None and geometry.find("mesh") is not None:
                    left_out += 1
    if left_out:
        warnings.warn(f"{left_out} mesh shapes were left out; demuvo reads boxes only")
    if not obstacles:
        raise WorldError("file: no box, cylinder or sphere stands on the floor")
    extents = [_extent(o) for o in obstacles]
    # Rounded first, so float noise in a turned box does not add a whole metre.
    x0 = math.floor(round(min(e[0] for e in extents) - MARGIN, 3))
    y0 = math.floor(round(min(e[1] for e in extents) - MARGIN, 3))
    x1 = math.ceil(round(max(e[2] for e in extents) + MARGIN, 3))
    y1 = math.ceil(round(max(e[3] for e in extents) + MARGIN, 3))
    return {
        "version": 1,
        "world": {
            "size": [float(x1 - x0), float(y1 - y0)],
            "origin": [float(x0), float(y0)],
            "seed": 1,
        },
        "obstacle": obstacles,
    }
