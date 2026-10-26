"""Shapes of obstacles and robots, and the one table that names them.

To add a shape: write a class like the ones below and add it to `SHAPES`.
The loader needs no other change.

Each shape class says:
    name: the word used in the world file, `shape = {name = "..."}`.
    keys: the other keys it accepts in that table.
    closed: False for a line that encloses no area (a robot cannot be one).
    pose_optional: True if the vertices can be given in world coordinates, so
        the object may leave out the pose.
    parse: builds the shape from its table and checks it.
    points: where its extent lies in the world, for the inside the world verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from demuvo.verify import (
    WorldError,
    check_keys,
    number,
    numbers,
    require_key,
    table,
    text,
)

Point = tuple[float, float]
Pose = tuple[float, float, float]


def place(pose: Pose, points) -> list[Point]:
    """Rotate points by the pose angle, then move them to the pose position."""
    x, y, angle = pose
    c, s = math.cos(angle), math.sin(angle)
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in points]


def _vertices(value, minimum: int, label: str) -> tuple[Point, ...]:
    """Check a list of at least [x, y] points."""
    if not isinstance(value, list) or len(value) < minimum:
        raise WorldError(f"{label}: 'vertices' needs at least {minimum} points")
    return tuple(numbers(v, 2, label, "vertices") for v in value)


@dataclass(frozen=True)
class Circle:
    """A circle.

    Attributes:
        radius: radius in metres.
    """

    radius: float

    name: ClassVar[str] = "circle"
    keys: ClassVar[set[str]] = {"radius"}
    closed: ClassVar[bool] = True
    pose_optional: ClassVar[bool] = False

    @classmethod
    def parse(cls, data: dict, label: str) -> Circle:
        """Build a circle from its table."""
        return cls(number(data, "radius", label))

    def points(self, pose: Pose) -> list[Point]:
        """Two corners of the square around the disc, which is exact for a circle."""
        x, y, _ = pose
        r = self.radius
        return [(x - r, y - r), (x + r, y + r)]


@dataclass(frozen=True)
class Box:
    """A rectangle centred on the pose.

    Attributes:
        size: length along x and width along y, before the rotation.
    """

    size: tuple[float, float]

    name: ClassVar[str] = "box"
    keys: ClassVar[set[str]] = {"size"}
    closed: ClassVar[bool] = True
    pose_optional: ClassVar[bool] = False

    @classmethod
    def parse(cls, data: dict, label: str) -> Box:
        """Build a box from its table."""
        return cls(numbers(require_key(data, "size", label), 2, label, "size", True))

    def points(self, pose: Pose) -> list[Point]:
        """The four corners."""
        half_x, half_y = self.size[0] / 2, self.size[1] / 2
        corners = [(sx * half_x, sy * half_y) for sx in (-1, 1) for sy in (-1, 1)]
        return place(pose, corners)


@dataclass(frozen=True)
class Polygon:
    """A closed polygon.

    Attributes:
        vertices: corners in the frame of the pose, in order around the shape.
    """

    vertices: tuple[Point, ...]

    name: ClassVar[str] = "polygon"
    keys: ClassVar[set[str]] = {"vertices"}
    closed: ClassVar[bool] = True
    pose_optional: ClassVar[bool] = True

    @classmethod
    def parse(cls, data: dict, label: str) -> Polygon:
        """Build a polygon from its table."""
        return cls(_vertices(require_key(data, "vertices", label), 3, label))

    def points(self, pose: Pose) -> list[Point]:
        """The vertices in the world."""
        return place(pose, self.vertices)


@dataclass(frozen=True)
class LineString:
    """A chain of wall segments with no area.

    Attributes:
        vertices: points joined in order, in the frame of the pose.
    """

    vertices: tuple[Point, ...]

    name: ClassVar[str] = "linestring"
    keys: ClassVar[set[str]] = {"vertices"}
    closed: ClassVar[bool] = False
    pose_optional: ClassVar[bool] = True

    @classmethod
    def parse(cls, data: dict, label: str) -> LineString:
        """Build a line string from its table."""
        return cls(_vertices(require_key(data, "vertices", label), 2, label))

    def points(self, pose: Pose) -> list[Point]:
        """The vertices in the world."""
        return place(pose, self.vertices)


Shape = Circle | Box | Polygon | LineString

SHAPES: dict[str, type[Shape]] = {
    cls.name: cls for cls in (Circle, Box, Polygon, LineString)
}


def parse_shape(value, label: str, robot: bool = False) -> Shape:
    """Build a shape from the `shape = {name = ..., ...}` table of an object.

    Args:
        value: the table read from the file.
        label: the object, for example "obstacle 2".
        robot: True if the shape is a robot's, which must enclose an area.
    """
    data = table(value, f"{label} shape")
    name = text(data, "name", f"{label} shape")
    if name not in SHAPES:
        known = ", ".join(SHAPES)
        raise WorldError(f"{label}: unknown shape '{name}', expected one of {known}")
    cls = SHAPES[name]
    if robot and not cls.closed:
        raise WorldError(f"{label}: shape '{name}' has no area, a robot needs one")
    check_keys(data, {"name"} | cls.keys, f"{label} shape")
    return cls.parse(data, label)
