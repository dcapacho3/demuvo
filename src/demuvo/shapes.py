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
    points: the corners (or the ends of a line) in the world; the loader checks
        that they lie inside the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from robomath import Transform2D, act_se2

from .verify import (
    WorldError,
    check_keys,
    number,
    numbers,
    require_key,
    table,
    text,
)

Point = tuple[float, float]


def _place(pose: Transform2D, points) -> list[Point]:
    """Rotate points by the pose angle, then move them to the pose position."""
    return list(map(tuple, act_se2(pose.matrix, points).tolist()))


def _vertices(value, minimum: int, label: str) -> tuple[Point, ...]:
    """Check a list of at least [x, y] points, none equal to the one before."""
    if not isinstance(value, list) or len(value) < minimum:
        raise WorldError(f"{label}: 'vertices' needs at least {minimum} points")
    points = tuple(
        numbers(v, 2, label, f"vertex {i}") for i, v in enumerate(value, start=1)
    )
    if any(a == b for a, b in zip(points, points[1:])):
        raise WorldError(f"{label}: two vertices in a row are equal")
    return points


def _cross(o: Point, a: Point, b: Point) -> float:
    """Twice the signed area of the triangle o, a, b."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on(a: Point, b: Point, p: Point) -> bool:
    """True if p, which is on the line through a and b, is between them."""
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and (
        min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _touch(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True if the segments ab and cd share a point."""
    d1, d2, d3, d4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return (
        (d1 == 0 and _on(a, b, c))
        or (d2 == 0 and _on(a, b, d))
        or (d3 == 0 and _on(c, d, a))
        or (d4 == 0 and _on(c, d, b))
    )


def _check_simple(vertices: tuple[Point, ...], label: str) -> None:
    """Refuse a polygon with no area, a spike, or edges that touch.

    The tests are exact, which suits coordinates typed by hand.
    """
    n = len(vertices)
    if vertices[0] == vertices[-1]:
        raise WorldError(f"{label}: the first and last vertex are equal")
    twice_area = sum(
        vertices[i][0] * vertices[(i + 1) % n][1]
        - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )
    if twice_area == 0:
        raise WorldError(f"{label}: the polygon has no area")
    for i in range(n):
        prev, here, after = vertices[i - 1], vertices[i], vertices[(i + 1) % n]
        turns_back = (here[0] - prev[0]) * (after[0] - here[0]) + (
            here[1] - prev[1]
        ) * (after[1] - here[1]) < 0
        if _cross(prev, here, after) == 0 and turns_back:
            raise WorldError(f"{label}: the polygon doubles back at vertex {i + 1}")
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _touch(vertices[i], vertices[i + 1], vertices[j], vertices[(j + 1) % n]):
                raise WorldError(f"{label}: edges {i + 1} and {j + 1} touch")


@dataclass(frozen=True)
class Circle:
    """A circle. The angle of the pose does not change it.

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

    def points(self, pose: Transform2D) -> list[Point]:
        """Two corners of the square around the disc, which is exact for a circle."""
        x, y = pose.translation
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

    def points(self, pose: Transform2D) -> list[Point]:
        """The four corners, in order around the box."""
        hx, hy = self.size[0] / 2, self.size[1] / 2
        return _place(pose, [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])


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
        vertices = _vertices(require_key(data, "vertices", label), 3, label)
        _check_simple(vertices, label)
        return cls(vertices)

    def points(self, pose: Transform2D) -> list[Point]:
        """The vertices in the world."""
        return _place(pose, self.vertices)


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

    def points(self, pose: Transform2D) -> list[Point]:
        """The vertices in the world."""
        return _place(pose, self.vertices)


class Shape(Protocol):
    """What every shape class offers; see the module docstring."""

    name: ClassVar[str]
    keys: ClassVar[set[str]]
    closed: ClassVar[bool]
    pose_optional: ClassVar[bool]

    @classmethod
    def parse(cls, data: dict, label: str) -> Shape:
        """Build the shape from its table."""

    def points(self, pose: Transform2D) -> list[Point]:
        """Where the shape lies in the world."""


# The one place that lists the shapes, a new shape class needs only an entry here.
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
