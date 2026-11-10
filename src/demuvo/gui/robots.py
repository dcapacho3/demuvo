"""Drawing the robots - each with its real footprint and a white arrow for its heading.

A robot is a disc for a circular body, and the outline for a box or any polygon,
in its own colour with a white edge. The arrow sits at the robot's origin and is
sized by the footprint's inscribed radius, so it looks the same on every shape.

Everything is drawn as sprites. A disc never turns, so its body is drawn once; only
the small arrow depends on the heading, and it is shared by every robot of that
size at that heading, in steps of two degrees. A polygon turns, so its body is
cached by heading too.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from robomath import Transform2D

from ..costmap.build import inscribed_radius
from ..world.shapes import Circle
from . import font, look
from .raster import Canvas, is_convex, rgba
from .sprites import Sprite, draw_sprites

# Headings are cached in steps of this many degrees.
HEADING_STEP = 2.0
# A label is drawn for every robot until there are more than this many.
LABELS_UP_TO = 40
# The corners of the heading arrow, as fractions of the inscribed radius.
ARROW_UPPER = ((0.62, 0), (-0.28, -0.5), (-0.05, 0))
ARROW_LOWER = ((0.62, 0), (-0.05, 0), (-0.28, 0.5))


class Footprint(NamedTuple):
    """The shape of a robot's body, as the renderer needs it.

    Attributes:
        circular: True for a disc.
        outline: the corners in the robot's frame, metres; None for a disc.
        inscribed: the radius of the largest circle about the origin that fits.
        bound: the radius of the smallest circle about the origin that holds it.
    """

    circular: bool
    outline: np.ndarray | None
    inscribed: float
    bound: float


def footprint(shape) -> Footprint:
    """The footprint of a robot shape."""
    inscribed = inscribed_radius(shape)
    if isinstance(shape, Circle):
        return Footprint(True, None, inscribed or shape.radius, shape.radius)
    outline = np.array(shape.points(Transform2D.identity()), dtype=float)
    bound = float(np.hypot(outline[:, 0], outline[:, 1]).max())
    if inscribed <= 0:
        inscribed = float(np.abs(outline).min() or 0.1)
    return Footprint(False, outline, inscribed, bound)


def _heading_arrow(canvas: Canvas, center, reach: float, theta: float) -> None:
    """The white arrow, `reach` pixels long for a fraction of 1, along an angle."""
    cos, sin = math.cos(theta), math.sin(theta)
    white = rgba("#ffffff")
    for corners in (ARROW_UPPER, ARROW_LOWER):
        points = [
            (
                center[0] + (u * cos - v * sin) * reach,
                center[1] - (u * sin + v * cos) * reach,
            )
            for u, v in corners
        ]
        canvas.convex(points, fill=white)


def _halo(canvas: Canvas, center, radius: float) -> None:
    """The soft white disc with a navy edge behind the selected robot."""
    canvas.circle(
        center, radius, fill=rgba("#ffffff", 0.43), stroke=rgba(look.NAVY), width=1.6
    )


def _halo_radius(foot: Footprint, scale: float) -> float:
    """The radius of the halo of a selected robot, in pixels."""
    return max(1.9 * foot.bound * scale, 16.0)


def _disc_drawing(foot: Footprint, color: str, scale: float, halo: float | None):
    """The drawing of a disc robot's body; `halo` is None if it is not selected."""

    def draw(canvas: Canvas, cx: float, cy: float) -> None:
        if halo is not None:
            _halo(canvas, (cx, cy), halo)
        canvas.circle(
            (cx, cy),
            foot.bound * scale,
            fill=rgba(color),
            stroke=rgba("#ffffff"),
            width=2.4,
        )

    return draw


def _polygon_drawing(foot, color, scale, heading, halo):
    """A drawing function for a polygon robot at a heading, with its arrow inside."""

    def draw(canvas: Canvas, cx: float, cy: float) -> None:
        if halo is not None:
            _halo(canvas, (cx, cy), halo)
        cos, sin = math.cos(heading), math.sin(heading)
        local = foot.outline
        corners = np.column_stack(
            [
                cx + (local[:, 0] * cos - local[:, 1] * sin) * scale,
                cy - (local[:, 0] * sin + local[:, 1] * cos) * scale,
            ]
        )
        if is_convex(corners):
            canvas.convex(corners, fill=rgba(color), stroke=rgba("#ffffff"), width=2.4)
        else:
            canvas.polygon(corners, rgba(color))
        _heading_arrow(canvas, (cx, cy), foot.inscribed * scale, heading)

    return draw


def _arrow_sprite(scene, reach: float, degrees: float) -> Sprite:
    """The white arrow shared by every disc robot of this inscribed size and heading."""
    heading = math.radians(degrees)

    def draw(canvas: Canvas, cx: float, cy: float) -> None:
        _heading_arrow(canvas, (cx, cy), reach, heading)

    return scene.sprites.get(("arrow", round(reach, 1), degrees), 2 * reach + 8, draw)


def _label_sprite(scene, name: str, color: str) -> Sprite:
    """The name of a robot in bold, in its colour."""
    face = font.bold()

    def draw(canvas: Canvas, _cx: float, cy: float) -> None:
        canvas.text(face, name, (4, cy + 4.5), size=13, color=rgba(color))

    return scene.sprites.get(("label", name, color), face.width(name, 13) + 8, draw)


def _placed(sprite: Sprite, x: float, y: float) -> tuple[Sprite, int, int]:
    """A sprite with the top-left corner that centres it on a point."""
    return sprite, round(x - sprite.width / 2), round(y - sprite.height / 2)


def _sprites_of(scene, name: str, x: float, y: float, degrees: float):
    """The body, arrow and label sprites of one robot, each placed on the screen.

    The arrow is None for a polygon, which has its own inside the body; the label
    is None for a robot in a crowd that is not selected.
    """
    foot, color, scale = scene.feet[name], scene.colors[name], scene.view.scale
    selected = name == scene.selected
    halo = _halo_radius(foot, scale) if selected else None
    size = 2 * (halo if selected else foot.bound * scale) + 14
    arrow = None
    if foot.circular:
        key = ("body", name, round(scale, 1), selected)
        drawing = _disc_drawing(foot, color, scale, halo)
        arrow = _placed(_arrow_sprite(scene, foot.inscribed * scale, degrees), x, y)
    else:
        key = ("robot", name, round(scale, 1), degrees, selected)
        drawing = _polygon_drawing(foot, color, scale, math.radians(degrees), halo)
    body = _placed(scene.sprites.get(key, size, drawing), x, y)
    label = None
    if len(scene.world.robots) <= LABELS_UP_TO or selected:
        above = y - halo - 8 if selected else y - foot.bound * scale - 10
        label = _placed(_label_sprite(scene, name, color), x, above)
    return body, arrow, label


def draw_robots(scene, canvas: Canvas, opacity: float) -> None:
    """Draw every robot of a scene: bodies, then arrows, then labels.

    Args:
        scene: the `SceneDraw`; it holds the world, view, colours and caches.
        canvas: where to draw, inside its clip.
        opacity: the opacity of the robots layer.
    """
    names = [r.name for r in scene.world.robots]
    poses = np.array([scene.pose(n) for n in names], dtype=float).reshape(-1, 3)
    px, py = scene.view.to_screen(poses[:, 0], poses[:, 1])
    degrees = np.round(np.degrees(poses[:, 2]) / HEADING_STEP) * HEADING_STEP
    made = [
        _sprites_of(scene, name, float(px[i]), float(py[i]), float(degrees[i]))
        for i, name in enumerate(names)
    ]
    for group in zip(*made):
        placed = [item for item in group if item is not None]
        if placed:
            sprites, xs, ys = zip(*placed)
            draw_sprites(canvas, list(sprites), xs, ys, opacity)


def robot_at(scene, sx: float, sy: float) -> str | None:
    """The robot under a pixel, the nearest if several; a small one is easy to hit."""
    best, best_distance = None, math.inf
    for robot in scene.world.robots:
        x, y = scene.view.to_screen(*scene.pose(robot.name)[:2])
        reach = max(scene.feet[robot.name].bound * scene.view.scale, 12.0)
        distance = math.hypot(sx - float(x), sy - float(y))
        if distance <= reach and distance < best_distance:
            best, best_distance = robot.name, distance
    return best


def robot_boxes(scene) -> dict[str, tuple[int, int, int, int]]:
    """The pixel box round each robot that holds its sprite, halo and label."""
    names = [r.name for r in scene.world.robots]
    poses = np.array([scene.pose(n) for n in names], dtype=float).reshape(-1, 3)
    sx, sy = scene.view.to_screen(poses[:, 0], poses[:, 1])
    reach = np.array(
        [_halo_radius(scene.feet[n], scene.view.scale) + 30 for n in names]
    )
    left, top = (sx - reach).astype(int), (sy - reach).astype(int)
    right, bottom = (sx + reach).astype(int), (sy + reach).astype(int)
    return {
        n: (int(left[i]), int(top[i]), int(right[i]), int(bottom[i]))
        for i, n in enumerate(names)
    }
