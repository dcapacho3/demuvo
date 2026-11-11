"""Line icons on a 24-unit grid, drawn with the renderer.

Every icon has the same weight, so a toolbar reads as one set. An icon is a list of
parts: ("line", points), ("fill", points), ("ring", x, y, radius) or
("disc", x, y, radius), in grid units.
"""

from __future__ import annotations

import math

from .raster import Canvas, rgba
from .strokes import polyline

GRID = 24.0
# Stroke width in grid units.
WEIGHT = 1.75
BOX = [(3.5, 4.5), (20.5, 4.5), (20.5, 19.5), (3.5, 19.5), (3.5, 4.5)]

ICONS = {
    "select": [
        (
            "line",
            [
                (6, 4),
                (6, 19),
                (10, 15),
                (13, 21),
                (15.5, 19.8),
                (12.6, 14),
                (18, 14),
                (6, 4),
            ],
        )
    ],
    "box": [("line", [(4.5, 6.5), (19.5, 6.5), (19.5, 17.5), (4.5, 17.5), (4.5, 6.5)])],
    "circle": [("ring", 12, 12, 7.5)],
    "robot_add": [
        ("ring", 10.5, 13.5, 6.5),
        ("line", [(10.5, 13.5), (15.5, 13.5)]),
        ("line", [(19, 3.5), (19, 9.5)]),
        ("line", [(16, 6.5), (22, 6.5)]),
    ],
    "goal": [
        ("line", [(6, 20.5), (6, 3.5)]),
        ("line", [(6, 4.5), (19, 8.5), (6, 12.5), (6, 4.5)]),
    ],
    "trash": [
        ("line", [(5, 7), (19, 7)]),
        ("line", [(9.5, 7), (10, 4.5), (14, 4.5), (14.5, 7)]),
        ("line", [(6.5, 8), (17.5, 8), (17.5, 20.5), (6.5, 20.5), (6.5, 8)]),
        ("line", [(10.5, 11.5), (10.5, 17)]),
        ("line", [(13.5, 11.5), (13.5, 17)]),
    ],
    "panel_left": [("line", BOX), ("line", [(9.5, 4.5), (9.5, 19.5)])],
    "panel_right": [("line", BOX), ("line", [(14.5, 4.5), (14.5, 19.5)])],
    "play": [("fill", [(8, 5.5), (19, 12), (8, 18.5)])],
    "pause": [
        ("fill", [(6.5, 5.5), (10.5, 5.5), (10.5, 18.5), (6.5, 18.5)]),
        ("fill", [(13.5, 5.5), (17.5, 5.5), (17.5, 18.5), (13.5, 18.5)]),
    ],
    "step": [
        ("fill", [(6, 5.5), (15, 12), (6, 18.5)]),
        ("fill", [(16.5, 5.5), (19.5, 5.5), (19.5, 18.5), (16.5, 18.5)]),
    ],
    "chevron_down": [("line", [(7, 10), (12, 15), (17, 10)])],
    "check": [("line", [(6, 12.5), (10.5, 17), (18, 8)])],
}
# The reset icon is an arc with an arrowhead, so it is worked out and not listed.
RESET = "reset"


def _reset(canvas: Canvas, place, width: float, color, scale: float) -> None:
    """An open circle arrow: 281 degrees of arc, and a head at its end."""
    arc = [
        place((12 + 7 * math.cos(math.radians(a)), 12 - 7 * math.sin(math.radians(a))))
        for a in range(140, 140 - 281, -10)
    ]
    polyline(canvas, arc, width, color)
    (x1, y1), (x0, y0) = arc[-1], arc[-3]
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    for side in (1, -1):
        back_x = x1 - 3.4 * scale * dx - side * 2.6 * scale * dy
        back_y = y1 - 3.4 * scale * dy + side * 2.6 * scale * dx
        polyline(canvas, [(x1, y1), (back_x, back_y)], width, color)


def draw_icon(
    canvas: Canvas,
    name: str,
    center: tuple[float, float],
    size: float,
    color: str,
    *,
    opacity: float = 1.0,
) -> None:
    """Draw an icon centred at `center`, `size` pixels across, in a "#rrggbb" colour.

    Raises:
        KeyError: if there is no icon of that name.
    """
    scale = size / GRID
    left, top = center[0] - size / 2, center[1] - size / 2
    paint = rgba(color, opacity)
    width = WEIGHT * scale

    def place(point):
        return (left + point[0] * scale, top + point[1] * scale)

    if name == RESET:
        _reset(canvas, place, width, paint, scale)
        return
    for kind, *args in ICONS[name]:
        if kind == "line":
            polyline(canvas, [place(q) for q in args[0]], width, paint)
        elif kind == "fill":
            canvas.convex(
                [place(q) for q in args[0]], fill=paint, stroke=paint, width=width * 0.5
            )
        elif kind == "ring":
            x, y = place((args[0], args[1]))
            canvas.circle(
                (x, y), args[2] * scale + width / 2, stroke=paint, width=width
            )
        else:
            canvas.circle(place((args[0], args[1])), args[2] * scale, fill=paint)
