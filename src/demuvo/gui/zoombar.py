"""The scale bar - a line over the scene that says how long some metres are."""

from __future__ import annotations

import numpy as np

from . import font, look
from .raster import Canvas, rgba
from .strokes import polyline
from .view import View

# The lengths a scale bar may have, in metres, and the pixels it may be across.
LENGTHS = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)
LONGEST = 150.0
# Pixels from the bottom left corner of the scene.
EDGE = 16.0


def nice_length(scale: float) -> float:
    """The longest of `LENGTHS` that is at most `LONGEST` pixels across at a scale."""
    fits = [m for m in LENGTHS if m * scale <= LONGEST]
    return float(fits[-1] if fits else LENGTHS[0])


def draw_scale_bar(canvas: Canvas, view: View, percent: int) -> None:
    """Draw the bar at the bottom left of the scene, with the zoom beside it."""
    x, y, _, height = view.rect
    metres = nice_length(view.scale)
    width = metres * view.scale
    left, base = x + EDGE, y + height - EDGE - 26
    canvas.rect(
        (left - 8, base - 26, max(width, 96.0) + 16, 38),
        rgba(look.PANEL, 0.8),
        radius=6,
    )
    colour = rgba(look.NAVY, 0.85)
    for points in (
        [(left, base), (left + width, base)],
        [(left, base - 4), (left, base + 4)],
        [(left + width, base - 4), (left + width, base + 4)],
    ):
        polyline(canvas, np.array(points, dtype=float), 2, colour)
    label = f"{metres:g} m" + (f"   {percent}%" if percent != 100 else "")
    canvas.text(
        font.regular(),
        label,
        (left, base - 8),
        size=11,
        color=rgba(look.NAVY, 0.85),
    )
