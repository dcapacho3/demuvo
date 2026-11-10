"""The view - how world metres map to window pixels.

World x runs right and y runs up; the window's y runs down. The view holds the
rectangle of the window the scene lives in, a scale in pixels per metre and the
pixel of the world's origin. `fit` sets them so the whole board shows, with a
margin and room at the bottom for the playback controls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Metres of margin round the board.
MARGIN = 0.35


@dataclass(frozen=True)
class View:
    """A mapping between world metres and window pixels.

    Attributes:
        bounds: the world's (x_min, y_min, x_max, y_max) in metres.
        rect: the window rectangle (x, y, width, height) in pixels.
        scale: pixels per metre.
        ox: the pixel x of world x = 0.
        oy: the pixel y of world y = 0.
    """

    bounds: tuple[float, float, float, float]
    rect: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)
    scale: float = 50.0
    ox: float = 0.0
    oy: float = 0.0

    def fit(
        self, rect: tuple[float, float, float, float], bottom_room: float = 60.0
    ) -> View:
        """A view that shows the whole board inside a rectangle, centred.

        Args:
            rect: (x, y, width, height) of the area in pixels.
            bottom_room: pixels kept free along the bottom edge.
        """
        x, y, width, height = rect
        x0, y0, x1, y1 = self.bounds
        board_w, board_h = x1 - x0 + 2 * MARGIN, y1 - y0 + 2 * MARGIN
        scale = min(width / board_w, (height - bottom_room) / board_h)
        left = x + (width - board_w * scale) / 2 + MARGIN * scale - x0 * scale
        top = y + (height - bottom_room - board_h * scale) / 2
        return replace(
            self, rect=rect, scale=scale, ox=left, oy=top + MARGIN * scale + y1 * scale
        )

    def to_screen(self, x, y):
        """Pixel x and y of world points; numbers or arrays."""
        return self.ox + np.asarray(x) * self.scale, self.oy - np.asarray(
            y
        ) * self.scale

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """World x and y of a pixel."""
        return (sx - self.ox) / self.scale, (self.oy - sy) / self.scale

    def board(self) -> tuple[float, float, float, float]:
        """The board's pixel box (left, top, right, bottom)."""
        x0, y0, x1, y1 = self.bounds
        (left, top), (right, bottom) = self.to_screen(x0, y1), self.to_screen(x1, y0)
        return float(left), float(top), float(right), float(bottom)

    def shifted(self, dx: float, dy: float) -> View:
        """The same view with every pixel moved by (dx, dy), to draw into a cut-out."""
        return replace(self, ox=self.ox + dx, oy=self.oy + dy)
