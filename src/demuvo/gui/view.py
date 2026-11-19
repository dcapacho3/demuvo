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
# How far a view may be zoomed out, as a share of the fitted view, and in, as the
# most pixels a metre may have: a cell of 0.05 m is then 20 pixels across.
MIN_ZOOM = 0.4
MAX_SCALE = 400.0
# Pixels of the board that always stay in the window, so it cannot be lost.
KEEP = 48.0


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


@dataclass
class Camera:
    """How far the user has zoomed and moved the view from the fitted one.

    The fitted view shows the whole board. A zoom of 2 shows half of it, about the
    middle of the scene, and the shift moves the picture by pixels on top of that.

    Attributes:
        zoom: how many times larger than the fitted view.
        shift: pixels the picture has been moved right and down.
    """

    zoom: float = 1.0
    shift: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def centre(rect, room: float) -> tuple[float, float]:
        """The pixel the fitted view is centred on: the middle of the free area."""
        x, y, width, height = rect
        return x + width / 2, y + (height - room) / 2

    @staticmethod
    def _limit(base: View) -> float:
        """The largest zoom: the one that gives `MAX_SCALE` pixels a metre."""
        return max(MAX_SCALE / base.scale, 1.0)

    def view(self, base: View, room: float) -> View:
        """The fitted view with this zoom and shift, kept so the board stays seen."""
        self.zoom = min(max(self.zoom, MIN_ZOOM), self._limit(base))
        cx, cy = self.centre(base.rect, room)
        sx, sy = self.shift
        zoomed = replace(
            base,
            scale=base.scale * self.zoom,
            ox=cx + (base.ox - cx) * self.zoom + sx,
            oy=cy + (base.oy - cy) * self.zoom + sy,
        )
        left, top, right, bottom = zoomed.board()
        x, y, width, height = base.rect
        dx = max(x + KEEP - right, 0.0) - max(left - (x + width - KEEP), 0.0)
        dy = max(y + KEEP - bottom, 0.0) - max(top - (y + height - KEEP), 0.0)
        if dx or dy:
            self.shift = (sx + dx, sy + dy)
            zoomed = zoomed.shifted(dx, dy)
        return zoomed

    def zoom_by(self, factor: float, about, base: View, room: float) -> None:
        """Zoom by a factor, keeping the pixel `about` over the same world point."""
        old = self.zoom
        new = min(max(old * factor, MIN_ZOOM), self._limit(base))
        cx, cy = self.centre(base.rect, room)
        ratio = new / old
        px, py = about if about is not None else (cx, cy)
        self.shift = (
            (px - cx) * (1 - ratio) + self.shift[0] * ratio,
            (py - cy) * (1 - ratio) + self.shift[1] * ratio,
        )
        self.zoom = new

    def pan_by(self, dx: float, dy: float) -> None:
        """Move the picture by pixels."""
        self.shift = (self.shift[0] + dx, self.shift[1] + dy)

    def reset(self) -> None:
        """Show the whole board again."""
        self.zoom, self.shift = 1.0, (0.0, 0.0)

    def frame(self, box, base: View, room: float, margin: float = 1.4) -> None:
        """Zoom and move so a box of world metres (x0, y0, x1, y1) fills the window."""
        x0, y0, x1, y1 = box
        _, _, width, height = base.rect
        wide = max(x1 - x0, 1e-6) * margin
        high = max(y1 - y0, 1e-6) * margin
        zoom = min(width / wide, (height - room) / high) / base.scale
        self.zoom = min(max(zoom, MIN_ZOOM), self._limit(base))
        cx, cy = self.centre(base.rect, room)
        self.shift = (
            -self.zoom * (base.ox - cx + (x0 + x1) / 2 * base.scale),
            -self.zoom * (base.oy - cy - (y0 + y1) / 2 * base.scale),
        )

    @property
    def percent(self) -> int:
        """The zoom as a percentage of the fitted view."""
        return round(self.zoom * 100)
