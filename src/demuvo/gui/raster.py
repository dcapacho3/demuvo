"""A small anti-aliased 2D renderer on numpy.

A canvas is an opaque picture of float32 RGB values from 0 to 1. A shape finds how
much of each pixel it covers, inside the box that holds it, and blends one colour
through that coverage. Rectangles, circles and convex polygons take the coverage
from a signed distance; any other polygon is sampled; text blends glyph bitmaps.

Coordinates are pixels, x to the right and y down, and the centre of a pixel is at
+0.5. A colour is (red, green, blue, opacity), each from 0 to 1.
"""

from __future__ import annotations

import math

import numpy as np

from .font import Font

Color = tuple[float, float, float, float]
Box = tuple[float, float, float, float]  # x, y, width, height
Point = tuple[float, float]
# Samples per pixel along each axis when a polygon is filled by sampling.
POLYGON_SAMPLES = 4


def rgba(hex_color: str, opacity: float = 1.0) -> Color:
    """A colour from "#rrggbb" and an opacity."""
    digits = hex_color.lstrip("#")
    red, green, blue = (int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (red, green, blue, opacity)


def _box_distance(px, py, box: Box, radius: float):
    """Signed distance in pixels from points to a rounded rectangle; negative inside."""
    x, y, w, h = box
    half_w, half_h = w / 2, h / 2
    r = min(radius, half_w, half_h)
    qx = np.abs(px - (x + half_w)) - (half_w - r)
    qy = np.abs(py - (y + half_h)) - (half_h - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - r


def is_convex(points: np.ndarray) -> bool:
    """True if a polygon's corners all turn the same way."""
    edge = np.roll(points, -1, axis=0) - points
    cross = edge[:, 0] * np.roll(edge[:, 1], -1) - edge[:, 1] * np.roll(edge[:, 0], -1)
    return bool((cross >= -1e-9).all() or (cross <= 1e-9).all())


def _ring(distance: np.ndarray, width: float) -> np.ndarray:
    """The coverage of a line `width` wide that runs just inside a distance of zero."""
    return np.clip(0.5 - (np.abs(distance + width / 2) - width / 2), 0, 1)


def _outside_distance(p: np.ndarray, px, py) -> np.ndarray:
    """How far points lie outside the edge lines of a convex polygon."""
    following = np.roll(p, -1, axis=0)
    area = 0.5 * np.sum(p[:, 0] * following[:, 1] - following[:, 0] * p[:, 1])
    sign = 1.0 if area > 0 else -1.0
    edge = following - p
    length = np.hypot(edge[:, 0], edge[:, 1])
    length[length == 0] = 1.0
    normal_x, normal_y = sign * edge[:, 1] / length, -sign * edge[:, 0] / length
    return (
        normal_x[:, None, None] * (px[None] - p[:, 0, None, None])
        + normal_y[:, None, None] * (py[None] - p[:, 1, None, None])
    ).max(axis=0)


def _sampled_coverage(p: np.ndarray, left: int, top: int, rows: int, cols: int):
    """The share of each pixel of a box that a polygon covers, by sampling."""
    n = POLYGON_SAMPLES
    sample_x = (left + (np.arange(cols * n) + 0.5) / n)[None, :]
    sample_y = (top + (np.arange(rows * n) + 0.5) / n)[:, None]
    inside = np.zeros((rows * n, cols * n), dtype=bool)
    for (ax, ay), (bx, by) in zip(p, np.roll(p, -1, axis=0)):
        if ay == by:
            continue
        crosses = (ay > sample_y) != (by > sample_y)
        at_x = (bx - ax) * (sample_y - ay) / (by - ay) + ax
        inside ^= crosses & (sample_x < at_x)
    return inside.reshape(rows, n, cols, n).mean(axis=(1, 3)).astype(np.float32)


class Canvas:
    """An opaque RGB picture and the calls that draw on it.

    Args:
        width: pixels across.
        height: pixels down.
        background: a colour, or "#rrggbb".

    Attributes:
        width: pixels across.
        height: pixels down.
        px: the picture, float32, shape (height, width, 3).
        clip: (x0, y0, x1, y1), the box that drawing is limited to.
    """

    def __init__(
        self, width: int, height: int, background: Color | str = "#ffffff"
    ) -> None:
        """Make a canvas of one colour."""
        self.width, self.height = int(width), int(height)
        self.px = np.empty((self.height, self.width, 3), dtype=np.float32)
        self.clip = (0, 0, self.width, self.height)
        self.fill(background)

    def fill(self, color: Color | str) -> None:
        """Paint the whole canvas, clip or no clip, in one colour."""
        self.px[:] = (rgba(color) if isinstance(color, str) else color)[:3]

    def set_clip(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Limit later drawing to a box; pass the whole canvas to lift the limit."""
        self.clip = (
            max(0, int(x0)),
            max(0, int(y0)),
            min(self.width, int(x1)),
            min(self.height, int(y1)),
        )

    def to_uint8(self) -> np.ndarray:
        """The picture as uint8, shape (height, width, 3)."""
        return (np.clip(self.px, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def _visible(self, x0: int, y0: int, width: int, height: int):
        """The part of a box inside the clip as (source, target) slices, or None."""
        cx0, cy0, cx1, cy1 = self.clip
        left, top = max(cx0 - x0, 0), max(cy0 - y0, 0)
        right, bottom = min(cx1 - x0, width), min(cy1 - y0, height)
        if right <= left or bottom <= top:
            return None
        source = (slice(top, bottom), slice(left, right))
        target = (slice(y0 + top, y0 + bottom), slice(x0 + left, x0 + right))
        return source, target

    def blend(self, x0: int, y0: int, coverage: np.ndarray, color: Color) -> None:
        """Blend a colour through a coverage mask whose top-left pixel is (x0, y0)."""
        found = self._visible(x0, y0, coverage.shape[1], coverage.shape[0])
        if found is None:
            return
        source, target = found
        weight = coverage[source] * color[3]
        region = self.px[target]
        region += (np.asarray(color[:3], dtype=np.float32) - region) * weight[..., None]

    def blend_rgba(
        self, x0: int, y0: int, image: np.ndarray, opacity: float = 1.0
    ) -> None:
        """Blend a float RGBA picture, shape (height, width, 4), at (x0, y0)."""
        found = self._visible(x0, y0, image.shape[1], image.shape[0])
        if found is None:
            return
        source, target = found
        part = image[source]
        region = self.px[target]
        region += (part[..., :3] - region) * (part[..., 3:4] * opacity)

    def _grid(self, x0: float, y0: float, x1: float, y1: float):
        """The pixel box that covers a rectangle inside the clip.

        Returns:
            (left, top, xs, ys) with the pixel centres as an open grid, a row of
            xs and a column of ys; None if the box is empty.
        """
        cx0, cy0, cx1, cy1 = self.clip
        left, top = max(math.floor(x0), cx0), max(math.floor(y0), cy0)
        right, bottom = min(math.ceil(x1), cx1), min(math.ceil(y1), cy1)
        if right <= left or bottom <= top:
            return None
        xs = (np.arange(left, right, dtype=np.float32) + 0.5)[None, :]
        ys = (np.arange(top, bottom, dtype=np.float32) + 0.5)[:, None]
        return left, top, xs, ys

    def rect(self, box: Box, color: Color, *, radius: float = 0.0) -> None:
        """A filled rectangle (x, y, width, height), rounded if `radius` is above 0."""
        x, y, w, h = box
        if radius <= 0:
            self._plain_rect(box, color)
            return
        grid = self._grid(x - 1, y - 1, x + w + 1, y + h + 1)
        if grid is not None:
            left, top, px, py = grid
            distance = _box_distance(px, py, box, radius)
            self.blend(left, top, np.clip(0.5 - distance, 0, 1), color)

    def _plain_rect(self, box: Box, color: Color) -> None:
        """A square-cornered rectangle; coverage is an edge share along each axis."""
        x, y, w, h = box
        cx0, cy0, cx1, cy1 = self.clip
        left, top = max(math.floor(x), cx0), max(math.floor(y), cy0)
        right, bottom = min(math.ceil(x + w), cx1), min(math.ceil(y + h), cy1)
        if right <= left or bottom <= top:
            return
        cols = np.arange(left, right, dtype=np.float32)
        rows = np.arange(top, bottom, dtype=np.float32)
        along_x = np.clip(np.minimum(cols + 1, x + w) - np.maximum(cols, x), 0, 1)
        along_y = np.clip(np.minimum(rows + 1, y + h) - np.maximum(rows, y), 0, 1)
        if color[3] >= 1.0 and along_x.min() == 1 and along_y.min() == 1:
            self.px[top:bottom, left:right] = color[:3]
        else:
            self.blend(left, top, along_y[:, None] * along_x[None, :], color)

    def rect_outline(
        self, box: Box, color: Color, *, width: float = 1.0, radius: float = 0.0
    ) -> None:
        """The edge of a rectangle, a line `width` wide that runs just inside it."""
        x, y, w, h = box
        grid = self._grid(x - 2, y - 2, x + w + 2, y + h + 2)
        if grid is not None:
            left, top, px, py = grid
            distance = _box_distance(px, py, box, radius)
            self.blend(left, top, _ring(distance, width), color)

    def circle(
        self,
        center: Point,
        radius: float,
        *,
        fill: Color | None = None,
        stroke: Color | None = None,
        width: float = 1.0,
    ) -> None:
        """A disc, and a ring of `width` just inside its radius, each if given."""
        cx, cy = center
        pad = radius + width + 2
        grid = self._grid(cx - pad, cy - pad, cx + pad, cy + pad)
        if grid is None:
            return
        left, top, px, py = grid
        distance = np.hypot(px - cx, py - cy) - radius
        if fill is not None:
            self.blend(left, top, np.clip(0.5 - distance, 0, 1), fill)
        if stroke is not None:
            self.blend(left, top, _ring(distance, width), stroke)

    def convex(
        self,
        points,
        *,
        fill: Color | None = None,
        stroke: Color | None = None,
        width: float = 1.0,
    ) -> None:
        """A convex polygon, points in order either way.

        Exact inside, and close at the corners outside, where the distance to the
        nearest edge line is used and not the distance to the corner.
        """
        p = np.asarray(points, dtype=np.float64)
        pad = width + 2
        low, high = p.min(axis=0) - pad, p.max(axis=0) + pad
        grid = self._grid(low[0], low[1], high[0], high[1])
        if grid is None:
            return
        left, top, px, py = grid
        distance = _outside_distance(p, px, py)
        if fill is not None:
            self.blend(left, top, np.clip(0.5 - distance, 0, 1), fill)
        if stroke is not None:
            self.blend(left, top, _ring(distance, width), stroke)

    def polygon(self, points, fill: Color) -> None:
        """Any simple polygon, filled by sampling 4 by 4 points in each pixel.

        Slower than `convex`, and right for a shape that is not convex.
        """
        p = np.asarray(points, dtype=np.float64)
        grid = self._grid(
            p[:, 0].min() - 1, p[:, 1].min() - 1, p[:, 0].max() + 1, p[:, 1].max() + 1
        )
        if grid is not None:
            left, top, px, py = grid
            coverage = _sampled_coverage(p, left, top, py.shape[0], px.shape[1])
            self.blend(left, top, coverage, fill)

    def text(
        self, font: Font, string: str, origin: Point, *, size: int, color: Color
    ) -> float:
        """Draw a string from `origin` = (x, baseline); return the x after it."""
        pen, baseline = origin
        for char in string:
            glyph = font.glyph(char, size)
            if glyph.coverage.size:
                left = round(pen) + glyph.left
                self.blend(left, round(baseline) - glyph.top, glyph.coverage, color)
            pen += glyph.advance
        return pen

    def image_scaled(
        self,
        image: np.ndarray,
        origin: Point,
        scale: float | tuple[float, float],
        opacity: float = 1.0,
    ) -> None:
        """Draw a small RGBA uint8 picture magnified without smoothing.

        Args:
            image: shape (rows, columns, 4).
            origin: where its top-left corner goes.
            scale: pixels per source pixel, or a pair for x and y.
            opacity: overall opacity.
        """
        scale_x, scale_y = (scale, scale) if np.isscalar(scale) else scale
        rows, cols = image.shape[:2]
        left, top = round(origin[0]), round(origin[1])
        size = (max(1, round(cols * scale_x)), max(1, round(rows * scale_y)))
        found = self._visible(left, top, *size)
        if found is None:
            return
        (_, _), (target_y, target_x) = found
        from_x = np.arange(target_x.start, target_x.stop) - left
        from_y = np.arange(target_y.start, target_y.stop) - top
        from_x = np.minimum((from_x / scale_x).astype(int), cols - 1)
        from_y = np.minimum((from_y / scale_y).astype(int), rows - 1)
        part = image[from_y[:, None], from_x[None, :]].astype(np.float32) / 255.0
        self.blend_rgba(target_x.start, target_y.start, part, opacity)
