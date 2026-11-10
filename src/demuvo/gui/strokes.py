"""Lines, dashes and rays - many points splatted at once.

A line is a run of points a little under a pixel apart. All the points of a call,
however many, are added into the picture together with bilinear weights, and only
the pixels they touch are blended. The work grows with the length of the lines,
not with the box that holds them, so a swarm with hundreds of paths and rays is
affordable.
"""

from __future__ import annotations

import math

import numpy as np

from .raster import Canvas, Color

# Distance between the points of a line, in pixels, and between those of a ray.
LINE_STEP = 0.6
RAY_STEP = 0.8
# A ray weighs a little more than one unit-wide stamp would.
RAY_GAIN = 1.25
# Stamps per point for a thin, medium and thick line.
THIN, MEDIUM = 1.6, 3.5


def _samples(canvas: Canvas, xs, ys, offsets, colors):
    """The splat positions inside the clip, for each offset, with their colours."""
    all_x = np.concatenate([xs + ox for ox, _ in offsets])
    all_y = np.concatenate([ys + oy for _, oy in offsets])
    rgb = None if colors is None else np.tile(colors, (len(offsets), 1))
    cx0, cy0, cx1, cy1 = canvas.clip
    keep = (
        (all_x >= cx0 + 1) & (all_x < cx1 - 1) & (all_y >= cy0 + 1) & (all_y < cy1 - 1)
    )
    return all_x[keep], all_y[keep], None if rgb is None else rgb[keep]


def _weights(canvas: Canvas, x, y, weight: float):
    """Bilinear weights of positions on the four pixels around each, summed per pixel.

    Returns:
        (flat pixel indices, the inverse map from the 2 by 2 splats to them, the
        weight of each splat, the total weight of each pixel)
    """
    x, y = x - 0.5, y - 0.5  # pixel centres are at +0.5
    fx, fy = np.floor(x), np.floor(y)
    ax, ay = (x - fx).astype(np.float32), (y - fy).astype(np.float32)
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    index = np.concatenate(
        [
            (iy + dy) * canvas.width + (ix + dx)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))
        ]
    )
    each = (
        np.concatenate([(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay])
        * weight
    )
    pixels, inverse = np.unique(index, return_inverse=True)
    total = np.bincount(inverse, weights=each, minlength=len(pixels)).astype(np.float32)
    return pixels, inverse, each, total


def _blend_pixels(canvas: Canvas, weights, color: Color, rgb) -> None:
    """Blend the summed splats into the touched pixels, inside the clip."""
    pixels, inverse, each, total = weights
    cx0, cy0, cx1, cy1 = canvas.clip
    px, py = pixels % canvas.width, pixels // canvas.width
    # A splat touches one pixel beyond its point, so the clip is checked again.
    inside = (px >= cx0) & (px < cx1) & (py >= cy0) & (py < cy1)
    opacity = np.where(inside, np.minimum(total, 1.0) * color[3], 0.0)
    flat = canvas.px.reshape(-1, 3)
    if rgb is None:
        mix = np.asarray(color[:3], dtype=np.float32)
    else:
        by_pixel = np.tile(rgb, (4, 1))
        channels = [
            np.bincount(inverse, weights=each * by_pixel[:, k], minlength=len(pixels))
            for k in range(3)
        ]
        mix = np.column_stack(channels).astype(np.float32)
        mix /= np.maximum(total, 1e-6)[:, None]
    flat[pixels] += (mix - flat[pixels]) * opacity.astype(np.float32)[:, None]


def stamp_points(
    canvas: Canvas,
    points: np.ndarray,
    weight: float,
    color: Color,
    *,
    offsets=((0.0, 0.0),),
    sample_rgb: np.ndarray | None = None,
) -> None:
    """Splat points with bilinear weights, once for every offset, and blend them.

    Args:
        canvas: where to draw.
        points: shape (n, 2), the x and y of each point, in pixels.
        weight: how much each splat adds to a pixel; a pixel's total is capped at 1.
        color: the colour, and its opacity.
        offsets: shifts that each point is also splatted at; they thicken a line.
        sample_rgb: a colour for each point, shape (n, 3), in place of `color`'s;
            where lines of different colours cross, the colours mix by weight.
    """
    if len(points) == 0:
        return
    x, y, rgb = _samples(canvas, points[:, 0], points[:, 1], offsets, sample_rgb)
    if x.size:
        _blend_pixels(canvas, _weights(canvas, x, y, weight), color, rgb)


def _along(points: np.ndarray, step: float):
    """Points every `step` pixels along a polyline, and how far along each one is."""
    pieces = np.diff(points, axis=0)
    length = np.hypot(pieces[:, 0], pieces[:, 1])
    total = float(length.sum())
    if total == 0:
        return points[:1, 0], points[:1, 1], np.zeros(1)
    reached = np.concatenate([[0.0], np.cumsum(length)])
    t = np.arange(0.0, total + step, step)
    t = t[t <= total]
    return np.interp(t, reached, points[:, 0]), np.interp(t, reached, points[:, 1]), t


def _brush(width: float):
    """The stamp offsets and weight that give a line a width in pixels."""
    count = 1 if width <= THIN else (4 if width <= MEDIUM else 7)
    if count == 1:
        return [(0.0, 0.0)], width * LINE_STEP
    r = (width - 1) / 2
    ring = [
        (
            r * math.cos(math.tau * k / (count - 1)),
            r * math.sin(math.tau * k / (count - 1)),
        )
        for k in range(count - 1)
    ]
    return [(0.0, 0.0), *ring], LINE_STEP * 1.1


def _dashed(xs, ys, t, dash):
    """The points of a line that fall in the dashes, for a (dash, gap) in pixels."""
    keep = (t % (dash[0] + dash[1])) < dash[0]
    return xs[keep], ys[keep]


def polyline(
    canvas: Canvas,
    points,
    width: float,
    color: Color,
    *,
    dash: tuple[float, float] | None = None,
) -> None:
    """A line through the points, `width` wide, dashed as (dash, gap) if given."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return
    xs, ys, t = _along(pts, LINE_STEP)
    if dash is not None:
        xs, ys = _dashed(xs, ys, t, dash)
    offsets, weight = _brush(width)
    stamp_points(canvas, np.column_stack([xs, ys]), weight, color, offsets=offsets)


def polylines(
    canvas: Canvas,
    paths: list[np.ndarray],
    colors: list[Color],
    width: float,
    *,
    dash: tuple[float, float] | None = None,
) -> None:
    """Many polylines, each in its colour, in one pass; this keeps a swarm cheap.

    The opacity is the mean of the colours' opacities.
    """
    xs_all, ys_all, rgb_all, opacities = [], [], [], []
    for points, color in zip(paths, colors):
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            continue
        xs, ys, t = _along(pts, LINE_STEP)
        if dash is not None:
            xs, ys = _dashed(xs, ys, t, dash)
        xs_all.append(xs)
        ys_all.append(ys)
        rgb_all.append(np.tile(np.asarray(color[:3], dtype=np.float32), (len(xs), 1)))
        opacities.append(color[3])
    if not xs_all:
        return
    offsets, weight = _brush(width)
    stamp_points(
        canvas,
        np.column_stack([np.concatenate(xs_all), np.concatenate(ys_all)]),
        weight,
        (0, 0, 0, float(np.mean(opacities))),
        offsets=offsets,
        sample_rgb=np.concatenate(rgb_all),
    )


def segments(canvas: Canvas, ends: np.ndarray, width: float, color: Color) -> None:
    """Many separate straight segments, such as lidar rays, in one go.

    Args:
        canvas: where to draw.
        ends: shape (n, 4), the x0, y0, x1, y1 of each segment.
        width: line width in pixels; a ray is at most one pixel wide.
        color: the colour and opacity.
    """
    x0, y0, x1, y1 = np.asarray(ends, dtype=np.float64).T
    length = np.hypot(x1 - x0, y1 - y0)
    counts = np.maximum((length / RAY_STEP).astype(int) + 1, 1)
    which = np.repeat(np.arange(len(x0)), counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    fraction = within / np.repeat(np.maximum(counts - 1, 1), counts)
    xs = x0[which] + (x1 - x0)[which] * fraction
    ys = y0[which] + (y1 - y0)[which] * fraction
    weight = min(width, 1.0) * RAY_STEP * RAY_GAIN
    stamp_points(canvas, np.column_stack([xs, ys]), weight, color)
