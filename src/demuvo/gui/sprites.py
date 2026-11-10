"""Sprites - small pictures kept as the list of their visible pixels.

A robot, a goal arrow or a label is the same small picture over and over, at many
places. Keeping only its visible pixels makes drawing cost what the sprite covers,
not its box, and many sprites go into the canvas in a few numpy passes.
"""

from __future__ import annotations

import math

import numpy as np

from .raster import Canvas

# Pixels fainter than this are left out of a sprite.
VISIBLE = 0.004


class Sprite:
    """A float RGBA picture, shape (height, width, 4), with straight alpha.

    Attributes:
        rgba: the picture.
        height: rows.
        width: columns.
        rgb: the colour of each visible pixel, shape (n, 3).
        alpha: the opacity of each visible pixel, shape (n,).
    """

    def __init__(self, rgba: np.ndarray) -> None:
        """Find the visible pixels."""
        self.rgba = rgba
        self.height, self.width = rgba.shape[:2]
        self._rows, self._cols = np.nonzero(rgba[..., 3] > VISIBLE)
        self.rgb = rgba[self._rows, self._cols, :3].astype(np.float32)
        self.alpha = rgba[self._rows, self._cols, 3].astype(np.float32)
        self._offsets: dict[int, np.ndarray] = {}

    def offsets(self, canvas_width: int) -> np.ndarray:
        """Flat positions of the visible pixels, on a canvas this wide."""
        if canvas_width not in self._offsets:
            self._offsets[canvas_width] = (
                self._rows.astype(np.int64) * canvas_width + self._cols
            )
        return self._offsets[canvas_width]


def _inside(canvas: Canvas, sprite: Sprite, x: int, y: int) -> bool:
    """True if the sprite at (x, y) lies wholly inside the clip."""
    x0, y0, x1, y1 = canvas.clip
    return x >= x0 and y >= y0 and x + sprite.width <= x1 and y + sprite.height <= y1


def _touches(canvas: Canvas, sprite: Sprite, x: int, y: int) -> bool:
    """True if the sprite at (x, y) overlaps the clip at all."""
    x0, y0, x1, y1 = canvas.clip
    return x < x1 and y < y1 and x + sprite.width > x0 and y + sprite.height > y0


def _waves(left, top, width, height) -> np.ndarray:
    """The batch of each sprite: one after the latest earlier sprite it overlaps."""
    right, bottom = left + width, top + height
    overlap = (
        (left[:, None] < right[None, :])
        & (left[None, :] < right[:, None])
        & (top[:, None] < bottom[None, :])
        & (top[None, :] < bottom[:, None])
    )
    wave = np.zeros(len(left), dtype=np.int64)
    for i in range(1, len(left)):
        earlier = overlap[i, :i]
        if earlier.any():
            wave[i] = wave[:i][earlier].max() + 1
    return wave


def _draw_batch(canvas: Canvas, placed, opacity: float) -> None:
    """Blend sprites that do not overlap, given as (sprite, x, y), in one pass."""
    index = np.concatenate(
        [s.offsets(canvas.width) + y * canvas.width + x for s, x, y in placed]
    )
    rgb = np.concatenate([s.rgb for s, _, _ in placed])
    alpha = np.concatenate([s.alpha for s, _, _ in placed]) * opacity
    flat = canvas.px.reshape(-1, 3)
    under = flat[index]
    flat[index] = under + (rgb - under) * alpha[:, None]


def draw_sprites(
    canvas: Canvas, sprites: list[Sprite], xs, ys, opacity: float = 1.0
) -> None:
    """Draw many sprites with their top-left corners at integer (xs, ys).

    Sprites wholly inside the clip go in batches. Two that overlap go in different
    batches, so the order of the list still decides which one is on top. A sprite
    partly outside the clip is drawn alone, cut to it, before the batches.
    """
    xs, ys = np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)
    whole = []
    for i, sprite in enumerate(sprites):
        if _inside(canvas, sprite, xs[i], ys[i]):
            whole.append(i)
        elif _touches(canvas, sprite, xs[i], ys[i]):
            canvas.blend_rgba(int(xs[i]), int(ys[i]), sprite.rgba, opacity)
    if not whole:
        return
    left, top = xs[whole], ys[whole]
    width = np.array([sprites[i].width for i in whole])
    height = np.array([sprites[i].height for i in whole])
    wave = _waves(left, top, width, height)
    for batch in range(int(wave.max()) + 1):
        members = [whole[j] for j in np.nonzero(wave == batch)[0]]
        _draw_batch(canvas, [(sprites[i], xs[i], ys[i]) for i in members], opacity)


def render_sprite(size: float | tuple[float, float], draw) -> Sprite:
    """A sprite from a drawing function.

    The function is called as `draw(canvas, cx, cy)`. It draws once on black and
    once on white; the difference between the two gives the opacity of each
    pixel, and the black picture divided by it gives the colour.

    Args:
        size: one number for a square about that many pixels across, made odd so
            its centre is the middle of a pixel; or (width, height) in whole pixels,
            used as they are, so an edge on the centre line stays crisp.
        draw: the drawing function, which draws at the centre (cx, cy).
    """
    if isinstance(size, tuple):
        width, height = (math.ceil(v) for v in size)
    else:
        width = height = math.ceil(size) | 1
    black, white = Canvas(width, height, "#000000"), Canvas(width, height, "#ffffff")
    for canvas in (black, white):
        draw(canvas, width / 2, height / 2)
    alpha = np.clip(1.0 - (white.px - black.px).mean(axis=2), 0.0, 1.0)
    color = black.px / np.maximum(alpha, 1e-4)[..., None]
    rgba = np.concatenate([np.clip(color, 0, 1), alpha[..., None]], axis=2)
    return Sprite(rgba.astype(np.float32))


class SpriteCache:
    """Sprites kept by key, so a robot is drawn once and then copied.

    Args:
        limit: the most sprites kept; the cache empties itself when it passes this.
    """

    def __init__(self, limit: int = 4000) -> None:
        """Start empty."""
        self.limit = limit
        self._sprites: dict[tuple, Sprite] = {}

    def get(self, key: tuple, size: float, draw) -> Sprite:
        """The sprite for a key, drawn by `draw` the first time."""
        if key not in self._sprites:
            if len(self._sprites) > self.limit:
                self._sprites.clear()
            self._sprites[key] = render_sprite(size, draw)
        return self._sprites[key]

    def clear(self) -> None:
        """Forget every sprite."""
        self._sprites.clear()

    def __len__(self) -> int:
        """How many sprites are kept."""
        return len(self._sprites)
