"""The cached layers of the scene - what is kept between frames, and how it is moved."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def slid(pixels: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """A picture moved by whole pixels; what the move uncovers is left unset."""
    height, width = pixels.shape[:2]
    moved = np.empty_like(pixels)
    sx0, sy0 = max(-dx, 0), max(-dy, 0)
    sx1, sy1 = width + min(-dx, 0), height + min(-dy, 0)
    moved[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = pixels[sy0:sy1, sx0:sx1]
    return moved


def uncovered(
    dx: int, dy: int, width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """The boxes (x0, y0, x1, y1) of a picture that a move by (dx, dy) leaves bare."""
    strips = []
    if dx > 0:
        strips.append((0, 0, dx, height))
    elif dx < 0:
        strips.append((width + dx, 0, width, height))
    left, right = max(dx, 0), width + min(dx, 0)
    if dy > 0:
        strips.append((left, 0, right, dy))
    elif dy < 0:
        strips.append((left, height + dy, right, height))
    return strips


@dataclass
class Cutout:
    """The cached pixels of the layers that change rarely."""

    pixels: np.ndarray | None = None
    key: object = None
    # The view the pixels stand for, and what else they were drawn from, so a view
    # that only moved can reuse them.
    drawn: tuple | None = None
    signature: tuple = ()
    checked: float = 0.0
    keep: tuple = ()
    world: object = None
