"""What a generator gives back: the world, and the grids it was drawn from.

This is the same for every style, so a caller that swaps one style for another
reads the same fields. The grids are kept because a window or a notebook will want
to draw the field under the obstacles, and because a test reads them to see what was
drawn without parsing a single shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..world import World
from .connectivity import Connectivity


@dataclass(frozen=True)
class Generated:
    """A world a generator drew, and how it got there.

    Attributes:
        world: the world, ready for `save_world`, the window or `Run`.
        field: the continuous field the style cut, shape (rows, columns), row 0 at
            the lowest y; None for a style that has no field.
        mask: the blocked cells the obstacles were traced from, the same shape.
        resolution: side of a cell of `field` and `mask` in metres.
        connectivity: the free-space report of the world that came back.
        attempt: which attempt this was, from 0; one more than it is were drawn.
    """

    world: World
    field: np.ndarray | None
    mask: np.ndarray
    resolution: float
    connectivity: Connectivity
    attempt: int = 0
