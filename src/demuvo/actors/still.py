"""The motion of an actor that names none: it stands where it was put."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robomath import Transform2D


@dataclass
class Still:
    """Stays at the start pose for good.

    This is what an actor without a `motion` gets: an obstacle that the global
    costmap is not built from.
    """

    _start: Transform2D | None = None

    def configure(self, start: Transform2D, rng: np.random.Generator) -> None:
        """Keep the start pose; the random stream is not used."""
        self._start = start

    def pose_at(self, time: float) -> Transform2D:
        """The start pose, whatever the time."""
        if self._start is None:
            raise RuntimeError("configure must be called before pose_at")
        return self._start
