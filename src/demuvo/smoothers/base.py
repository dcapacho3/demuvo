"""What a smoother offers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..costmap import Costmap


class Smoother(Protocol):
    """What every smoother class offers.

    The navigation calls `configure` once, before the first path, with the robot's
    global costmap (None if the world turns the costmap off), and `smooth` after
    each plan.
    """

    def configure(self, costmap: Costmap | None) -> None:
        """Take the costmap the smoother checks paths against."""

    def smooth(self, path: np.ndarray) -> np.ndarray:
        """The path improved, shape (N, 3), with the same start and goal."""
