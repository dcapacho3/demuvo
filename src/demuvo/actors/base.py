"""What a motion offers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from robomath import Transform2D


class Motion(Protocol):
    """What every motion class offers.

    A motion is built from the actor's `motion_params` alone, then configured
    once with where the actor starts and a random stream that is its own. From
    then on `pose_at` is asked for the pose at a time, in any order and any
    number of times, and must give the same answer for the same time: the pose
    is a function of time, not of the calls before it. A motion that draws
    random numbers draws them in `configure` and keeps them.
    """

    def configure(self, start: Transform2D, rng: np.random.Generator) -> None:
        """Take the actor's start pose and its random stream."""

    def pose_at(self, time: float) -> Transform2D:
        """The actor's pose at this simulated time, in seconds, from 0."""
