"""What a controller sees and what it offers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from robomath import Transform2D

from ..models import Twist


@dataclass(frozen=True, eq=False)
class Observation:
    """What a controller knows at one step.

    Attributes:
        time: simulated time in seconds.
        dt: seconds since the controller last ran; its control period on
            the first run.
        pose: where the robot believes it is: the estimate of its estimator
        or the true pose if the robot's `control_pose` is "truth".
        path: the path to follow, shape (N, 3) of (x, y, angle); empty if the
            robot has no goal.
        goal: the goal, or None.
        goal_tolerance: distance in metres and angle in radians within which
            the robot counts as at its goal.
        path_changes: how many times the robot's path has been replaced.
        A controller that keeps state about the path
        starts over when this number changes.
    """

    time: float
    dt: float
    pose: Transform2D
    path: np.ndarray
    goal: Transform2D | None
    goal_tolerance: tuple[float, float]
    path_changes: int = 0


class Controller(Protocol):
    """What every controller class offers."""

    def command(self, obs: Observation) -> Twist:
        """The command for this step."""
