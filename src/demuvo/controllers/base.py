"""What a controller sees and what it offers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from robomath import Transform2D

from ..costmap import Costmap
from ..models import Limits, Twist
from ..sensing import Scan


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
        scans: the latest scan of each range sensor the robot carries, by the
            sensor's name; empty for a robot with none, and before its first scan.
        costmap: the robot's local costmap, built from its scans, or None for a
            robot with none and before its first update.
        path_changes: how many times the robot's path has been replaced.
        A controller that keeps state about the path
        starts over when this number changes.
        limits: the commands the robot's model can follow (see `Limits`), or
            None when the caller does not know the model. A controller that
            reads it commands only what the robot will actually do; one that
            does not assumes a robot that can turn in place.
    """

    time: float
    dt: float
    pose: Transform2D
    path: np.ndarray
    goal: Transform2D | None
    goal_tolerance: tuple[float, float]
    path_changes: int = 0
    scans: Mapping[str, Scan] = field(default_factory=dict)
    costmap: Costmap | None = None
    limits: Limits | None = None


class Controller(Protocol):
    """What every controller class offers."""

    def command(self, obs: Observation) -> Twist:
        """The command for this step."""


def car_limits(obs: Observation) -> Limits | None:
    """The limits of a robot that cannot turn in place, or None.

    The built-in controllers drive a robot that can turn in place exactly as
    they did before limits existed, so a diff drive's behaviour does not
    change; only a robot whose limits forbid a turn on the spot, a car, is
    driven within them.
    """
    limits = obs.limits
    if limits is None or limits.turns_in_place:
        return None
    return limits
