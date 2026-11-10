"""What a robot senses and how it estimates its pose: the shared types."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from robomath import Transform2D

from ..models import Twist
from .raycast import Scene


@dataclass(frozen=True)
class StepTruth:
    """What the simulation knows about one robot after a step.

    Only sensors see it; an estimator never does.

    Attributes:
        time: simulated time at the end of the step, in seconds.
        pose: the true pose at the end of the step.
        command: the command the model followed, after `clip`. A wheel turns
            as commanded even if a wall stops the robot.
        dt: length of the step in seconds.
        scene: a call that gives what a ray can hit at the end of the step, or
            None if the simulation gives none. It builds the scene on the first
            call of a step, so a step that no sensor asks about costs nothing.
        owner: the index of this robot's body in that scene, so a sensor can
            leave it out; None if the robot has no body there.
    """

    time: float
    pose: Transform2D
    command: Twist
    dt: float
    scene: Callable[[], Scene] | None = None
    owner: int | None = None


@dataclass(frozen=True)
class Reading:
    """What a sensor reports at one step.

    Attributes:
        sensor: name of the sensor, for example "encoders".
        time: simulated time of the reading, in seconds.
        data: the numbers; their meaning belongs to the sensor.
    """

    sensor: str
    time: float
    data: np.ndarray


@dataclass(frozen=True)
class Estimate:
    """What an estimator believes.

    Attributes:
        pose: the estimated pose.
        cov: covariance of the pose, or None if the estimator gives none.
        map: a map the estimator built, or None.
    """

    pose: Transform2D
    cov: np.ndarray | None = None
    map: object | None = None


class Sensor(Protocol):
    """What every sensor offers. It holds its own random generator."""

    name: ClassVar[str]
    options: ClassVar[frozenset[str]]

    def read(self, truth: StepTruth) -> Reading:
        """The reading after a step."""


class Estimator(Protocol):
    """What every estimator offers.

    The estimator starts from the start pose of its robot. `needs` names the
    sensors it reads; the robot builds them.
    """

    name: ClassVar[str]
    needs: ClassVar[tuple[str, ...]]

    def update(
        self, command: Twist, dt: float, readings: Sequence[Reading]
    ) -> Estimate:
        """The estimate after a step.

        Args:
            command: the clipped command of the step.
            dt: length of the step in seconds.
            readings: one reading of each sensor in `needs`.
        """
