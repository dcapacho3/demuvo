"""The estimators including the true pose, and dead reckoning from the encoders."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import ClassVar

from robomath import Transform2D

from ..models import Twist
from .base import Estimate, Reading


def _find(readings: Sequence[Reading], sensor: str) -> Reading:
    """The reading of the named sensor."""
    for reading in readings:
        if reading.sensor == sensor:
            return reading
    raise ValueError(f"no reading from '{sensor}'")


class GroundTruth:
    """Reports the true pose, which a perfect-pose sensor gives. The default."""

    name: ClassVar[str] = "ground_truth"
    needs: ClassVar[tuple[str, ...]] = ("perfect_pose",)

    def __init__(self, model, pose: Transform2D) -> None:
        """Take the same arguments as every estimator."""

    def update(  # pylint: disable=unused-argument
        self, command: Twist, dt: float, readings: Sequence[Reading]
    ) -> Estimate:
        """The pose the sensor read."""
        return Estimate(Transform2D.from_xyt(_find(readings, "perfect_pose").data))


class DeadReckoning:
    """Adds up the body movement that each step's wheel increments imply.

    It starts from the start pose and uses only the encoders, so its error grows
    with distance and does not shrink.

    Args:
        model: a model with `from_encoders`.
        pose: where the robot starts.
    """

    name: ClassVar[str] = "dead_reckoning"
    needs: ClassVar[tuple[str, ...]] = ("encoders",)

    def __init__(self, model, pose: Transform2D) -> None:
        """Start at `pose`."""
        self._model = model
        self._pose = pose

    def update(  # pylint: disable=unused-argument
        self, command: Twist, dt: float, readings: Sequence[Reading]
    ) -> Estimate:
        """The previous pose moved by the step the wheels report."""
        data = _find(readings, "encoders").data
        self._pose = self._pose.rplus(self._model.from_encoders(data))
        return Estimate(self._pose)


ESTIMATORS = {cls.name: cls for cls in (GroundTruth, DeadReckoning)}


def resolve_estimator(name: str, model: str, model_class: type, label: str) -> str:
    """The estimator a robot can use: `name`, or ground truth if it cannot.

    A model that has no wheel form (`encoders` and `from_encoders`) cannot feed
    an estimator that needs the encoders. The robot then uses the true pose and
    a warning says so.

    Args:
        name: the estimator the world file asks for.
        model: the name of the robot's model, for the message.
        model_class: the model class.
        label: the robot, for example "robot 'r1'".
    """
    wheels = hasattr(model_class, "encoders") and hasattr(model_class, "from_encoders")
    if "encoders" in ESTIMATORS[name].needs and not wheels:
        warnings.warn(
            f"{label}: model '{model}' has no encoder form, so '{name}' cannot "
            "run; the robot uses ground_truth instead"
        )
        return GroundTruth.name
    return name
