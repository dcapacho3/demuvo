"""A robot: its model, its sensors and what it believes about its pose."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sensing import ESTIMATORS, SENSORS, Estimate, StepTruth, stream

if TYPE_CHECKING:
    from ..world import RobotSpec


class Robot:
    """The side of a robot that the simulation does not own.

    The simulation holds the true pose. The robot senses the step and updates
    its estimate; a controller sees the estimate only.

    Args:
        spec: the robot as the world file describes it.
        model: its kinematic model.
        seed: the seed of the world, which sets every random stream.
        defaults: the sensor settings of the world, by sensor name; the robot's
            own settings replace them key by key.

    Attributes:
        spec: the robot's spec.
        model: its model.
        sensors: the sensors its estimator needs.
        estimator: the estimator.
        estimate: the latest estimate; the start pose before the first step.
    """

    def __init__(self, spec: RobotSpec, model, seed: int, defaults: dict) -> None:
        """Build the estimator and the sensors it needs."""
        self.spec = spec
        self.model = model
        self.estimator = ESTIMATORS[spec.estimator](model, spec.pose)
        self.sensors = []
        for kind in self.estimator.needs:
            options = {**defaults.get(kind, {}), **spec.sensors.get(kind, {})}
            rng = stream(seed, spec.name, kind)
            self.sensors.append(SENSORS[kind](model, rng, **options))
        self.estimate = Estimate(spec.pose)

    def observe(self, truth: StepTruth) -> Estimate:
        """Sense the step and update the estimate."""
        readings = [sensor.read(truth) for sensor in self.sensors]
        self.estimate = self.estimator.update(truth.command, truth.dt, readings)
        return self.estimate
