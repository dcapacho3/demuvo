"""A robot: its model, its sensors and what it believes about its pose."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..sensing import (
    ESTIMATORS,
    PAYLOAD,
    SENSORS,
    Estimate,
    Reading,
    StepTruth,
    stream,
)
from ..world import WorldError

if TYPE_CHECKING:
    from ..costmap import Costmap
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
        costmap: the robot's global costmap, or None. The estimate carries it
            unless the estimator built a map of its own.

    Attributes:
        spec: the robot's spec.
        model: its model.
        sensors: the sensors its estimator needs.
        payload: the sensors it carries because its `sensors` table names them,
            by name, such as a lidar.
        readings: the latest reading of each payload sensor, by name; empty until
            the first step.
        estimator: the estimator.
        estimate: the latest estimate; the start pose before the first step.
    """

    def __init__(
        self,
        spec: RobotSpec,
        model,
        seed: int,
        defaults: dict,
        costmap: Costmap | None = None,
    ) -> None:
        """Build the estimator and the sensors it needs."""
        self.spec = spec
        self.model = model
        self.estimator = ESTIMATORS[spec.estimator](model, spec.pose)
        self.sensors = [
            self._build(kind, kind, model, seed, defaults)
            for kind in self.estimator.needs
        ]
        self.payload = {
            name: self._build(settings.get("type", name), name, model, seed, defaults)
            for name, settings in spec.sensors.items()
            if settings.get("type", name) in PAYLOAD
        }
        self.readings: dict[str, Reading] = {}
        self.costmap = costmap
        self.estimate = Estimate(spec.pose, map=costmap)

    def _build(self, kind: str, name: str, model, seed: int, defaults: dict):
        """Build a sensor of this kind under this name; the robot's settings win."""
        own = {k: v for k, v in self.spec.sensors.get(name, {}).items() if k != "type"}
        try:
            rng = stream(seed, self.spec.name, name)
            return SENSORS[kind](model, rng, **{**defaults.get(kind, {}), **own})
        except ValueError as error:
            raise WorldError(
                f"robot '{self.spec.name}': sensors.{name}: {error}"
            ) from error

    def observe(self, truth: StepTruth) -> Estimate:
        """Sense the step and update the estimate."""
        for name, sensor in self.payload.items():
            self.readings[name] = sensor.read(truth)
        readings = [sensor.read(truth) for sensor in self.sensors]
        estimate = self.estimator.update(truth.command, truth.dt, readings)
        if estimate.map is None:
            estimate = replace(estimate, map=self.costmap)
        self.estimate = estimate
        return estimate
