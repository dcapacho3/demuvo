"""Sensors, estimators and the robot that joins them.

A robot's true pose stays in the simulation. Its sensors read each step, and an
estimator turns the readings into the pose the robot believes. To add a sensor
or an estimator, write a class like those in `sensors` and `estimators` and add
it to `SENSORS` or `ESTIMATORS`.

The `Robot` that joins them is in `demuvo.sim`.
"""

from .base import Estimate, Estimator, Reading, Scan, Sensor, StepTruth
from .estimators import ESTIMATORS, DeadReckoning, GroundTruth, resolve_estimator
from .lidar import Lidar
from .raycast import Scene, cast
from .sensors import PAYLOAD, SENSORS, Encoders, PerfectPose
from .streams import stream

__all__ = [
    "ESTIMATORS",
    "PAYLOAD",
    "SENSORS",
    "DeadReckoning",
    "Encoders",
    "Estimate",
    "Estimator",
    "GroundTruth",
    "Lidar",
    "PerfectPose",
    "Reading",
    "Scan",
    "Scene",
    "Sensor",
    "StepTruth",
    "cast",
    "resolve_estimator",
    "stream",
]
