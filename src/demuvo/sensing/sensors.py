"""The sensors: a perfect pose and wheel encoders."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from .base import Reading, StepTruth
from .lidar import Lidar


class PerfectPose:
    """The true pose, for an estimator that trusts it and for comparison."""

    name: ClassVar[str] = "perfect_pose"
    options: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, model, rng: np.random.Generator) -> None:
        """Take the same arguments as every sensor; it needs neither."""

    def read(self, truth: StepTruth) -> Reading:
        """The true (x, y, angle)."""
        return Reading(self.name, truth.time, np.array(truth.pose.xyt))


class Encoders:
    """The wheel encoders of a model.

    Each step the model says how far its wheels turned for the command it
    followed. The sensor adds the two errors of a real encoder - a random
    scale error on each wheel's increment, and the rounding to whole ticks.
    The rounding keeps the part of a tick it dropped and adds it to
    the next step, so the count stays within one tick of the true total.

    Args:
        model: a model that has `encoders` and `from_encoders`.
        rng: this sensor's own random generator.
        noise: standard deviation of the scale error of each wheel's increment,
            as a fraction of the increment (0.01 is one per cent). A wheel that
            does not turn has no error.
        ticks_per_rev: ticks in one turn of the wheel; 0 means no rounding.
    """

    name: ClassVar[str] = "encoders"
    options: ClassVar[frozenset[str]] = frozenset({"noise", "ticks_per_rev"})

    def __init__(
        self,
        model,
        rng: np.random.Generator,
        noise: float = 0.0,
        ticks_per_rev: float = 0,
    ) -> None:
        """Keep the model and the settings; start with no wheel movement."""
        if noise < 0 or ticks_per_rev < 0:
            raise ValueError("noise and ticks_per_rev must not be negative")
        self._model = model
        self._rng = rng
        self._noise = noise
        self._ticks = int(ticks_per_rev)
        self._angle: np.ndarray | None = None  # wheel angle so far, with errors
        self._counted: np.ndarray | None = None  # whole ticks reported so far

    def read(self, truth: StepTruth) -> Reading:
        """The wheel increments of this step, as the encoders report them."""
        ideal = np.asarray(self._model.encoders(truth.command, truth.dt), dtype=float)
        if self._angle is None:
            self._angle = np.zeros_like(ideal)
            self._counted = np.zeros_like(ideal)
        noisy = ideal * (1.0 + self._noise * self._rng.standard_normal(ideal.shape))
        if not self._ticks:
            return Reading(self.name, truth.time, noisy)
        self._angle += noisy
        counted = np.round(self._angle * self._ticks / math.tau)
        step = (counted - self._counted) * math.tau / self._ticks
        self._counted = counted
        return Reading(self.name, truth.time, step)


SENSORS = {cls.name: cls for cls in (PerfectPose, Encoders, Lidar)}

# Sensors a robot carries when its `sensors` table names them, under a name of its
# own if it likes, so a robot can have several. Every other sensor is built
# because the robot's estimator reads it.
PAYLOAD = frozenset({"lidar"})
