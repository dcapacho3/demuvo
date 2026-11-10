"""The lidar - a ring of range readings from rays cast into the scene."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
from robomath import Transform2D

from .base import Reading, Scan, StepTruth
from .raycast import cast


@dataclass(frozen=True)
class LidarSettings:
    """The settings of a lidar.

    Attributes:
        beams: number of rays, a whole number of 1 or more.
        fov: the angle the beams cover, in radians, above 0 and up to 2 pi. With
            2 pi the beams are spread round the whole circle; with less, they
            run from one edge to the other, centred on the sensor's heading.
        max_range: the farthest it reads, in metres.
        min_range: the nearest it reads, in metres; a reading below it is `inf`.
        noise: standard deviation of the error added to each range, in metres.
        period: seconds between scans.
        mount: where the lidar sits on the robot, as (x, y, angle) in the robot's
            frame, in metres and radians, the same form as a pose.
    """

    beams: int = 360
    fov: float = math.tau
    max_range: float = 8.0
    min_range: float = 0.0
    noise: float = 0.0
    period: float = 0.1
    mount: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Check the settings."""
        if self.beams != int(self.beams) or self.beams < 1:
            raise ValueError(
                f"beams must be a whole number of 1 or more, got {self.beams!r}"
            )
        if not 0 < self.fov <= math.tau + 1e-9:
            raise ValueError(
                f"fov must be above 0 and at most 2 pi radians, got {self.fov!r}"
            )
        if not 0 <= self.min_range < self.max_range:
            raise ValueError(
                f"min_range must be 0 or more and below max_range, got "
                f"{self.min_range!r} and {self.max_range!r}"
            )
        if self.noise < 0 or self.period <= 0:
            raise ValueError("noise must not be negative and period must be positive")
        if len(self.mount) != 3 or not all(math.isfinite(v) for v in self.mount):
            raise ValueError(f"mount must be (x, y, angle), got {self.mount!r}")
        object.__setattr__(self, "mount", tuple(float(v) for v in self.mount))
        object.__setattr__(self, "beams", int(self.beams))  # a world file gives 4.0


class Lidar:
    """A lidar mounted on the robot.

    It casts rays and reads the distance to the first thing each meets, leaving
    out the robot's own body. A ray that meets nothing, or a reading outside
    `min_range` and `max_range`, gives `inf`. The lidar scans every `period`
    seconds, as a real one does, whatever the step length; between scans it
    reports the last scan again, with the time it was taken.

    The reading is the array of ranges, one per beam, in the order of `angles`.

    Args:
        model: the robot's model; the lidar does not use it.
        rng: this sensor's own random generator.
        **settings: the fields of `LidarSettings`.

    Attributes:
        angles: the angle of each beam in the sensor's frame, in radians.
        mount: where the sensor sits on the robot, as (x, y, angle).
        max_range: the farthest it reads, in metres.
        min_range: the nearest it reads, in metres.
    """

    name: ClassVar[str] = "lidar"
    options: ClassVar[frozenset[str]] = frozenset(
        field.name for field in fields(LidarSettings)
    )
    vectors: ClassVar[dict[str, int]] = {"mount": 3}

    def __init__(  # pylint: disable=unused-argument
        self, model, rng: np.random.Generator, **settings
    ) -> None:
        """Check the settings and lay out the beams."""
        self._settings = LidarSettings(**settings)
        half = self._settings.fov / 2
        if self._settings.beams == 1:
            self.angles = np.zeros(1)  # a single beam looks straight ahead
        elif self._settings.fov >= math.tau:
            self.angles = np.linspace(-math.pi, math.pi, self._settings.beams, False)
        else:
            self.angles = np.linspace(-half, half, self._settings.beams)
        self.mount = self._settings.mount
        self._mount_pose = Transform2D.from_xyt(self.mount)
        self.max_range = self._settings.max_range
        self.min_range = self._settings.min_range
        self._rng = rng
        self._reading: Reading | None = None
        self._due = 0.0

    def scan(self, reading: Reading) -> Scan:
        """A reading of this lidar as a `Scan`, with its beams and limits."""
        return Scan(
            reading.time,
            self.angles,
            reading.data,
            self.min_range,
            self.max_range,
            self.mount,
        )

    def read(self, truth: StepTruth) -> Reading:
        """The ranges of the latest scan, taking a new one if its time has come.

        Raises:
            ValueError: if the simulation gave no scene to cast into.
        """
        if truth.scene is None:
            raise ValueError("the lidar needs the scene, but the simulation gave none")
        settings = self._settings
        if self._reading is None or truth.time + 1e-9 >= self._due:
            x, y, yaw = (truth.pose @ self._mount_pose).xyt
            ranges = cast(
                np.array([x, y]),
                yaw + self.angles,
                truth.scene(),
                settings.max_range,
                truth.owner,
            )
            noisy = ranges + settings.noise * self._rng.standard_normal(len(ranges))
            keep = (
                np.isfinite(ranges)
                & (noisy >= settings.min_range)
                & (noisy <= settings.max_range)
            )
            self._reading = Reading(
                self.name, truth.time, np.where(keep, noisy, np.inf)
            )
            scans = math.floor(truth.time / settings.period + 1e-9) + 1
            self._due = scans * settings.period
        return self._reading
