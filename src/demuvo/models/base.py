"""The velocity command and what every model class offers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol

from robomath import Transform2D

# The fields of a Twist, in order. A tuple is much faster to walk than `fields()`,
# and `Twist` is built several times per robot and step.
FIELDS = ("vx", "vy", "vz", "wx", "wy", "wz")


@dataclass(frozen=True)
class Twist:
    """A velocity command, with the names and axes of a ROS Twist.

    Attributes:
        vx, vy, vz: speed along the robot's x, y and z axes in m/s.
        wx, wy, wz: turn rate about those axes in rad/s.

    Raises:
        ValueError: if a value is nan or infinite.
    """

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    def __post_init__(self) -> None:
        """Refuse nan and infinity, so they never become a speed."""
        for name in FIELDS:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {self}")


@dataclass(frozen=True)
class Limits:
    """The commands a model can follow: |vx| <= max_v and |wz| <= max_w(vx).

    Every model bounds the turn rate as a function of the forward speed,
    min(cap, slope * |vx|). A diff drive's bound is flat (cap = max_w, no
    slope), and a car's is a wedge through the origin (no cap, slope =
    tan(max_steer) / wheelbase), so it cannot turn in place. A controller
    reads this from `Observation.limits` to command only what the robot
    will actually do.

    Attributes:
        max_v: largest forward speed in m/s, in either direction.
        cap: turn-rate bound in rad/s that does not depend on speed;
            infinity for a model without one.
        slope: turn rate per m/s of speed, in rad/m; infinity for a model
            whose bound does not grow with speed.
    """

    max_v: float
    cap: float = math.inf
    slope: float = math.inf

    def __post_init__(self) -> None:
        """Refuse bounds that are not positive, and a cone that is unbounded."""
        for name in ("max_v", "cap", "slope"):
            value = getattr(self, name)
            if math.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} must be above 0, got {value}")
        if math.isinf(self.cap) and math.isinf(self.slope):
            raise ValueError("cap and slope cannot both be infinite")

    def max_w(self, vx: float) -> float:
        """The largest |wz| the model follows at a forward speed of vx."""
        if math.isinf(self.slope):
            return self.cap
        return min(self.cap, self.slope * abs(vx))

    @property
    def turns_in_place(self) -> bool:
        """True if the robot can turn while standing still."""
        return self.max_w(0.0) > 0.0

    def clip(self, command: Twist) -> Twist:
        """The command cut onto the set: vx first, then wz at that vx."""
        vx = max(-self.max_v, min(self.max_v, command.vx))
        bound = self.max_w(vx)
        return Twist(vx=vx, wz=max(-bound, min(bound, command.wz)))


class Model(Protocol):
    """What every model class offers; see `demuvo.models`."""

    name: ClassVar[str]
    uses: ClassVar[tuple[str, ...]]

    def limits(self) -> Limits:
        """The commands this robot can follow; see `Limits`."""

    def clip(self, command: Twist) -> Twist:
        """The command limited to what the robot can do."""

    def step(self, pose: Transform2D, command: Twist, dt: float) -> Transform2D:
        """The pose after dt seconds of command."""


class WheeledModel(Model, Protocol):
    """A model that also knows its wheels, so an estimator can read encoders."""

    def encoders(self, command: Twist, dt: float):
        """The wheel increments for a clipped command over dt, as an array."""

    def from_encoders(self, readings):
        """The (x, y, angle) step for wheel increments, for `Transform2D.rplus`."""
