"""The velocity command and what every model class offers."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
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
    """The commands a model can follow.

    The set is |vx| <= max_v, |vy| <= max_vy and |wz| <= max_w(vx).

    Every model bounds the turn rate as a function of the forward speed,
    min(cap, slope * |vx|). A diff drive's bound is flat (cap = max_w, no
    slope), and a car's is a wedge through the origin (no cap, slope =
    tan(max_steer) / wheelbase), so it cannot turn in place. Only a robot that
    can move sideways, a mecanum robot, has a max_vy above 0; for every other
    robot vy is not part of the set. A controller reads this from
    `Observation.limits` to command only what the robot will actually do.

    Attributes:
        max_v: largest forward speed in m/s, in either direction.
        cap: turn-rate bound in rad/s that does not depend on speed;
            infinity for a model without one.
        slope: turn rate per m/s of speed, in rad/m; infinity for a model
            whose bound does not grow with speed.
        max_vy: largest sideways speed in m/s, in either direction; 0 for a
            robot that cannot move sideways.
    """

    max_v: float
    cap: float = math.inf
    slope: float = math.inf
    max_vy: float = 0.0

    def __post_init__(self) -> None:
        """Refuse bounds that are not positive, and a cone that is unbounded."""
        for name in ("max_v", "cap", "slope"):
            value = getattr(self, name)
            if math.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} must be above 0, got {value}")
        if math.isinf(self.cap) and math.isinf(self.slope):
            raise ValueError("cap and slope cannot both be infinite")
        if not (math.isfinite(self.max_vy) and self.max_vy >= 0.0):
            raise ValueError(f"max_vy must be finite and 0 or more, got {self.max_vy}")

    def max_w(self, vx: float) -> float:
        """The largest |wz| the model follows at a forward speed of vx."""
        if math.isinf(self.slope):
            return self.cap
        return min(self.cap, self.slope * abs(vx))

    @property
    def turns_in_place(self) -> bool:
        """True if the robot can turn while standing still."""
        return self.max_w(0.0) > 0.0

    @property
    def strafes(self) -> bool:
        """True if the robot can move sideways."""
        return self.max_vy > 0.0

    def clip(self, command: Twist) -> Twist:
        """The command cut onto the set: (vx, vy) first, then wz at that vx.

        A robot that cannot strafe drops vy. Otherwise vx and vy are shrunk by
        one factor, the most either axis needs, so the direction of travel is
        kept. The axis that binds is set to its bound exactly, and with no vy
        this is the plain clamp of vx. The rest of the Twist is dropped.
        """
        vx = command.vx
        vy = command.vy if self.strafes else 0.0
        over_x = abs(vx) / self.max_v
        over_y = abs(vy) / self.max_vy if vy else 0.0
        if over_x > 1.0 or over_y > 1.0:
            if over_x >= over_y:
                vx, vy = math.copysign(self.max_v, vx), vy / over_x
            else:
                vx, vy = vx / over_y, math.copysign(self.max_vy, vy)
            # The axis that was shrunk can come out one rounding over its bound.
            vx = max(-self.max_v, min(self.max_v, vx))
            vy = max(-self.max_vy, min(self.max_vy, vy))
        bound = self.max_w(vx)
        return Twist(vx=vx, vy=vy, wz=max(-bound, min(bound, command.wz)))


def warn_ignored(model, command: Twist) -> None:
    """Warn if the command asks for a Twist field the model does not use.

    Args:
        model: a model, with `name` and `uses`.
        command: the command it was given.
    """
    ignored = [
        name for name in FIELDS if name not in model.uses and getattr(command, name)
    ]
    if ignored:
        warnings.warn(f"{model.name} ignores {', '.join(ignored)}")


def move(pose: Transform2D, command: Twist, dt: float) -> Transform2D:
    """The pose after dt of a constant command: an exact arc, or a straight line.

    The one pose update every model shares: the SE(2) exponential of the body
    twist held for dt. It trusts the command, so a model clips first; a model
    whose clip zeroes vy moves with no sideways part.
    """
    return pose.rplus(np.array([command.vx * dt, command.vy * dt, command.wz * dt]))


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


class RollingWheels:
    """What a model with two rolling rear or drive wheels shares.

    The wheels roll with no slip, so the model follows vx and wz and ignores the
    rest of a Twist (with one warning), and its wheels turn as the command says. A
    class that has these attributes, `name`, `track`, `wheel_radius` and `limits`,
    takes `clip`, `encoders` and `from_encoders` from it; its `step` is
    `move(pose, clip(command), dt)`.
    """

    uses: ClassVar[tuple[str, ...]] = ("vx", "wz")
    # Settings only the encoders read; a form shows them when the robot has some.
    encoder_only: ClassVar[tuple[str, ...]] = ("wheel_radius", "track")

    def clip(self, command: Twist) -> Twist:
        """Limit a command to what the robot can do.

        Warns once if the command asks for a Twist field the model ignores.
        """
        warn_ignored(self, command)
        return self.limits().clip(command)

    def encoders(self, command: Twist, dt: float) -> np.ndarray:
        """How far the left and right wheels turn in dt, in radians.

        Args:
            command: a command already limited by `clip`.
            dt: length of the step in seconds.
        """
        half = self.track / 2
        left = command.vx - command.wz * half
        right = command.vx + command.wz * half
        return np.array([left, right]) * dt / self.wheel_radius

    def from_encoders(self, readings: np.ndarray) -> np.ndarray:
        """The (x, y, angle) step that wheel increments (left, right) imply.

        It is the constant twist over the step, in the robot's own frame, in the
        form `Transform2D.rplus` takes.
        """
        left, right = readings
        return np.array(
            [
                self.wheel_radius * (left + right) / 2,
                0.0,
                self.wheel_radius * (right - left) / self.track,
            ]
        )
