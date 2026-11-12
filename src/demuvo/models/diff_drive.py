"""The differential-drive model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from robomath import Transform2D

from ..plugins import check_positive
from .base import FIELDS, Twist

_USES = ("vx", "wz")
_UNUSED = tuple(name for name in FIELDS if name not in _USES)


@dataclass(frozen=True)
class DiffDrive:
    """Two wheels on one axle.

    It follows vx and wz. It cannot move sideways, so it ignores vy, and the
    rest of the Twist is out of its plane.
    The step is exact for a constant command, the robot follows a circular arc
    of radius v / w, or a straight line when w is zero.

    Its wheels turn as the command says, with no slip. `encoders` gives each
    wheel's angle increment over one step, and `from_encoders` gives back the
    twist that increment came from.

    Attributes:
        max_v: largest forward speed in m/s, in either direction.
        max_w: largest turn rate in rad/s, in either direction.
        wheel_radius: radius of both wheels in metres.
        track: distance between the two wheels in metres.
    """

    max_v: float = 1.0
    max_w: float = 1.0
    wheel_radius: float = 0.05
    track: float = 0.3

    name: ClassVar[str] = "diff_drive"
    uses: ClassVar[tuple[str, ...]] = _USES

    def __post_init__(self) -> None:
        """Refuse a wheel size that is not a positive number."""
        check_positive(self, "wheel_radius", "track")

    def clip(self, command: Twist) -> Twist:
        """Limit a command to the speeds this robot can reach.

        Warns once if the command asks for something the robot cannot do.
        """
        ignored = [name for name in _UNUSED if getattr(command, name) != 0.0]
        if ignored:
            warnings.warn(f"{self.name} ignores {', '.join(ignored)}")
        return Twist(
            vx=max(-self.max_v, min(self.max_v, command.vx)),
            wz=max(-self.max_w, min(self.max_w, command.wz)),
        )

    def step(self, pose: Transform2D, command: Twist, dt: float) -> Transform2D:
        """The pose after dt seconds of command."""
        command = self.clip(command)
        return pose.rplus(np.array([command.vx * dt, 0.0, command.wz * dt]))

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
