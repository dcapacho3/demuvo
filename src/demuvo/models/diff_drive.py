"""The differential-drive model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from robomath import Transform2D

from ..plugins import check_positive
from .base import Limits, RollingWheels, Twist, move


@dataclass(frozen=True)
class DiffDrive(RollingWheels):
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

    def __post_init__(self) -> None:
        """Refuse a wheel size that is not a positive number."""
        check_positive(self, "wheel_radius", "track")

    def limits(self) -> Limits:
        """The feasible commands: a flat bound, the same max_w at every speed."""
        return Limits(max_v=self.max_v, cap=self.max_w)

    def step(self, pose: Transform2D, command: Twist, dt: float) -> Transform2D:
        """The pose after dt seconds of command."""
        return move(pose, self.clip(command), dt)
