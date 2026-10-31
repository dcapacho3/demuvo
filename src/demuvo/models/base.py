"""The velocity command and what every model class offers."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar, Protocol

from robomath import Transform2D


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
        for field in fields(self):
            if not math.isfinite(getattr(self, field.name)):
                raise ValueError(f"{field.name} must be finite, got {self}")


class Model(Protocol):
    """What every model class offers; see `demuvo.models`."""

    name: ClassVar[str]
    uses: ClassVar[tuple[str, ...]]

    def clip(self, command: Twist) -> Twist:
        """The command limited to what the robot can do."""

    def step(self, pose: Transform2D, command: Twist, dt: float) -> Transform2D:
        """The pose after dt seconds of command."""
