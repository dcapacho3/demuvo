"""The differential-drive model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
from robomath import Transform2D

from .base import Twist


@dataclass(frozen=True)
class DiffDrive:
    """Two wheels on one axle.

    It follows vx and wz. It cannot move sideways, so it ignores vy, and the
    rest of the Twist is out of its plane.
    The step is exact for a constant command, the robot follows a circular arc
    of radius v / w, or a straight line when w is zero.

    Attributes:
        max_v: largest forward speed in m/s, in either direction.
        max_w: largest turn rate in rad/s, in either direction.
    """

    max_v: float = 1.0
    max_w: float = 1.0

    name: ClassVar[str] = "diff_drive"
    uses: ClassVar[tuple[str, ...]] = ("vx", "wz")

    def clip(self, command: Twist) -> Twist:
        """Limit a command to the speeds this robot can reach.

        Warns once if the command asks for something the robot cannot do.
        """
        ignored = [
            f.name
            for f in fields(command)
            if f.name not in self.uses and getattr(command, f.name) != 0.0
        ]
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
