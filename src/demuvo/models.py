"""Kinematic models.

To add a model: write a class like DiffDrive and add it to MODELS. The
world loader and the simulation need no other change.

A command is a `Twist`, the six numbers of a ROS `cmd_vel`. A model follows the
parts it can use and ignores the rest, with one warning. The world is flat, so
today every model uses vx, vy and wz at most.

Each model class includes:
    name: the word used in the world file
    uses: the Twist fields it follows
    clip: the command limited to what the robot can do
    step: the pose after `dt` seconds of a constant command.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields
from typing import ClassVar, Protocol

import numpy as np
from robomath import Transform2D

from .verify import check_keys, number, table


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
    """What every model class offers; see the module docstring."""

    name: ClassVar[str]
    uses: ClassVar[tuple[str, ...]]

    def clip(self, command: Twist) -> Twist:
        """The command limited to what the robot can do."""

    def step(self, pose: Transform2D, command: Twist, dt: float) -> Transform2D:
        """The pose after dt seconds of command."""


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


# The one place that lists the models, a new model class needs only an entry here.
MODELS: dict[str, type[Model]] = {cls.name: cls for cls in (DiffDrive,)}


def parse_params(value, model: str, label: str) -> dict[str, float]:
    """Check the `params` table of a robot against the fields of its model.

    Args:
        value: the table read from the file.
        model: name of the robot's model, which must be in `MODELS`.
        label: the robot, for example "robot 'r1'".

    Returns:
        The parameters as keyword arguments for the model class.
    """
    data = table(value, f"{label} params")
    check_keys(data, {f.name for f in fields(MODELS[model])}, f"{label} params")
    return {key: number(data, key, f"{label} params") for key in data}
