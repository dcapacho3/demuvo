"""Kinematic models.

To add a model: write a class like DiffDrive and add it to MODELS. The
world loader and the simulation need no other change.

Each model class includes:
    name: the word used in the world file
    step: the pose after `dt` seconds of a constant command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
from robomath import rplus_se2

from .shapes import Pose
from .verify import check_keys, number, table


def _to_matrix(pose: Pose) -> np.ndarray:
    """The 3x3 matrix of an (x, y, angle) pose."""
    x, y, angle = pose
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def _to_pose(matrix: np.ndarray) -> Pose:
    """The (x, y, angle) of a 3x3 pose matrix, with the angle in (-pi, pi]."""
    angle = math.atan2(matrix[1, 0], matrix[0, 0])
    return (float(matrix[0, 2]), float(matrix[1, 2]), angle)


@dataclass(frozen=True)
class DiffDrive:
    """Two wheels on one axle.

    The command is (v, w): forward speed in m/s and turn rate in rad/s.
    The step is exact for a constant command, the robot follows a circular arc
    of radius v / w, or a straight line when w is zero.

    Attributes:
        max_v: largest forward speed in m/s, in either direction.
        max_w: largest turn rate in rad/s, in either direction.
    """

    max_v: float = 1.0
    max_w: float = 1.0

    name: ClassVar[str] = "diff_drive"

    def clip(self, command: tuple[float, float]) -> tuple[float, float]:
        """Limit a command to the speeds this robot can reach."""
        v, w = command
        return (
            max(-self.max_v, min(self.max_v, v)),
            max(-self.max_w, min(self.max_w, w)),
        )

    def step(self, pose: Pose, command: tuple[float, float], dt: float) -> Pose:
        """The pose after dt seconds of command."""
        v, w = self.clip(command)
        moved = rplus_se2(_to_matrix(pose), np.array([v * dt, 0.0, w * dt]))
        return _to_pose(moved)


MODELS = {cls.name: cls for cls in (DiffDrive,)}


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
