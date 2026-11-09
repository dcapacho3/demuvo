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

A model that can also give its wheel increments has `encoders` and
`from_encoders` (see `WheeledModel`); only then can a robot use dead reckoning.
"""

from .base import Model, Twist, WheeledModel
from .diff_drive import DiffDrive
from .registry import MODELS, parse_params

__all__ = ["MODELS", "DiffDrive", "Model", "Twist", "WheeledModel", "parse_params"]
