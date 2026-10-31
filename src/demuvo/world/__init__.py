"""Definition of a 2D world.

Including size, obstacles and robots, all from a TOML file.
"""

from ..verify import WorldError
from .load import FORMAT_VERSION, load_world
from .spec import (
    DEFAULT_CONTROL_PERIOD,
    DEFAULT_DT,
    DEFAULT_GOAL_TOLERANCE,
    Obstacle,
    RobotSpec,
    Task,
    World,
)

__all__ = [
    "DEFAULT_CONTROL_PERIOD",
    "DEFAULT_DT",
    "DEFAULT_GOAL_TOLERANCE",
    "FORMAT_VERSION",
    "Obstacle",
    "RobotSpec",
    "Task",
    "World",
    "WorldError",
    "load_world",
]
