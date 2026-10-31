"""A top-down view of a simulation, drawn with pygame.

This module is not imported by `demuvo`, so the core runs without pygame.
Install the `view` extra of the package to use it.

The window and the GIF share one drawing routine: a frame is drawn at twice
the size and shrunk, which smooths the edges.
"""

from .gif import record
from .scene import View, outline
from .style import HUD, OBSTACLE, ROBOT_COLORS, SPEEDS, TRAIL
from .window import play

__all__ = [
    "HUD",
    "OBSTACLE",
    "ROBOT_COLORS",
    "SPEEDS",
    "TRAIL",
    "View",
    "outline",
    "play",
    "record",
]
