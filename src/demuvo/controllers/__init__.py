"""Controllers - the code that decides how a robot follows its path.

A controller is a class with one method, `command(obs)`, that returns the Twist
for this step. It sees only the `Observation`, never the simulation, so the
same class could run in a ROS node that gets /odom, /plan and /scan. The world
file names it as `controller = "module:Class"`; demuvo builds the class once for
each robot that names it, with the robot's `controller_params` as keyword
arguments.
"""

from .base import Controller, Observation
from .pure_pursuit import PurePursuit

__all__ = ["Controller", "Observation", "PurePursuit"]
