"""Planners: the code that finds a path from the start to the goal.

A planner is a class with one method, `plan(world, start, goal)`, that returns
the path as an array of shape (N, 3), one (x, y, angle) pose per row.
The world file names it as `planner = "module:Class"`;
demuvo builds the class once for each robot that names it, with the robot's
`planner_params` as keyword arguments.
"""

from .base import Planner
from .straight_line import StraightLine

__all__ = ["Planner", "StraightLine"]
