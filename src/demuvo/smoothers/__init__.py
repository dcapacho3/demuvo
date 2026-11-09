"""Smoothers - the code that improves a path after the planner made it.

A smoother is a class with two methods: `configure(costmap)`, called once with the
robot's global costmap, and `smooth(path)`, which takes a path of shape (N, 3) and
returns a path of the same kind. The world file names it as
`smoother = "module:Class"`, with keyword arguments in `smoother_params`. A robot
with no smoother follows the planner's path as it is. An empty path is never
passed to a smoother.
"""

from .base import Smoother

__all__ = ["Smoother"]
