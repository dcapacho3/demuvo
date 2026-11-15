"""Actors: movable obstacles the global costmap is not built from, and how they move.

A motion is a class with two methods: `configure(start, rng)`, called once with
the actor's start pose and a random stream of its own, and `pose_at(time)`, which
gives the actor's pose at a simulated time in seconds. The pose is a function of
time alone, so a run repeats, a seek never drifts and nothing has to be replayed.
The world file names it as `motion = "module:Class"`; demuvo builds the class
once for each actor that names it, with the actor's `motion_params` as keyword
arguments.
"""

from .base import Motion
from .still import Still
from .waypoints import MODES, Waypoints

__all__ = ["MODES", "Motion", "Still", "Waypoints"]
