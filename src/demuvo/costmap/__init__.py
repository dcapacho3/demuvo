"""The costmap - what a planner searches.

A global costmap has a static layer (the map, generated from the obstacles or
read from a `map.yaml`) and an inflation layer (cost spread out by the
robot's radius).
"""

from .build import build_costmaps, inscribed_radius
from .grid import (
    FREE,
    INSCRIBED,
    LETHAL,
    MAX_NON_OBSTACLE,
    NO_INFORMATION,
    Costmap,
)

__all__ = [
    "FREE",
    "INSCRIBED",
    "LETHAL",
    "MAX_NON_OBSTACLE",
    "NO_INFORMATION",
    "Costmap",
    "build_costmaps",
    "inscribed_radius",
]
