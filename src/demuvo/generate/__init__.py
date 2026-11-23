"""Worlds drawn from a seed, and the check that the free space of one is usable.

`generate_field` draws clutter from a smooth random field, `generate_rooms` cuts a
building into rooms joined by corridors, `generate_maze` lays a maze of corridors,
and `generate_caves` wears caverns out of noise and keeps the one cave the robot
can move about in. `check_connectivity` says
whether the free space of any world is one piece a robot can use, and every style
calls it. `place_pair` picks a start and a goal out of the free space a check
measured, so a map becomes a question to put to a planner.

A style is sized by the robot that will drive the map, never by a length written into
the code: the `Clearance` carries the robot, taken from the robots of the base world
when it does not name a radius of its own, and what a style draws follows from it. A
generator given no robot at all refuses to draw rather than guess one.

Nothing here is read by the simulation: a generated world is an ordinary `World` of
boxes and polygons, which saves, opens in the window and runs like any other.
"""

from .caves import CaveStyle, cavern, generate_caves
from .connectivity import (
    Clearance,
    Connectivity,
    GenerateError,
    check_connectivity,
    check_grid,
    clearance_for,
    lane,
    numbered,
    room,
    within,
)
from .field import FieldStyle, cut_mask, field_of, generate_field
from .maze import MazeStyle, generate_maze, maze_corridor
from .pairs import CONTROLLER, PLANNER, pair_world, place_pair
from .result import Generated
from .rooms import RoomsStyle, corridor_width, generate_rooms
from .shapes import grid_obstacles

# The style, its settings class and its function, by the name a person types.
STYLES = {
    "field": (FieldStyle, generate_field),
    "rooms": (RoomsStyle, generate_rooms),
    "caves": (CaveStyle, generate_caves),
    "maze": (MazeStyle, generate_maze),
}

__all__ = [
    "STYLES",
    "CONTROLLER",
    "PLANNER",
    "CaveStyle",
    "Clearance",
    "Connectivity",
    "FieldStyle",
    "GenerateError",
    "Generated",
    "MazeStyle",
    "RoomsStyle",
    "cavern",
    "check_connectivity",
    "check_grid",
    "clearance_for",
    "corridor_width",
    "cut_mask",
    "field_of",
    "generate_caves",
    "generate_field",
    "generate_maze",
    "generate_rooms",
    "grid_obstacles",
    "lane",
    "maze_corridor",
    "numbered",
    "pair_world",
    "place_pair",
    "room",
    "within",
]
