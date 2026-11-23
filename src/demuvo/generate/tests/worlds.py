"""The robot and the base world that the style tests of a generator share."""

from robomath import Transform2D

from demuvo.generate import Clearance
from demuvo.world import RobotSpec, World
from demuvo.world.shapes import Circle

SIZE = (20.0, 14.0)
# The robot a map is laid out for when no base world carries one.
ROBOT = Clearance(radius=0.25)


def a_robot(name="r1", x=1.5, y=1.5, radius=0.2):
    """A round robot at a place."""
    return RobotSpec(
        name, "diff_drive", Circle(radius), Transform2D.from_xyt([x, y, 0.0])
    )


def base_world(*robots, tasks=(), obstacles=(), size=SIZE):
    """A world for a map to be put into."""
    return World(
        size=size,
        seed=1,
        obstacles=tuple(obstacles),
        robots=tuple(robots) or (a_robot(),),
        tasks=tuple(tasks),
    )
