"""The objects a world file describes."""

from __future__ import annotations

from dataclasses import dataclass, field

from robomath import Transform2D

from .shapes import Shape

DEFAULT_GOAL_TOLERANCE = (0.1, 0.1)
DEFAULT_DT = 0.01
DEFAULT_CONTROL_PERIOD = 0.05


@dataclass(frozen=True)
class Obstacle:
    """A fixed object.

    Attributes:
        pose: where its origin is and which way it points.
        shape: its shape.
    """

    pose: Transform2D
    shape: Shape


@dataclass(frozen=True)
class RobotSpec:
    """A robot as the world file describes it.

    Attributes:
        name: unique name.
        model: name of a kinematic model, for example "diff_drive".
        shape: its body, a shape that encloses an area.
        pose: where it starts.
        controller: "module:Class" of the controller, or None.
        params: values that replace the model defaults, such as max_v.
        planner: "module:Class" of the planner, or None for the default.
        controller_params: keyword arguments for the controller class.
        planner_params: keyword arguments for the planner class.
        control_period: seconds between two runs of the controller; at least
            the simulation step. The command stays in force between runs.
    """

    name: str
    model: str
    shape: Shape
    pose: Transform2D
    controller: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    planner: str | None = None
    controller_params: dict = field(default_factory=dict)
    planner_params: dict = field(default_factory=dict)
    control_period: float = DEFAULT_CONTROL_PERIOD


@dataclass(frozen=True)
class Task:
    """Something a robot is to do: go to a goal.

    Several tasks for one robot, in file order, are waypoints.

    Attributes:
        robot: name of the robot.
        goal: where it should end.
        goal_tolerance: distance in metres and angle in radians within which
            the robot counts as at its goal.
        stop: True if the robot must also be standing still to finish the
            task. None means yes for the robot's last task and for any task
            with a `wait`, no for the others.
        wait: seconds the robot must hold the goal before the task is done.
    """

    robot: str
    goal: Transform2D
    goal_tolerance: tuple[float, float] = DEFAULT_GOAL_TOLERANCE
    stop: bool | None = None
    wait: float = 0.0


@dataclass(frozen=True)
class World:
    """Everything in a world file.

    Attributes:
        size: width along x and height along y in metres; the border is a wall.
        seed: seed for every random number the run uses.
        obstacles: the obstacles, in file order.
        robots: the robots, in file order.
        tasks: what the robots are to do, in file order.
        dt: length of a simulation step in seconds.
    """

    size: tuple[float, float]
    seed: int
    obstacles: tuple[Obstacle, ...]
    robots: tuple[RobotSpec, ...]
    tasks: tuple[Task, ...] = ()
    dt: float = DEFAULT_DT
