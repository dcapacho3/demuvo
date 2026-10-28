"""Definition of a 2D world.

Including size, obstacles and robots, all from a TOML file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from robomath import Transform2D

from .models import MODELS, parse_params
from .shapes import Shape, parse_shape
from .verify import (
    WorldError,
    check_keys,
    inside,
    items,
    numbers,
    require_key,
    table,
    text,
)

FORMAT_VERSION = 1

# Nav2 uses 0.25 m and 0.25 rad; demuvo asks for a little more.
DEFAULT_GOAL_TOLERANCE = (0.1, 0.1)

__all__ = [
    "DEFAULT_GOAL_TOLERANCE",
    "FORMAT_VERSION",
    "Obstacle",
    "RobotSpec",
    "Task",
    "World",
    "WorldError",
    "load_world",
]


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


@dataclass(frozen=True)
class Task:
    """Something a robot is to do: go to a goal.

    Several tasks for one robot, in file order, are waypoints.

    Attributes:
        robot: name of the robot.
        goal: where it should end.
        goal_tolerance: distance in metres and angle in radians within which
            the robot counts as at its goal.
    """

    robot: str
    goal: Transform2D
    goal_tolerance: tuple[float, float] = DEFAULT_GOAL_TOLERANCE


@dataclass(frozen=True)
class World:
    """Everything in a world file.

    Attributes:
        size: width along x and height along y in metres; the border is a wall.
        seed: seed for every random number the run uses.
        obstacles: the obstacles, in file order.
        robots: the robots, in file order.
        tasks: what the robots are to do, in file order.
    """

    size: tuple[float, float]
    seed: int
    obstacles: tuple[Obstacle, ...]
    robots: tuple[RobotSpec, ...]
    tasks: tuple[Task, ...] = ()


def _parse_obstacle(item: dict, index: int, size) -> Obstacle:
    """Build and check one `[[obstacle]]`."""
    label = f"obstacle {index}"
    item = table(item, label)
    check_keys(item, {"shape", "pose"}, label)
    shape = parse_shape(require_key(item, "shape", label), label)
    if "pose" in item or not shape.pose_optional:
        pose = _pose(item, label)
    else:
        pose = Transform2D.identity()
    if not inside(shape.points(pose), size):
        raise WorldError(f"{label}: must lie inside the world")
    return Obstacle(pose, shape)


def _pose(item: dict, label: str, key: str = "pose") -> Transform2D:
    """Read a required (x, y, angle) as a pose."""
    return Transform2D.from_xyt(numbers(require_key(item, key, label), 3, label, key))


def _plugin(item: dict, key: str, label: str) -> str | None:
    """Read an optional "module:Class" string."""
    if key not in item:
        return None
    value = text(item, key, label)
    module, _, cls = value.partition(":")
    if not (module and cls):
        raise WorldError(f"{label}: '{key}' must be 'module:Class'")
    return value


def _plugin_params(item: dict, key: str, plugin: str, label: str) -> dict:
    """Read the keyword arguments of a plugin; they need the plugin's key."""
    if key not in item:
        return {}
    if plugin not in item:
        raise WorldError(f"{label}: '{key}' needs '{plugin}'")
    return dict(table(item[key], f"{label} {key}"))


def _parse_robot(item: dict, index: int, size, names: set[str]) -> RobotSpec:
    """Build and check one `[[robot]]`; `names` holds the names read so far."""
    item = table(item, f"robot {index}")
    name = text(item, "name", f"robot {index}")
    label = f"robot '{name}'"
    if name in names:
        raise WorldError(f"{label}: name used twice")
    names.add(name)
    check_keys(
        item,
        {
            "name",
            "model",
            "shape",
            "pose",
            "planner",
            "planner_params",
            "controller",
            "controller_params",
            "params",
        },
        label,
    )
    model = text(item, "model", label)
    if model not in MODELS:
        known = ", ".join(MODELS)
        raise WorldError(f"{label}: unknown model '{model}', expected one of {known}")
    params = parse_params(item.get("params", {}), model, label)
    shape = parse_shape(require_key(item, "shape", label), label, robot=True)
    pose = _pose(item, label)
    controller = _plugin(item, "controller", label)
    planner = _plugin(item, "planner", label)
    controller_params = _plugin_params(item, "controller_params", "controller", label)
    planner_params = _plugin_params(item, "planner_params", "planner", label)
    if not inside(shape.points(pose), size):
        raise WorldError(f"{label}: 'pose' must lie inside the world")
    return RobotSpec(
        name,
        model,
        shape,
        pose,
        controller,
        params,
        planner,
        controller_params,
        planner_params,
    )


def _parse_task(item: dict, index: int, size, robots: dict[str, RobotSpec]) -> Task:
    """Build and check one `[[task]]`; the robot must be in `robots`."""
    label = f"task {index}"
    item = table(item, label)
    check_keys(item, {"robot", "goal", "goal_tolerance"}, label)
    name = text(item, "robot", label)
    if name not in robots:
        raise WorldError(f"{label}: no robot named '{name}'")
    goal = _pose(item, label, "goal")
    if not inside(robots[name].shape.points(goal), size):
        raise WorldError(f"{label}: 'goal' must lie inside the world")
    tolerance = DEFAULT_GOAL_TOLERANCE
    if "goal_tolerance" in item:
        tolerance = numbers(item["goal_tolerance"], 2, label, "goal_tolerance", True)
    return Task(name, goal, tolerance)


def load_world(path: str | Path) -> World:
    """Read a world file and check it.

    Args:
        path: path of a TOML file in the format of `examples/worlds/demo.toml`.

    Returns:
        The world.

    Raises:
        WorldError: if anything in the file is wrong. The message says which
            object (for example "obstacle 2" or "robot 'r1'") and which key.
        OSError: if the file cannot be read.
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as error:
        raise WorldError(f"file: not valid TOML: {error}") from error

    version = require_key(data, "version", "file")
    if version != FORMAT_VERSION:
        raise WorldError(
            f"file: version {version} is not supported, expected {FORMAT_VERSION}"
        )
    check_keys(data, {"version", "world", "obstacle", "robot", "task"}, "file")

    world = table(require_key(data, "world", "file"), "world")
    check_keys(world, {"size", "seed"}, "world")
    size = numbers(require_key(world, "size", "world"), 2, "world", "size", True)
    seed = require_key(world, "seed", "world")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorldError("world: 'seed' must be an integer")

    obstacles = tuple(
        _parse_obstacle(item, i, size)
        for i, item in enumerate(items(data, "obstacle"), start=1)
    )

    names: set[str] = set()
    robots = tuple(
        _parse_robot(item, i, size, names)
        for i, item in enumerate(items(data, "robot"), start=1)
    )
    if not robots:
        raise WorldError("file: needs at least one [[robot]]")

    by_name = {r.name: r for r in robots}
    tasks = tuple(
        _parse_task(item, i, size, by_name)
        for i, item in enumerate(items(data, "task"), start=1)
    )
    return World(size, seed, obstacles, robots, tasks)
