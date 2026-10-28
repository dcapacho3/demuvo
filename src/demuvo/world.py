"""Definition of a 2D world.

Including size, obstacles and robots, all from a TOML file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .models import MODELS, parse_params
from .poses import Pose
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
    "World",
    "WorldError",
    "load_world",
]


@dataclass(frozen=True)
class Obstacle:
    """A fixed object.

    Attributes:
        pose: x, y coordinates of the origin and the angle in radians.
        shape: its shape.
    """

    pose: Pose
    shape: Shape


@dataclass(frozen=True)
class RobotSpec:
    """A robot as the world file describes it.

    Attributes:
        name: unique name.
        model: name of a kinematic model, for example "diff_drive".
        shape: its body, a shape that encloses an area.
        pose: where it starts: x, y and angle in radians.
        goal: where it should end: x, y and angle in radians, or None.
        controller: "module:Class" of the controller, or None.
        params: values that replace the model defaults, such as max_v.
        planner: "module:Class" of the planner, or None for the default.
        controller_params: keyword arguments for the controller class.
        planner_params: keyword arguments for the planner class.
        goal_tolerance: distance in metres and angle in radians within which
            the robot counts as at its goal.
    """

    name: str
    model: str
    shape: Shape
    pose: Pose
    goal: Pose | None = None
    controller: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    planner: str | None = None
    controller_params: dict = field(default_factory=dict)
    planner_params: dict = field(default_factory=dict)
    goal_tolerance: tuple[float, float] = DEFAULT_GOAL_TOLERANCE


@dataclass(frozen=True)
class World:
    """Everything in a world file.

    Attributes:
        size: width along x and height along y in metres; the border is a wall.
        seed: seed for every random number the run uses.
        obstacles: the obstacles, in file order.
        robots: the robots, in file order.
    """

    size: tuple[float, float]
    seed: int
    obstacles: tuple[Obstacle, ...]
    robots: tuple[RobotSpec, ...]


def _parse_obstacle(item: dict, index: int, size) -> Obstacle:
    """Build and check one `[[obstacle]]`."""
    label = f"obstacle {index}"
    item = table(item, label)
    check_keys(item, {"shape", "pose"}, label)
    shape = parse_shape(require_key(item, "shape", label), label)
    if "pose" in item or not shape.pose_optional:
        pose = numbers(require_key(item, "pose", label), 3, label, "pose")
    else:
        pose = (0.0, 0.0, 0.0)
    if not inside(shape.points(pose), size):
        raise WorldError(f"{label}: must lie inside the world")
    return Obstacle(pose, shape)


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
            "goal",
            "goal_tolerance",
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
    pose = numbers(require_key(item, "pose", label), 3, label, "pose")
    goal = numbers(item["goal"], 3, label, "goal") if "goal" in item else None
    if "goal_tolerance" in item and goal is None:
        raise WorldError(f"{label}: 'goal_tolerance' needs 'goal'")
    tolerance = DEFAULT_GOAL_TOLERANCE
    if "goal_tolerance" in item:
        tolerance = numbers(item["goal_tolerance"], 2, label, "goal_tolerance", True)
    controller = _plugin(item, "controller", label)
    planner = _plugin(item, "planner", label)
    controller_params = _plugin_params(item, "controller_params", "controller", label)
    planner_params = _plugin_params(item, "planner_params", "planner", label)
    for key, where in (("pose", pose), ("goal", goal)):
        if where is not None and not inside(shape.points(where), size):
            raise WorldError(f"{label}: '{key}' must lie inside the world")
    return RobotSpec(
        name,
        model,
        shape,
        pose,
        goal,
        controller,
        params,
        planner,
        controller_params,
        planner_params,
        tolerance,
    )


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
    check_keys(data, {"version", "world", "obstacle", "robot"}, "file")

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

    return World(size, seed, obstacles, robots)
