"""Reading and checking a world file."""

from __future__ import annotations

import tomllib
from pathlib import Path

from robomath import Transform2D

from ..models import MODELS, parse_params
from ..verify import (
    WorldError,
    check_keys,
    inside,
    items,
    number,
    numbers,
    require_key,
    table,
    text,
)
from .shapes import parse_shape
from .spec import (
    DEFAULT_CONTROL_PERIOD,
    DEFAULT_DT,
    DEFAULT_GOAL_TOLERANCE,
    Obstacle,
    RobotSpec,
    Task,
    World,
)

FORMAT_VERSION = 1


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
            "control_period",
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
    period = DEFAULT_CONTROL_PERIOD
    if "control_period" in item:
        period = number(item, "control_period", label)
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
        period,
    )


def _parse_task(item: dict, index: int, size, robots: dict[str, RobotSpec]) -> Task:
    """Build and check one `[[task]]`; the robot must be in `robots`."""
    label = f"task {index}"
    item = table(item, label)
    check_keys(item, {"robot", "goal", "goal_tolerance", "stop", "wait"}, label)
    name = text(item, "robot", label)
    if name not in robots:
        raise WorldError(f"{label}: no robot named '{name}'")
    goal = _pose(item, label, "goal")
    if not inside(robots[name].shape.points(goal), size):
        raise WorldError(f"{label}: 'goal' must lie inside the world")
    tolerance = DEFAULT_GOAL_TOLERANCE
    if "goal_tolerance" in item:
        tolerance = numbers(item["goal_tolerance"], 2, label, "goal_tolerance", True)
    stop = item.get("stop")
    if stop is not None and not isinstance(stop, bool):
        raise WorldError(f"{label}: 'stop' must be true or false")
    wait = number(item, "wait", label) if "wait" in item else 0.0
    return Task(name, goal, tolerance, stop, wait)


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
    check_keys(world, {"size", "seed", "dt"}, "world")
    size = numbers(require_key(world, "size", "world"), 2, "world", "size", True)
    seed = require_key(world, "seed", "world")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorldError("world: 'seed' must be an integer")
    dt = number(world, "dt", "world") if "dt" in world else DEFAULT_DT

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
    return World(size, seed, obstacles, robots, tasks, dt)
