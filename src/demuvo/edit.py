"""Editing a world - each function returns a new world and leaves the old one as it was.

A window that lets a user add a robot, drag it, set a goal or change a setting
calls these. So can a script that builds worlds. An edit that cannot work is
refused with a `WorldError` that names the robot and the setting, and no half
edited world comes out.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

from robomath import Transform2D

from . import options
from .models import MODELS, parse_params
from .nav.navigation import DEFAULT_PLANNER, Navigation
from .plugins import load_class
from .sim.collisions import Checker
from .sim.simulation import Simulation
from .verify import WorldError, inside, numbers
from .world import DEFAULT_GOAL_TOLERANCE, RobotSpec, World
from .world.load import _parse_robot, _parse_task

# The field of a robot that holds the settings of each part.
SETTINGS_FIELD = {
    "planner": "planner_params",
    "smoother": "smoother_params",
    "controller": "controller_params",
    "model": "params",
}
# What an object of each part must offer; the same check as when a world loads.
METHODS = {
    "planner": ("configure", "plan"),
    "smoother": ("configure", "smooth"),
    "controller": ("command",),
}
KIND_WORDS = {float: "a number", int: "a whole number", bool: "true or false"}


def _robot(world: World, name: str) -> RobotSpec:
    """The robot of that name, or a plain error."""
    for spec in world.robots:
        if spec.name == name:
            return spec
    raise WorldError(f"no robot named '{name}'")


def _replace_robot(world: World, spec: RobotSpec) -> World:
    """The world with this robot in place of the one of the same name."""
    robots = tuple(spec if r.name == spec.name else r for r in world.robots)
    return dataclasses.replace(world, robots=robots)


def _no_overlap(world: World, mover: str) -> None:
    """Raise unless every robot is clear of the obstacles, the border and the others.

    A pair is named with the robot being edited, `mover`, first.
    """
    poses = {r.name: r.pose for r in world.robots}
    for contact in Checker(world).check(poses):
        first, second = contact.a, contact.b
        if contact.b_moves and second == mover:
            first, second = second, first
        other = f"robot '{second}'" if contact.b_moves else second
        raise WorldError(f"robot '{first}' would overlap {other}")


def _starter(size: tuple[float, float]) -> RobotSpec:
    """The robot a new world starts with, in its middle."""
    return _parse_robot(
        {
            "name": "r1",
            "model": "diff_drive",
            "shape": {"name": "circle", "radius": 0.2},
            "pose": [size[0] / 2, size[1] / 2, 0.0],
            "planner": options.planners()["A*"],
            "controller": options.controllers()["Pure pursuit"],
        },
        1,
        (0.0, 0.0, size[0], size[1]),
        set(),
    )


def new_world(size: tuple[float, float] = (10.0, 6.0), seed: int = 1) -> World:
    """An empty world with one robot, r1, in the middle.

    A world needs at least one robot, so a new one starts with one.
    """
    return World(tuple(size), seed, (), (_starter(size),))


def validate(world: World) -> None:
    """Raise if the world cannot run: a robot overlaps, a part will not load.

    Raises:
        WorldError: naming the robot and the setting that is wrong.
        ValueError: if a robot starts inside another or a model setting is bad.
    """
    sim = Simulation(world)
    Navigation(world).check(sim.dt)


def add_robot(
    world: World,
    name: str,
    pose: Sequence[float],
    *,
    radius: float = 0.2,
    planner: str | None = None,
    controller: str | None = None,
) -> World:
    """Add a diff-drive robot with a disc body.

    Args:
        world: the world to start from.
        name: a name no other robot has.
        pose: (x, y, angle) where it starts.
        radius: radius of its body, in metres.
        planner: "module:Class"; A* if None.
        controller: "module:Class"; pure pursuit if None.

    Raises:
        WorldError: if the name is taken, the robot does not fit inside the
            world, or it would overlap something.
    """
    item = {
        "name": name,
        "model": "diff_drive",
        "shape": {"name": "circle", "radius": radius},
        "pose": list(pose),
        "planner": planner or options.planners()["A*"],
        "controller": controller or options.controllers()["Pure pursuit"],
    }
    spec = _parse_robot(
        item, len(world.robots) + 1, world.bounds, {r.name for r in world.robots}
    )
    added = dataclasses.replace(world, robots=(*world.robots, spec))
    _no_overlap(added, name)
    return added


def free_pose(world: World, radius: float = 0.2) -> tuple[float, float, float]:
    """A pose where a new robot of that radius fits, as near the middle as it can.

    It tries the centres of a one-metre grid, nearest the middle first.

    Raises:
        WorldError: if there is no such place.
    """
    x0, y0, x1, y1 = world.bounds
    middle = ((x0 + x1) / 2, (y0 + y1) / 2)
    centres = [
        (x0 + 0.5 + i, y0 + 0.5 + j)
        for i in range(int(x1 - x0))
        for j in range(int(y1 - y0))
    ]
    centres.sort(key=lambda c: math.dist(c, middle))
    for x, y in centres:
        try:
            add_robot(world, "new robot", (x, y, 0.0), radius=radius)
        except WorldError:
            continue
        return x, y, 0.0
    raise WorldError("there is no free place for another robot")


def remove_robot(world: World, name: str) -> World:
    """Remove a robot and its goals.

    Raises:
        WorldError: if there is no such robot, or it is the last one.
    """
    _robot(world, name)
    if len(world.robots) == 1:
        raise WorldError(f"robot '{name}' is the only robot; a world needs one")
    return dataclasses.replace(
        world,
        robots=tuple(r for r in world.robots if r.name != name),
        tasks=tuple(t for t in world.tasks if t.robot != name),
    )


def set_pose(world: World, name: str, pose: Sequence[float]) -> World:
    """Move a robot to a pose.

    Raises:
        WorldError: if the pose is outside the world or overlaps something.
    """
    spec = _robot(world, name)
    label = f"robot '{name}'"
    moved = Transform2D.from_xyt(numbers(list(pose), 3, label, "pose"))
    if not inside(spec.shape.points(moved), world.bounds):
        raise WorldError(f"{label}: 'pose' must lie inside the world")
    edited = _replace_robot(world, dataclasses.replace(spec, pose=moved))
    _no_overlap(edited, name)
    return edited


def set_goal(
    world: World,
    name: str,
    goal: Sequence[float],
    tolerance: Sequence[float] = DEFAULT_GOAL_TOLERANCE,
) -> World:
    """Give a robot one goal, in place of any it had.

    Args:
        world: the world to start from.
        name: the robot.
        goal: (x, y, angle).
        tolerance: distance in metres and angle in radians that count as there.

    Raises:
        WorldError: if the goal is outside the world.
    """
    robots = {r.name: r for r in world.robots}
    _robot(world, name)
    task = _parse_task(
        {"robot": name, "goal": list(goal), "goal_tolerance": list(tolerance)},
        len(world.tasks) + 1,
        world.bounds,
        robots,
    )
    kept = tuple(t for t in world.tasks if t.robot != name)
    return dataclasses.replace(world, tasks=(*kept, task))


def goal_of(world: World, name: str) -> tuple[float, float, float] | None:
    """The first goal of a robot as (x, y, angle), or None if it has none."""
    for task in world.tasks:
        if task.robot == name:
            return tuple(float(v) for v in task.goal.xyt)
    return None


def set_model(world: World, name: str, model: str) -> World:
    """Change the kinematic model of a robot; its model settings go back to defaults."""
    spec = _robot(world, name)
    if model not in MODELS:
        known = ", ".join(MODELS)
        raise WorldError(
            f"robot '{name}': unknown model '{model}', expected one of {known}"
        )
    return _replace_robot(world, dataclasses.replace(spec, model=model, params={}))


def set_part(world: World, name: str, part: str, path: str | None) -> World:
    """Choose the planner, smoother or controller of a robot.

    The new part starts with its default settings. None means the default planner,
    no smoother, or a robot that is driven from outside.

    Args:
        world: the world to start from.
        name: the robot.
        part: "planner", "smoother" or "controller".
        path: "module:Class", or None.

    Raises:
        WorldError: if the path is not "module:Class" or cannot be loaded.
    """
    spec = _robot(world, name)
    if part not in METHODS:
        raise WorldError(f"'{part}' is not a part; use planner, smoother or controller")
    if path is not None:
        module, _, cls = path.partition(":")
        if not (module and cls):
            raise WorldError(f"robot '{name}': {part} must be 'module:Class'")
        load_class(path, {}, name, part, METHODS[part])
    changes = {part: path, SETTINGS_FIELD[part]: {}}
    return _replace_robot(world, dataclasses.replace(spec, **changes))


def _checked(value, kind: type, name: str, part: str, key: str):
    """A setting value as its kind, or a plain error."""
    if kind is bool:
        good = isinstance(value, bool)
    elif kind is int:
        good = isinstance(value, int) and not isinstance(value, bool)
    else:
        good = isinstance(value, int | float) and not isinstance(value, bool)
        good = good and math.isfinite(value)
    if not good:
        raise WorldError(
            f"robot '{name}': {part} setting '{key}' must be "
            f"{KIND_WORDS[kind]}, got {value!r}"
        )
    return float(value) if kind is float else value


def set_settings(world: World, name: str, part: str, settings: dict) -> World:
    """Replace the settings of one part of a robot.

    Args:
        world: the world to start from.
        name: the robot.
        part: "planner", "smoother", "controller" or "model".
        settings: every setting to set, by name; the rest stay at their default.

    Raises:
        WorldError: if a setting does not exist, has the wrong kind, or the
            part refuses its value.
    """
    spec = _robot(world, name)
    if part not in SETTINGS_FIELD:
        raise WorldError(f"'{part}' is not a part of a robot")
    path = spec.model if part == "model" else getattr(spec, part)
    if part == "planner":
        path = path or DEFAULT_PLANNER
    if path is None:
        raise WorldError(f"robot '{name}' has no {part}, so it has nothing to set")
    known = (
        options.model_settings(path) if part == "model" else options.settings_of(path)
    )
    kinds = {s.name: s.kind for s in known}
    for key in settings:
        if key not in kinds:
            have = ", ".join(kinds) or "none"
            raise WorldError(
                f"robot '{name}': {part} has no setting '{key}'; it has {have}"
            )
    if part == "model":
        values = parse_params(settings, spec.model, f"robot '{name}'")
    else:
        values = {k: _checked(v, kinds[k], name, part, k) for k, v in settings.items()}
        load_class(path, values, name, part, METHODS[part])
    return _replace_robot(
        world, dataclasses.replace(spec, **{SETTINGS_FIELD[part]: values})
    )


def settings_of_robot(world: World, name: str, part: str) -> dict:
    """The settings now set for one part of a robot; a part at its defaults gives {}."""
    return dict(getattr(_robot(world, name), SETTINGS_FIELD[part]))


def set_control_period(world: World, name: str, period: float) -> World:
    """Set the seconds between two runs of a robot's controller.

    Raises:
        WorldError: if the period is not a positive number.
    """
    spec = _robot(world, name)
    (value,) = numbers([period], 1, f"robot '{name}'", "control_period", True)
    return _replace_robot(world, dataclasses.replace(spec, control_period=value))
