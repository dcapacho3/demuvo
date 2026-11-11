"""Editing a world - each function returns a new world and leaves the old one as it was.

A window that lets a user add a robot, drag it, set a goal or change a setting
calls these. So can a script that builds worlds. An edit that cannot work is
refused with a `WorldError` that names the robot and the setting, and no half
edited world comes out.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from pathlib import Path

from robomath import Transform2D

from . import options
from .models import MODELS, parse_params
from .nav.navigation import DEFAULT_PLANNER, Navigation
from .plugins import check_positive, load_class
from .sensing import ESTIMATORS, SENSORS, resolve_estimator
from .sim.collisions import Checker
from .sim.simulation import Simulation
from .verify import WorldError, inside, numbers
from .world import DEFAULT_GOAL_TOLERANCE, RobotSpec, World
from .world.load import (
    _parse_obstacle,
    _parse_robot,
    _parse_sensors,
    _parse_task,
)
from .world.shapes import parse_shape
from .world.spec import LOCAL_POSITIVE

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


def _obstacle(world: World, index: int):
    """The obstacle at a 0-based place, or a plain error."""
    if not 0 <= index < len(world.obstacles):
        raise WorldError(f"no obstacle number {index + 1}")
    return world.obstacles[index]


def _with_obstacles(world: World, obstacles: tuple) -> World:
    """The world with these obstacles, refused if a robot would touch one."""
    edited = dataclasses.replace(world, obstacles=obstacles)
    if edited.robots:
        _no_overlap(edited, "")
    return edited


def add_obstacle(
    world: World, kind: str, pose: Sequence[float], size: Sequence[float]
) -> World:
    """Add a box or a circle.

    Args:
        world: the world to start from.
        kind: "box" or "circle".
        pose: (x, y, angle) of its centre.
        size: (length, width) of a box, or (radius,) of a circle.

    Raises:
        WorldError: if the kind is unknown, or the obstacle would leave the world
            or touch a robot.
    """
    if kind == "box":
        shape = {"name": "box", "size": list(size)}
    elif kind == "circle":
        shape = {"name": "circle", "radius": size[0]}
    else:
        raise WorldError(f"unknown obstacle '{kind}', expected box or circle")
    made = _parse_obstacle(
        {"shape": shape, "pose": list(pose)}, len(world.obstacles) + 1, world.bounds
    )
    return _with_obstacles(world, (*world.obstacles, made))


def move_obstacle(world: World, index: int, pose: Sequence[float]) -> World:
    """Move the obstacle at a 0-based place to a pose."""
    old = _obstacle(world, index)
    label = f"obstacle {index + 1}"
    moved = Transform2D.from_xyt(numbers(list(pose), 3, label, "pose"))
    if not inside(old.shape.points(moved), world.bounds):
        raise WorldError(f"{label}: must lie inside the world")
    items = list(world.obstacles)
    items[index] = dataclasses.replace(old, pose=moved)
    return _with_obstacles(world, tuple(items))


def resize_obstacle(world: World, index: int, size: Sequence[float]) -> World:
    """Give the obstacle at a 0-based place a new size, keeping its pose."""
    old = _obstacle(world, index)
    if old.shape.name not in ("box", "circle"):
        raise WorldError(f"obstacle {index + 1}: only a box or a circle has a size")
    shape = (
        {"name": "circle", "radius": size[0]}
        if old.shape.name == "circle"
        else {"name": "box", "size": list(size)}
    )
    made = _parse_obstacle(
        {"shape": shape, "pose": [float(v) for v in old.pose.xyt]},
        index + 1,
        world.bounds,
    )
    items = list(world.obstacles)
    items[index] = made
    return _with_obstacles(world, tuple(items))


def remove_obstacle(world: World, index: int) -> World:
    """Remove the obstacle at a 0-based place."""
    _obstacle(world, index)
    kept = tuple(o for i, o in enumerate(world.obstacles) if i != index)
    return dataclasses.replace(world, obstacles=kept)


def set_size(world: World, size: Sequence[float]) -> World:
    """Give the world a new width and height, keeping its origin.

    Raises:
        WorldError: if a robot, a goal or an obstacle would fall outside.
    """
    width, height = numbers(list(size), 2, "world", "size", True)
    return _fits(dataclasses.replace(world, size=(width, height)))


def set_seed(world: World, seed: int) -> World:
    """Give the world a seed, which sets every random number a run uses."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorldError("world: 'seed' must be an integer")
    return dataclasses.replace(world, seed=seed)


def set_step(world: World, dt: float) -> World:
    """Give the world a step length in seconds.

    Raises:
        WorldError: if it is not a positive number.
    """
    (value,) = numbers([dt], 1, "world", "dt", True)
    return dataclasses.replace(world, dt=value)


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


def add_outline(world: World, kind: str, vertices: Sequence[Sequence[float]]) -> World:
    """Add a polygon or a line that follows the given points, in world metres.

    Args:
        world: the world to start from.
        kind: "polygon" (closed, at least three points, edges that do not cross)
            or "linestring" (open, at least two).
        vertices: the corners, in order.

    Raises:
        WorldError: if the kind or the points are wrong, or the shape would leave
            the world or touch a robot.
    """
    if kind not in ("polygon", "linestring"):
        raise WorldError(f"unknown outline '{kind}', expected polygon or linestring")
    shape = {"name": kind, "vertices": [list(map(float, v)) for v in vertices]}
    made = _parse_obstacle({"shape": shape}, len(world.obstacles) + 1, world.bounds)
    return _with_obstacles(world, (*world.obstacles, made))


def set_robot_shape(world: World, name: str, kind: str, size: Sequence[float]) -> World:
    """Give a robot a disc or a box for a body.

    Args:
        world: the world to start from.
        name: the robot.
        kind: "circle" or "box".
        size: (radius,) of a circle, or (length, width) of a box.

    Raises:
        WorldError: if the body would leave the world, cover a goal or touch
            something.
    """
    spec = _robot(world, name)
    label = f"robot '{name}'"
    if kind == "circle":
        table = {"name": "circle", "radius": size[0]}
    elif kind == "box":
        table = {"name": "box", "size": list(size)}
    else:
        raise WorldError(f"{label}: unknown body '{kind}', expected circle or box")
    shape = parse_shape(table, label, robot=True)
    if not inside(shape.points(spec.pose), world.bounds):
        raise WorldError(f"{label}: the body would leave the world")
    for task in world.tasks:
        if task.robot == name and not inside(shape.points(task.goal), world.bounds):
            raise WorldError(f"{label}: the body would not fit at its goal")
    edited = _replace_robot(world, dataclasses.replace(spec, shape=shape))
    _no_overlap(edited, name)
    return edited


def set_estimator(world: World, name: str, estimator: str) -> World:
    """Choose where a robot's controller gets its pose from.

    Raises:
        WorldError: if the estimator is unknown or the robot's model cannot run it.
    """
    spec = _robot(world, name)
    label = f"robot '{name}'"
    if estimator not in ESTIMATORS:
        known = ", ".join(ESTIMATORS)
        raise WorldError(f"{label}: unknown estimator '{estimator}', expected {known}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        usable = resolve_estimator(estimator, spec.model, MODELS[spec.model], label)
    if usable != estimator:
        raise WorldError(f"{label}: model '{spec.model}' cannot run '{estimator}'")
    return _replace_robot(world, dataclasses.replace(spec, estimator=estimator))


def set_control_pose(world: World, name: str, which: str) -> World:
    """Choose the pose the controller sees: "estimate" or "truth"."""
    spec = _robot(world, name)
    if which not in ("estimate", "truth"):
        raise WorldError(f"robot '{name}': 'control_pose' must be estimate or truth")
    return _replace_robot(world, dataclasses.replace(spec, control_pose=which))


def set_sensor(
    world: World, robot: str | None, sensor: str, settings: dict | None
) -> World:
    """Set, change or remove a sensor of a robot, or a default of every robot.

    Args:
        world: the world to start from.
        robot: the robot, or None for the settings every robot starts from.
        sensor: the sensor name, such as "encoders" or "lidar". A robot may carry
            several lidars under names of its own, whose settings say
            `type = "lidar"`.
        settings: its settings by name (a lidar's `mount` is three numbers), or
            None to remove it.

    Raises:
        WorldError: if the sensor or a setting is unknown or out of range.
    """
    label = "world" if robot is None else f"robot '{robot}'"
    spec = None if robot is None else _robot(world, robot)
    kept = dict(world.sensors if spec is None else spec.sensors)
    if settings is None:
        kept.pop(sensor, None)
    else:
        kept[sensor] = dict(settings)
        kind = settings.get("type", sensor)
        if kind not in SENSORS:
            raise WorldError(f"{label}: unknown sensor '{kind}'")
    checked = _parse_sensors(kept, label, named=spec is not None)
    if spec is None:
        return dataclasses.replace(world, sensors=checked)
    return _replace_robot(world, dataclasses.replace(spec, sensors=checked))


def _task(world: World, index: int):
    """The task at a 0-based place, or a plain error."""
    if not 0 <= index < len(world.tasks):
        raise WorldError(f"no goal number {index + 1}")
    return world.tasks[index]


# Passed for `stop` to leave it as it is; None means the default rule.
KEEP = "keep"


def _task_text(task, goal=None, tolerance=None, stop=KEEP, wait=None) -> dict:
    """A task as the world file would write it, with some parts changed."""
    item = {
        "robot": task.robot,
        "goal": [float(v) for v in (goal if goal is not None else task.goal.xyt)],
        "goal_tolerance": list(tolerance or task.goal_tolerance),
    }
    wait = task.wait if wait is None else wait
    if wait != 0:  # the file leaves a zero wait out
        item["wait"] = wait
    stop = task.stop if stop == KEEP else stop
    if stop is not None:
        item["stop"] = stop
    return item


def add_goal(world: World, name: str, goal: Sequence[float]) -> World:
    """Add a goal after the ones a robot has; several make waypoints."""
    _robot(world, name)
    task = _parse_task(
        {"robot": name, "goal": [float(v) for v in goal]},
        len(world.tasks) + 1,
        world.bounds,
        {r.name: r for r in world.robots},
    )
    return dataclasses.replace(world, tasks=(*world.tasks, task))


def move_goal(world: World, index: int, goal: Sequence[float]) -> World:
    """Move the goal at a 0-based place."""
    return _retask(world, index, goal=goal)


def set_goal_options(
    world: World,
    index: int,
    *,
    tolerance: Sequence[float] | None = None,
    stop: bool | None | str = KEEP,
    wait: float | None = None,
) -> World:
    """Set the tolerance, whether the robot must stop, and the wait of a goal.

    A part left out stays as it was. `stop` None is the default rule: stop at the
    last goal and at any goal with a wait. A `wait` of 0 is no wait.
    """
    return _retask(world, index, tolerance=tolerance, stop=stop, wait=wait)


def _retask(world: World, index: int, **changes) -> World:
    """Rebuild the task at a place with changes, checked as a file would be."""
    old = _task(world, index)
    task = _parse_task(
        _task_text(old, **changes),
        index + 1,
        world.bounds,
        {r.name: r for r in world.robots},
    )
    items = list(world.tasks)
    items[index] = task
    return dataclasses.replace(world, tasks=tuple(items))


def remove_goal(world: World, index: int) -> World:
    """Remove the goal at a 0-based place."""
    _task(world, index)
    return dataclasses.replace(
        world, tasks=tuple(t for i, t in enumerate(world.tasks) if i != index)
    )


def goals_of(world: World, name: str) -> list[int]:
    """The places in `world.tasks` of a robot's goals, in order."""
    return [i for i, t in enumerate(world.tasks) if t.robot == name]


def set_costmap(world: World, which: str, values: dict) -> World:
    """Change settings of the global or the local costmap.

    Args:
        world: the world to start from.
        which: "global" or "local".
        values: settings by name, such as `{"resolution": 0.1}`; the rest stay.
            Numbers are metres or seconds and must be above 0, `enabled` is
            true or false, `plugins` is a list, and `source` is "generate" or
            the path of a `map.yaml`.

    Raises:
        WorldError: if a setting is unknown or has a bad value.
    """
    if which not in ("global", "local"):
        raise WorldError(f"'{which}' is not a costmap; use global or local")
    old = world.costmap if which == "global" else world.local_costmap
    known = {f.name for f in dataclasses.fields(old)}
    label = f"costmap.{which}"
    for key, value in values.items():
        if key not in known:
            raise WorldError(f"{label}: no setting '{key}'; it has {', '.join(known)}")
        _check_costmap_value(label, key, value, getattr(old, key))
    values = {k: tuple(v) if k == "plugins" else v for k, v in values.items()}
    try:
        new = dataclasses.replace(old, **values)
        check_positive(new, *_COSTMAP_NUMBERS[which])
    except ValueError as error:
        raise WorldError(f"{label}: {error}") from error
    if which == "global":
        return dataclasses.replace(world, costmap=new)
    return dataclasses.replace(world, local_costmap=new)


_COSTMAP_NUMBERS = {
    "global": ("resolution", "inflation_radius", "cost_scaling_factor"),
    "local": LOCAL_POSITIVE,
}


def _check_costmap_value(label: str, key: str, value, current) -> None:
    """Refuse a costmap value of the wrong kind, or a missing map or bad layers."""
    if isinstance(current, bool):
        good = isinstance(value, bool)
    elif isinstance(current, float):
        good = isinstance(value, int | float) and not isinstance(value, bool)
        good = good and math.isfinite(value)
    elif isinstance(current, tuple):
        good = isinstance(value, list | tuple) and all(
            isinstance(v, str) for v in value
        )
    else:
        good = isinstance(value, str)
    if not good:
        raise WorldError(f"{label}: '{key}' has the wrong kind of value, {value!r}")
    if key == "source" and value != "generate" and not Path(value).is_file():
        raise WorldError(f"{label}: no map file '{value}'")
    if key == "unknown" and value not in ("free", "blocked"):
        raise WorldError(f"{label}: 'unknown' must be free or blocked")
    if key == "plugins":
        order = (
            ("static", "inflation")
            if label.endswith("global")
            else (
                "obstacle",
                "inflation",
            )
        )
        if list(value) != [p for p in order if p in value]:
            raise WorldError(f"{label}: 'plugins' must list {' and '.join(order)}")


def set_origin(world: World, origin: Sequence[float]) -> World:
    """Move the lower-left corner of the world, keeping its size.

    Raises:
        WorldError: if a robot, goal or obstacle would be outside.
    """
    x, y = numbers(list(origin), 2, "world", "origin")
    edited = dataclasses.replace(world, origin=(x, y))
    return _fits(edited)


def _fits(world: World) -> World:
    """The world, if everything lies inside it; else a plain error."""
    for obstacle in world.obstacles:
        if not inside(obstacle.shape.points(obstacle.pose), world.bounds):
            raise WorldError("world: an obstacle would fall outside")
    for spec in world.robots:
        if not inside(spec.shape.points(spec.pose), world.bounds):
            raise WorldError(f"robot '{spec.name}' would fall outside")
    for task in world.tasks:
        if not inside([tuple(task.goal.translation)], world.bounds):
            raise WorldError(f"the goal of '{task.robot}' would fall outside")
    return world
