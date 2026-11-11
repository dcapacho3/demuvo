"""Writing a world back to a file - the inverse of `load_world`.

Python has no TOML writer in its standard library, so this one writes the few
kinds of value a world holds: numbers, text, true and false, lists and tables.
Comments and the order of keys in the original file are not kept. A pose is
written to 12 decimals, so a file reads back to the same pose within 1e-12.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import fields
from pathlib import Path

from robomath import Transform2D

from .shapes import Shape
from .spec import (
    DEFAULT_CONTROL_PERIOD,
    DEFAULT_DT,
    DEFAULT_GOAL_TOLERANCE,
    CostmapSpec,
    LocalCostmapSpec,
    RobotSpec,
    Task,
    World,
)

DIGITS = 12
BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _key(name: str) -> str:
    """A key as TOML wants it: bare if it can be, else quoted."""
    return name if BARE_KEY.fullmatch(name) else json.dumps(name)


def _value(value) -> str:
    """A Python value as TOML text; a table becomes an inline table."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items())
            + " }"
        )
    raise TypeError(f"cannot write a {type(value).__name__} to a world file")


def _pose(pose: Transform2D) -> list[float]:
    """(x, y, angle) of a pose, rounded so a saved file stays tidy."""
    return [round(float(v), DIGITS) for v in pose.xyt]


def _shape(shape: Shape) -> dict:
    """The `shape` table of a shape: its name, then its own fields."""
    return {"name": shape.name} | {
        f.name: getattr(shape, f.name) for f in fields(shape)
    }


def _lines(table: dict) -> list[str]:
    """One `key = value` line for each entry of a table."""
    return [f"{_key(k)} = {_value(v)}" for k, v in table.items()]


def _global(spec: CostmapSpec, base: Path) -> dict:
    """The `[costmap.global]` table, with only what differs from the defaults."""
    out = {"enabled": spec.enabled, "resolution": spec.resolution}
    out["plugins"] = list(spec.plugins)
    if "static" in spec.plugins and spec.source != "generate":
        out["static"] = {
            "source": os.path.relpath(spec.source, base),
            "unknown": spec.unknown,
        }
    if "inflation" in spec.plugins:
        out["inflation"] = {
            "inflation_radius": spec.inflation_radius,
            "cost_scaling_factor": spec.cost_scaling_factor,
        }
    return out


def _local(spec: LocalCostmapSpec) -> dict:
    """The `[costmap.local]` table."""
    out = {
        "enabled": spec.enabled,
        "window": spec.window,
        "resolution": spec.resolution,
        "period": spec.period,
        "plugins": list(spec.plugins),
    }
    if "obstacle" in spec.plugins:
        out["obstacle"] = {
            "obstacle_max_range": spec.obstacle_max_range,
            "raytrace_max_range": spec.raytrace_max_range,
        }
    if "inflation" in spec.plugins:
        out["inflation"] = {
            "inflation_radius": spec.inflation_radius,
            "cost_scaling_factor": spec.cost_scaling_factor,
        }
    return out


def _robot(spec: RobotSpec) -> list[str]:
    """The `[[robot]]` block of a robot; a key left at its default is left out."""
    out = {
        "name": spec.name,
        "model": spec.model,
        "shape": _shape(spec.shape),
        "pose": _pose(spec.pose),
    }
    if spec.params:
        out["params"] = spec.params
    for part in ("planner", "smoother", "controller"):
        if getattr(spec, part):
            out[part] = getattr(spec, part)
            if getattr(spec, f"{part}_params"):
                out[f"{part}_params"] = getattr(spec, f"{part}_params")
    if spec.control_period != DEFAULT_CONTROL_PERIOD:
        out["control_period"] = spec.control_period
    if spec.estimator != "ground_truth":
        out["estimator"] = spec.estimator
    if spec.control_pose != "estimate":
        out["control_pose"] = spec.control_pose
    if spec.sensors:
        out["sensors"] = spec.sensors
    return ["[[robot]]", *_lines(out), ""]


def _task(task: Task) -> list[str]:
    """The `[[task]]` block of a task."""
    out = {"robot": task.robot, "goal": _pose(task.goal)}
    if task.goal_tolerance != DEFAULT_GOAL_TOLERANCE:
        out["goal_tolerance"] = list(task.goal_tolerance)
    if task.stop is not None:
        out["stop"] = task.stop
    if task.wait:
        out["wait"] = task.wait
    return ["[[task]]", *_lines(out), ""]


def world_text(world: World, base: Path) -> str:
    """The text of a world file for this world.

    Args:
        world: the world to write.
        base: the folder the file will be in; a map file is written as a path
            relative to it.
    """
    world_table = {"size": list(world.size), "seed": world.seed}
    if world.origin != (0.0, 0.0):
        world_table["origin"] = list(world.origin)
    if world.dt != DEFAULT_DT:
        world_table["dt"] = world.dt
    lines = ["version = 1", "", "[world]", *_lines(world_table), ""]
    for obstacle in world.obstacles:
        table = {"pose": _pose(obstacle.pose), "shape": _shape(obstacle.shape)}
        if obstacle.name:
            table = {"name": obstacle.name, **table}
        lines += ["[[obstacle]]", *_lines(table), ""]
    for spec in world.robots:
        lines += _robot(spec)
    for task in world.tasks:
        lines += _task(task)
    if world.sensors:
        lines += ["[sensors]", *_lines(world.sensors), ""]
    if world.costmap != CostmapSpec():
        lines += ["[costmap.global]", *_lines(_global(world.costmap, base)), ""]
    if world.local_costmap != LocalCostmapSpec():
        lines += ["[costmap.local]", *_lines(_local(world.local_costmap)), ""]
    return "\n".join(lines)


def save_world(world: World, path: str | Path) -> None:
    """Write a world to a TOML file that `load_world` reads back.

    The file has the same robots, obstacles, tasks and settings. Comments of an
    earlier file are lost, and a pose is written to 12 decimals.

    Args:
        world: the world to write.
        path: where to write it; an existing file is replaced.
    """
    path = Path(path)
    path.write_text(world_text(world, path.resolve().parent), encoding="utf-8")
