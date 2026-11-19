"""Editing the actors of a world; `demuvo.edit` re-exports these functions."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from . import options
from .edit_core import at, replaced, without
from .sim.actors import Actors
from .verify import WorldError, numbers
from .world import ActorSpec, World
from .world.load import _parse_actor

__all__ = [
    "ACTOR_RADIUS",
    "MOTION_START",
    "actor_names",
    "add_actor",
    "move_actor",
    "remove_actor",
    "rename_actor",
    "resize_actor",
    "set_actor_flag",
    "set_actor_motion",
    "set_actor_motion_params",
]

# What an actor added in a window is like: a disc that stands still.
ACTOR_RADIUS = 0.25
# The settings of a motion that a form cannot fill in, and what to start them with.
MOTION_START = {"points": [[1.0, 0.0]]}


def _actor(world: World, index: int) -> ActorSpec:
    """The actor at a 0-based place.

    Raises:
        WorldError: if there is no actor there.
    """
    return at(world.actors, index, "there is no actor {}")


def _with_actor(world: World, index: int, item: dict) -> World:
    """The world with the actor at a place replaced by what this table describes.

    Raises:
        WorldError: if the table is wrong, or its motion will not build.
    """
    names = {a.name for i, a in enumerate(world.actors) if i != index}
    names |= {r.name for r in world.robots}
    made = _parse_actor(item, index + 1, world.bounds, names, world.actor_defaults)
    edited = dataclasses.replace(world, actors=replaced(world.actors, index, made))
    Actors(edited)  # builds every motion, and says which actor refused its settings
    return edited


def _actor_table(spec: ActorSpec) -> dict:
    """The `[[actor]]` table of an actor, as the loader reads it back."""
    item = {
        "name": spec.name,
        "shape": {"name": spec.shape.name}
        | {f.name: getattr(spec.shape, f.name) for f in dataclasses.fields(spec.shape)},
        "pose": [float(v) for v in spec.pose.xyt],
        "solid": spec.solid,
        "collide": spec.collide,
    }
    if spec.motion:
        item["motion"] = spec.motion
        item["motion_params"] = dict(spec.motion_params)
    return item


def actor_names(world: World) -> set[str]:
    """Every name already taken, by a robot or by an actor."""
    return {r.name for r in world.robots} | {a.name for a in world.actors}


def add_actor(
    world: World,
    name: str,
    pose: Sequence[float],
    *,
    radius: float = ACTOR_RADIUS,
    motion: str | None = None,
    motion_params: dict | None = None,
) -> World:
    """Add an actor with a disc body, standing still unless a motion is given.

    Args:
        world: the world to start from.
        name: a name no robot and no other actor has.
        pose: (x, y, angle) where it starts.
        radius: radius of its body, in metres.
        motion: "module:Class" of its motion, or None to stand still.
        motion_params: the settings of that motion.

    Raises:
        WorldError: if the name is taken or the actor would leave the world.
    """
    item = {
        "name": name,
        "shape": {"name": "circle", "radius": radius},
        "pose": list(pose),
    }
    if motion:
        item["motion"] = motion
        item["motion_params"] = dict(motion_params or {})
    made = _parse_actor(
        item,
        len(world.actors) + 1,
        world.bounds,
        actor_names(world),
        world.actor_defaults,
    )
    added = dataclasses.replace(world, actors=(*world.actors, made))
    Actors(added)  # builds the motion, so a bad setting never reaches a world
    return added


def remove_actor(world: World, index: int) -> World:
    """Remove the actor at a 0-based place."""
    _actor(world, index)
    return dataclasses.replace(world, actors=without(world.actors, index))


def move_actor(world: World, index: int, pose: Sequence[float]) -> World:
    """Move the actor at a 0-based place to a pose.

    Raises:
        WorldError: if the pose would take it outside the world.
    """
    spec = _actor(world, index)
    label = f"actor '{spec.name}'"
    moved = numbers(list(pose), 3, label, "pose")
    return _with_actor(world, index, _actor_table(spec) | {"pose": list(moved)})


def resize_actor(world: World, index: int, size: Sequence[float]) -> World:
    """Give the actor at a 0-based place a disc or box body of this size.

    Args:
        world: the world to start from.
        index: the 0-based place of the actor.
        size: (radius,) for a disc, or (length, width) for a box.
    """
    spec = _actor(world, index)
    shape = (
        {"name": "circle", "radius": size[0]}
        if len(size) == 1
        else {"name": "box", "size": [float(v) for v in size]}
    )
    return _with_actor(world, index, _actor_table(spec) | {"shape": shape})


def rename_actor(world: World, index: int, new: str) -> World:
    """Give the actor at a 0-based place another name.

    Raises:
        WorldError: if the name is empty or another body already has it.
    """
    spec = _actor(world, index)
    name = new.strip()
    if not name:
        raise WorldError(f"actor '{spec.name}': a name cannot be empty")
    return _with_actor(world, index, _actor_table(spec) | {"name": name})


def set_actor_flag(world: World, index: int, key: str, on: bool) -> World:
    """Set `solid` or `collide` of the actor at a 0-based place."""
    spec = _actor(world, index)
    if key not in ("solid", "collide"):
        raise WorldError(f"actor '{spec.name}': '{key}' is not solid or collide")
    return _with_actor(world, index, _actor_table(spec) | {key: bool(on)})


def set_actor_motion(world: World, index: int, path: str | None) -> World:
    """Give the actor at a 0-based place a motion, or None to stand still.

    The settings a form cannot fill in, such as the points of a path, are started
    off relative to where the actor stands, so the motion runs as soon as it is
    chosen.
    """
    spec = _actor(world, index)
    item = _actor_table(spec)
    item.pop("motion", None)
    item.pop("motion_params", None)
    if path is not None:
        item["motion"] = path
        item["motion_params"] = _started_params(spec, path)
    return _with_actor(world, index, item)


def _started_params(spec: ActorSpec, path: str) -> dict:
    """The settings a motion starts with: the ones it had, then what it still needs.

    A form shows numbers and switches, so a setting it cannot show, such as the
    points of a path, is started off here, relative to where the actor stands.
    """
    takes = options.arguments_of(path)
    params = {k: v for k, v in spec.motion_params.items() if k in takes}
    x, y, _ = spec.pose.xyt
    for key, value in MOTION_START.items():
        if key in takes and key not in params:
            params[key] = [[float(x) + dx, float(y) + dy] for dx, dy in value]
    return params


def set_actor_motion_params(world: World, index: int, settings: dict) -> World:
    """Change the settings of the motion of the actor at a 0-based place.

    Raises:
        WorldError: if the actor has no motion, or a setting is wrong.
    """
    spec = _actor(world, index)
    if spec.motion is None:
        raise WorldError(f"actor '{spec.name}': it has no motion to set")
    params = dict(spec.motion_params) | dict(settings)
    item = _actor_table(spec) | {"motion_params": params}
    return _with_actor(world, index, item)
