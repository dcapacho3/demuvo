"""Small helpers that the edit functions of robots, obstacles and actors share."""

from __future__ import annotations

from .sim.actors import Actors
from .sim.collisions import Checker
from .verify import WorldError
from .world import World


def at(items: tuple, index: int, message: str):
    """The item at a 0-based place; the message, with its 1-based number, if none.

    Raises:
        WorldError: if there is no item there.
    """
    if not 0 <= index < len(items):
        raise WorldError(message.format(index + 1))
    return items[index]


def replaced(items: tuple, index: int, new) -> tuple:
    """The items with the one at a 0-based place replaced."""
    return (*items[:index], new, *items[index + 1 :])


def without(items: tuple, index: int) -> tuple:
    """The items but for the one at a 0-based place."""
    return tuple(item for i, item in enumerate(items) if i != index)


def no_overlap(world: World, mover: str) -> None:
    """Raise unless every robot is clear of the obstacles, the border and the others.

    A pair is named with the robot being edited, `mover`, first. An actor counts
    where it starts, so a robot is never put inside a solid one.
    """
    poses = {r.name: r.pose for r in world.robots}
    actors = Actors(world).poses_at(0.0)
    for contact in Checker(world).check(poses, actors=actors):
        if not contact.pushes:
            continue
        first, second = contact.a, contact.b
        if contact.b_moves and second == mover:
            first, second = second, first
        other = f"robot '{second}'" if contact.b_moves else second
        raise WorldError(f"robot '{first}' would overlap {other}")
