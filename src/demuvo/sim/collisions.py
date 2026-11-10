"""The collision checker.

The checker only reports. It does not move anything and it holds no policy.

Obstacles never move, so their parts go into a quadtree once, at load. Robots
change every step, so their boxes are compared all at once with numpy. Each body
also has a layer and a mask, a pair is tested only if each one's layer is in the
other's mask. The default is that everything meets everything.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
from robomath import Transform2D

from ..world import World
from ..world.shapes import Part, border_parts
from .gjk import touch
from .quadtree import Box, Quadtree

ALL = -1
# Overlaps lower than this are considered rounding errors.
MIN_DEPTH = 1e-9


@dataclass(frozen=True)
class Contact:
    """Two bodies that overlap.

    Attributes:
        a: name of a robot.
        b: name of what it touches: another robot, "obstacle N" (the Nth in the
            world file, from 1) or "border".
        normal: unit vector pointing from a toward b.
        depth: how far they overlap, in metres: the least distance that parts them.
        b_moves: True if b is a robot, which the caller may move too.
    """

    a: str
    b: str
    normal: tuple[float, float]
    depth: float
    b_moves: bool


@dataclass(frozen=True)
class Collision:
    """A pair that came into contact.

    Attributes:
        time: simulated time of the step, in seconds.
        a: name of a robot.
        b: name of what it hit: a robot, "obstacle N" or "border".
    """

    time: float
    a: str
    b: str


def _union(parts: list[Part]) -> Box:
    boxes = [p.box for p in parts]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _deepest(
    first: list[Part], second: list[Part]
) -> tuple[tuple[float, float], float] | None:
    """The deepest overlap between any part of one body and any of the other."""
    best, depth = None, MIN_DEPTH
    for p in first:
        for q in second:
            found = touch(p, q)
            if found is not None and found[1] > depth:
                best, depth = found, found[1]
    return best


class Checker:
    """Finds the overlaps among the robots and the fixed world.

    Args:
        world: the world; its obstacles and its border are fixed from here on.
        layers: optional (layer, mask) for a robot by name or an obstacle by
            "obstacle N"; a body not listed gets (1, ALL).
    """

    def __init__(
        self,
        world: World,
        layers: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        """Cut the fixed world into parts and put them in the quadtree."""
        self._shapes = {r.name: r.shape for r in world.robots}
        self._seen: dict[str, tuple[Transform2D, list[Part], Box]] = {}
        self._layers = dict(layers or {})
        self._fixed: list[tuple[str, Part]] = []
        for i, obstacle in enumerate(world.obstacles, start=1):
            for part in obstacle.shape.parts(obstacle.pose):
                self._fixed.append((f"obstacle {i}", part))
        self._fixed += [("border", part) for part in border_parts(world.bounds)]
        self._tree = Quadtree(world.bounds)
        for index, (_, part) in enumerate(self._fixed):
            self._tree.insert(index, part.box)

    def _meet(self, a: str, b: str) -> bool:
        layer_a, mask_a = self._layers.get(a, (1, ALL))
        layer_b, mask_b = self._layers.get(b, (1, ALL))
        return bool(layer_a & mask_b) and bool(layer_b & mask_a)

    def _body(self, name: str, pose: Transform2D) -> tuple[list[Part], Box]:
        """The parts of a robot at a pose and the box around them."""
        seen = self._seen.get(name)
        if seen is None or seen[0] is not pose:
            parts = self._shapes[name].parts(pose)
            seen = self._seen[name] = (pose, parts, _union(parts))
        return seen[1], seen[2]

    def check(
        self, poses: Mapping[str, Transform2D], moved: Collection[str] | None = None
    ) -> list[Contact]:
        """Every overlap among the robots at these poses, and with the world.

        Args:
            poses: the pose of each robot to test, by name; the same set of
                poses as one step proposes.
            moved: if given, only the contacts of these robots are looked for.
                Use it to test again after moving some robots, when the others
                were clear before. None tests all.

        Returns:
            One contact per pair of bodies, the deepest, in a fixed order:
            robot against robot first, then robot against the world.
        """
        names = list(poses)
        bodies = {n: self._body(n, poses[n]) for n in names}
        parts = {n: bodies[n][0] for n in names}
        boxes = np.array([bodies[n][1] for n in names]).reshape(-1, 4)
        contacts = []
        meets = (boxes[:, None, 0] <= boxes[None, :, 2]) & (
            boxes[None, :, 0] <= boxes[:, None, 2]
        )
        meets &= (boxes[:, None, 1] <= boxes[None, :, 3]) & (
            boxes[None, :, 1] <= boxes[:, None, 3]
        )
        if moved is not None:
            flags = np.array([n in moved for n in names])
            meets &= flags[:, None] | flags[None, :]
        for i, j in zip(*np.nonzero(np.triu(meets, 1))):
            a, b = names[i], names[j]
            if self._meet(a, b) and (found := _deepest(parts[a], parts[b])):
                contacts.append(Contact(a, b, found[0], found[1], True))
        for name in names if moved is None else [n for n in names if n in moved]:
            contacts += self._against_world(name, parts[name])
        return contacts

    def _against_world(self, name: str, parts: list[Part]) -> list[Contact]:
        """The deepest contact of one robot with each fixed body it touches."""
        deepest: dict[str, tuple[tuple[float, float], float]] = {}
        for part in parts:
            for index in self._tree.query(part.box):
                label, fixed = self._fixed[index]
                if not self._meet(name, label):
                    continue
                found = touch(part, fixed)
                if found and found[1] > MIN_DEPTH:
                    if label not in deepest or found[1] > deepest[label][1]:
                        deepest[label] = found
        return [
            Contact(name, label, normal, depth, False)
            for label, (normal, depth) in deepest.items()
        ]
