"""The collision checker.

The checker only reports. It does not move anything and it holds no policy.

Obstacles never move, so their parts go into a quadtree once, at load. Robots
change every step, so their boxes are compared all at once with numpy. Each body
also has a layer and a mask, a pair is tested only if each one's layer is in the
other's mask. The default is that everything meets everything.

Actors move too, and are given to `check` beside the robots. An actor is never
moved by what it touches: its pose comes from its motion alone. A robot that
meets a `solid` actor is pushed out of it, a robot that meets one that is not
`solid` passes through and the contact is only recorded, and two actors are
tested only if both carry `collide`, and then neither is moved. An actor is not
tested against the obstacles or the border: where its motion takes it is the
world file's business, not the checker's.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import lru_cache

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
        pushes: True if the overlap must be cleared, which is the usual case;
            False records it and leaves both where they are, as for a robot
            inside an actor that is not `solid` and for two actors.
    """

    a: str
    b: str
    normal: tuple[float, float]
    depth: float
    b_moves: bool
    pushes: bool = True


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


def _box_pairs(boxes: np.ndarray, names: list[str], moved):
    """The index pairs of bodies whose boxes overlap, those of `moved` only if given."""
    meets = (boxes[:, None, 0] <= boxes[None, :, 2]) & (
        boxes[None, :, 0] <= boxes[:, None, 2]
    )
    meets &= (boxes[:, None, 1] <= boxes[None, :, 3]) & (
        boxes[None, :, 1] <= boxes[:, None, 3]
    )
    if moved is not None:
        flags = np.array([n in moved for n in names])
        meets &= flags[:, None] | flags[None, :]
    rows, cols = _upper_pairs(len(names))
    hits = meets[rows, cols]
    return rows[hits], cols[hits]


@lru_cache(maxsize=16)
def _upper_pairs(count: int) -> tuple[np.ndarray, np.ndarray]:
    """The i < j index pairs of a count-by-count matrix, kept between steps.

    `np.triu` builds a fresh mask every call, which a step loop pays every
    step; these pairs are read only and shared.
    """
    return np.triu_indices(count, 1)


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
        world: the world; its obstacles and its border are fixed from here on,
            and its actors are the bodies `check` may be given beside the robots.
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
        self._actors = {a.name: a for a in world.actors}
        self._shapes |= {a.name: a.shape for a in world.actors}
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

    def _pair(self, a: str, b: str) -> tuple[bool, bool] | None:
        """Whether to test these two, and then `b_moves` and `pushes` of the contact.

        `a` is a robot unless both are actors, since `check` lists the robots
        first and takes its pairs from the upper triangle.
        """
        second = self._actors.get(b)
        if second is None:  # two robots: either may be pushed
            return True, True
        if a not in self._actors:  # a robot against an actor
            return False, second.solid
        if self._actors[a].collide and second.collide:  # two actors: recorded only
            return False, False
        return None

    def _body(self, name: str, pose: Transform2D) -> tuple[list[Part], Box]:
        """The parts of a robot at a pose and the box around them."""
        seen = self._seen.get(name)
        if seen is None or seen[0] is not pose:
            parts = self._shapes[name].parts(pose)
            seen = self._seen[name] = (pose, parts, _union(parts))
        return seen[1], seen[2]

    def check(
        self,
        poses: Mapping[str, Transform2D],
        moved: Collection[str] | None = None,
        actors: Mapping[str, Transform2D] | None = None,
    ) -> list[Contact]:
        """Every overlap among the robots and the actors, and with the world.

        Args:
            poses: the pose of each robot to test, by name; the same set of
                poses as one step proposes.
            moved: if given, only the contacts of these robots are looked for.
                Use it to test again after moving some robots, when the others
                were clear before. None tests all. An actor never moves between
                two calls of one step, so it need not be listed.
            actors: the pose of each actor to test, by name, or None for none.
                An actor is tested against the robots, and against another actor
                when both carry `collide`.

        Returns:
            One contact per pair of bodies, the deepest, in a fixed order:
            robot against robot and against actor first, then robot against the
            world. A contact whose `pushes` is False is reported but must be
            left as it is.
        """
        all_poses = {**poses, **(actors or {})}
        names = list(all_poses)
        bodies = {n: self._body(n, all_poses[n]) for n in names}
        parts = {n: bodies[n][0] for n in names}
        boxes = np.array([bodies[n][1] for n in names]).reshape(-1, 4)
        contacts = []
        rows, cols = _box_pairs(boxes, names, moved)
        for i, j in zip(rows, cols):
            a, b = names[i], names[j]
            rule = self._pair(a, b)
            if rule is None or not self._meet(a, b):
                continue
            if found := _deepest(parts[a], parts[b]):
                contacts.append(Contact(a, b, found[0], found[1], *rule))
        robots = poses if moved is None else [n for n in poses if n in moved]
        for name in robots:
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
