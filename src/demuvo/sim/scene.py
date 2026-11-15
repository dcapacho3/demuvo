"""The scene - the segments and circles a ray can hit, built from the world.

There are three kinds of body in it. The border and the obstacles never move and
are built once, as `Scene.FIXED`. A robot's body has its index in `world.robots`
as its owner, so its own sensors can leave it out. An actor's body has its index
in `world.actors` plus the number of robots, so it belongs to no robot and every
robot's lidar sees it; `solid` says whether a robot is pushed out of an actor and
has nothing to do with being seen.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..sensing import Scene
from ..world import World
from ..world.shapes import Part, border_parts


def _part_arrays(part: Part) -> tuple[list[list[float]], list[list[float]]]:
    """The segments and circles that make up the outline of one convex part.

    A part is a polygon swelled by its radius. Its outline is each edge moved
    out by the radius, and a circle of that radius at each corner. A ray from
    outside meets the part where it first meets one of these.
    """
    vertices = np.asarray(part.vertices, dtype=float)
    radius = part.radius
    if len(vertices) == 1:
        return [], ([[*vertices[0], radius]] if radius > 0 else [])
    count = len(vertices) if len(vertices) > 2 else 1  # two corners make one edge
    segments = []
    for k in range(count):
        a, b = vertices[k], vertices[(k + 1) % len(vertices)]
        length = float(np.hypot(*(b - a)))
        if length == 0:
            continue
        if radius == 0:
            segments.append([*a, *b])
            continue
        out = np.array([b[1] - a[1], a[0] - b[0]]) / length * radius
        segments.append([*(a + out), *(b + out)])
        if len(vertices) == 2:  # a swelled wall is open on both sides
            segments.append([*(a - out), *(b - out)])
    circles = [[*vertex, radius] for vertex in vertices] if radius > 0 else []
    return segments, circles


def parts_scene(bodies: Sequence[tuple[int, Sequence[Part]]]) -> Scene:
    """A scene from bodies, each an owner and the convex parts it is made of."""
    segments, circles, segment_owner, circle_owner = [], [], [], []
    for owner, parts in bodies:
        for part in parts:
            new_segments, new_circles = _part_arrays(part)
            segments += new_segments
            segment_owner += [owner] * len(new_segments)
            circles += new_circles
            circle_owner += [owner] * len(new_circles)
    return Scene(
        np.array(segments, dtype=float).reshape(-1, 4),
        np.array(segment_owner, dtype=int),
        np.array(circles, dtype=float).reshape(-1, 3),
        np.array(circle_owner, dtype=int),
    )


def static_scene(world: World) -> Scene:
    """The border and the obstacles of the world, which never move."""
    parts = border_parts(world.bounds)
    for obstacle in world.obstacles:
        parts += obstacle.shape.parts(obstacle.pose)
    return parts_scene([(Scene.FIXED, parts)])


def actor_owner(robots: int, index: int) -> int:
    """The owner of the actor at `index` in a world with this many robots.

    Actors are numbered after the robots, so a robot's own index still names its
    own body and no actor can ever be skipped as if it were one.
    """
    return robots + index


def with_robots(
    static: Scene,
    bodies: Sequence[Sequence[Part]],
    actors: Sequence[Sequence[Part]] = (),
) -> Scene:
    """The static scene plus the body of each robot, then of each actor.

    A robot's owner is its index; an actor's is `actor_owner`, which is its own
    index after the robots.
    """
    moving = parts_scene(
        list(enumerate(bodies))
        + [(actor_owner(len(bodies), i), parts) for i, parts in enumerate(actors)]
    )
    return Scene(
        np.concatenate([static.segments, moving.segments]),
        np.concatenate([static.segment_owner, moving.segment_owner]),
        np.concatenate([static.circles, moving.circles]),
        np.concatenate([static.circle_owner, moving.circle_owner]),
    )
